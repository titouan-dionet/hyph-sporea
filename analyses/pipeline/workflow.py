"""
Module de gestion du workflow pour le projet HYPH-SPOREA.

Ce module fournit des fonctionnalités équivalentes au package 'targets' de R pour Python,
permettant de définir, visualiser et exécuter des workflows de traitement de données.
"""

import os
import time
import json
import inspect
import hashlib
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from functools import wraps
import concurrent.futures
from datetime import datetime


class Target:
    """
    Classe représentant une cible (target) dans le workflow.
    
    Une cible encapsule une fonction avec ses entrées et sorties,
    et fournit des mécanismes pour la vérification de l'invalidation.
    
    Attributes:
        name (str): Nom de la cible
        func (callable): Fonction à exécuter
        inputs (list): Liste des cibles d'entrée
        file_inputs (list): Liste des fichiers d'entrée
        file_outputs (list): Liste des fichiers de sortie
        metadata_path (Path): Chemin vers le fichier de métadonnées
        status (str): État d'exécution ('pending', 'running', 'completed', 'failed')
        result: Résultat de l'exécution
        error: Erreur survenue lors de l'exécution
        elapsed_time (float): Temps d'exécution en secondes
    """
    
    def __init__(self, name, func, inputs=None, file_inputs=None, file_outputs=None, metadata_dir=None):
        """
        Initialise une cible du workflow.
        
        Args:
            name (str): Nom unique de la cible
            func (callable): Fonction à exécuter
            inputs (list, optional): Liste des cibles d'entrée
            file_inputs (list, optional): Liste des fichiers d'entrée
            file_outputs (list, optional): Liste des fichiers de sortie
            metadata_dir (str, optional): Répertoire pour les métadonnées
        """
        self.name = name
        self.func = func
        self.inputs = inputs or []
        self.file_inputs = [Path(f) for f in (file_inputs or [])]
        self.file_outputs = [Path(f) for f in (file_outputs or [])]
        
        # Configuration du répertoire de métadonnées
        if metadata_dir is None:
            metadata_dir = Path('.workflow') / 'metadata'
        else:
            metadata_dir = Path(metadata_dir)
        
        os.makedirs(metadata_dir, exist_ok=True)
        self.metadata_path = metadata_dir / f"{name.replace(' ', '_')}.json"
        
        # État d'exécution
        self.status = 'pending'
        self.result = None
        self.error = None
        self.elapsed_time = None
    
    def is_invalidated(self):
        """
        Vérifie si la cible doit être réexécutée.
        
        Une cible est invalidée si:
        - Son fichier de métadonnées n'existe pas
        - Les fichiers de sortie spécifiés n'existent pas
        - Les fichiers d'entrée ont été modifiés depuis la dernière exécution
        - Le code de la fonction a changé
        - L'une des cibles d'entrée a été invalidée
        
        Returns:
            bool: True si la cible doit être réexécutée, False sinon
        """
        # Vérifier si le fichier de métadonnées existe
        if not self.metadata_path.exists():
            return True
        
        # Charger les métadonnées
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception:
            return True
        
        # Vérifier si les fichiers de sortie existent
        for file_path in self.file_outputs:
            if not file_path.exists():
                return True
        
        # Vérifier si les fichiers d'entrée ont été modifiés
        for file_path in self.file_inputs:
            if file_path.exists():
                mtime = os.path.getmtime(file_path)
                if 'file_mtimes' not in metadata or str(file_path) not in metadata['file_mtimes']:
                    return True
                if mtime > metadata['file_mtimes'][str(file_path)]:
                    return True
        
        # Vérifier si le code de la fonction a changé
        func_hash = self._get_function_hash()
        if 'function_hash' not in metadata or metadata['function_hash'] != func_hash:
            return True
        
        # Vérifier si les cibles d'entrée sont invalidées
        for input_target in self.inputs:
            if input_target.is_invalidated():
                return True
        
        return False
    
    def execute(self, force=False):
        """
        Exécute la cible si elle est invalidée ou si force est True.
        
        Args:
            force (bool, optional): Si True, exécute la cible même si elle n'est pas invalidée.
                Par défaut False.
        
        Returns:
            Any: Résultat de l'exécution de la fonction
        """
        # Vérifier si la cible doit être exécutée
        if not force and not self.is_invalidated():
            print(f"Cible '{self.name}' à jour, pas d'exécution nécessaire.")
            
            # Charger le résultat précédent si disponible
            result_path = Path(str(self.metadata_path).replace('.json', '.pickle'))
            if result_path.exists():
                try:
                    with open(result_path, 'rb') as f:
                        self.result = pickle.load(f)
                except Exception as e:
                    print(f"Erreur lors du chargement du résultat précédent: {str(e)}")
            
            self.status = 'completed'
            return self.result
        
        # Exécuter les cibles d'entrée
        for input_target in self.inputs:
            input_target.execute()
        
        # Exécuter la fonction
        self.status = 'running'
        start_time = time.time()
        
        try:
            print(f"Exécution de la cible '{self.name}'...")
            
            # Appeler la fonction
            if self.inputs:
                # Passer les résultats des cibles d'entrée
                input_results = [input_target.result for input_target in self.inputs]
                self.result = self.func(*input_results)
            else:
                self.result = self.func()
            
            # Sauvegarder les métadonnées
            self._save_metadata()
            
            # Sauvegarder le résultat
            result_path = Path(str(self.metadata_path).replace('.json', '.pickle'))
            with open(result_path, 'wb') as f:
                pickle.dump(self.result, f)
            
            self.status = 'completed'
            print(f"Cible '{self.name}' terminée avec succès.")
        
        except Exception as e:
            self.error = e
            self.status = 'failed'
            print(f"Erreur lors de l'exécution de la cible '{self.name}': {str(e)}")
            raise
        
        finally:
            self.elapsed_time = time.time() - start_time
        
        return self.result
    
    def _get_function_hash(self):
        """
        Calcule un hash du code source de la fonction.
        
        Returns:
            str: Hash du code source
        """
        # Obtenir le code source de la fonction
        try:
            source = inspect.getsource(self.func)
        except (IOError, TypeError):
            # Recourir à une méthode alternative si le code source n'est pas disponible
            source = str(self.func.__code__.co_code)
        
        # Créer un hash du code source
        return hashlib.md5(source.encode()).hexdigest()
    
    def _save_metadata(self):
        """
        Sauvegarde les métadonnées de la cible.
        """
        # Collecter les temps de modification des fichiers d'entrée
        file_mtimes = {}
        for file_path in self.file_inputs:
            if file_path.exists():
                file_mtimes[str(file_path)] = os.path.getmtime(file_path)
        
        # Créer les métadonnées
        metadata = {
            'name': self.name,
            'function_hash': self._get_function_hash(),
            'file_mtimes': file_mtimes,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': self.elapsed_time
        }
        
        # Sauvegarder les métadonnées
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


class Workflow:
    """
    Classe pour la gestion d'un workflow complet.
    
    Un workflow est constitué d'un ensemble de cibles (targets) interconnectées
    qui peuvent être exécutées dans l'ordre approprié.
    
    Attributes:
        name (str): Nom du workflow
        targets (dict): Dictionnaire des cibles par nom
        metadata_dir (Path): Répertoire pour les métadonnées
    """
    
    def __init__(self, name, metadata_dir=None):
        """
        Initialise un nouveau workflow.
        
        Args:
            name (str): Nom du workflow
            metadata_dir (str, optional): Répertoire pour les métadonnées
        """
        self.name = name
        self.targets = {}
        
        # Configuration du répertoire de métadonnées
        if metadata_dir is None:
            metadata_dir = Path('.workflow') / name
        else:
            metadata_dir = Path(metadata_dir) / name
        
        os.makedirs(metadata_dir, exist_ok=True)
        self.metadata_dir = metadata_dir
    
    def add_target(self, name, func, inputs=None, file_inputs=None, file_outputs=None):
        """
        Ajoute une cible au workflow.
        
        Args:
            name (str): Nom unique de la cible
            func (callable): Fonction à exécuter
            inputs (list, optional): Liste des noms des cibles d'entrée
            file_inputs (list, optional): Liste des fichiers d'entrée
            file_outputs (list, optional): Liste des fichiers de sortie
        
        Returns:
            Target: La cible créée
        """
        # Convertir les noms des cibles d'entrée en objets Target
        input_targets = []
        if inputs:
            for input_name in inputs:
                if input_name not in self.targets:
                    raise ValueError(f"La cible d'entrée '{input_name}' n'existe pas")
                input_targets.append(self.targets[input_name])
        
        # Créer la cible
        target = Target(
            name=name,
            func=func,
            inputs=input_targets,
            file_inputs=file_inputs,
            file_outputs=file_outputs,
            metadata_dir=self.metadata_dir
        )
        
        # Ajouter la cible au workflow
        self.targets[name] = target
        
        return target
    
    def execute(self, target_name=None, force=False, parallel=False):
        """
        Exécute une ou toutes les cibles du workflow.
        
        Args:
            target_name (str, optional): Nom de la cible à exécuter.
                Si None, exécute toutes les cibles.
            force (bool, optional): Si True, exécute les cibles même si elles ne sont pas invalidées.
                Par défaut False.
            parallel (bool, optional): Si True, exécute les cibles en parallèle quand c'est possible.
                Par défaut False.
        
        Returns:
            dict: Résultats d'exécution de chaque cible
        """
        results = {}
        
        if target_name:
            # Exécuter une cible spécifique
            if target_name not in self.targets:
                raise ValueError(f"La cible '{target_name}' n'existe pas")
            
            target = self.targets[target_name]
            results[target_name] = target.execute(force=force)
        
        elif parallel:
            # Exécution parallèle (expérimental)
            # Construire le graphe de dépendances
            graph = self._build_dependency_graph()
            
            # Déterminer les niveaux d'exécution
            levels = []
            remaining = set(self.targets.keys())
            
            while remaining:
                # Trouver les cibles qui n'ont plus de dépendances
                level = []
                for name in list(remaining):
                    dependencies = set()
                    for dep_name in graph.predecessors(name):
                        if dep_name in remaining:
                            dependencies.add(dep_name)
                    
                    if not dependencies:
                        level.append(name)
                        remaining.remove(name)
                
                if not level:
                    # Circuit détecté dans le graphe
                    raise ValueError("Circuit détecté dans le graphe de dépendances")
                
                levels.append(level)
            
            # Exécuter chaque niveau en parallèle
            for level in levels:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {}
                    for name in level:
                        target = self.targets[name]
                        if force or target.is_invalidated():
                            futures[executor.submit(target.execute, force)] = name
                    
                    for future in concurrent.futures.as_completed(futures):
                        name = futures[future]
                        try:
                            results[name] = future.result()
                        except Exception as e:
                            print(f"Erreur lors de l'exécution de la cible '{name}': {str(e)}")
                            raise
        
        else:
            # Exécution séquentielle de toutes les cibles
            for name, target in self.targets.items():
                results[name] = target.execute(force=force)
        
        return results
    
    def visualize(self, output_path=None, show=True):
        """
        Visualise le workflow sous forme de graphe orienté.
        
        Args:
            output_path (str, optional): Chemin pour sauvegarder l'image
            show (bool, optional): Si True, affiche le graphe. Par défaut True.
        
        Returns:
            matplotlib.figure.Figure: Figure créée
        """
        # Construire le graphe
        G = self._build_dependency_graph()
        
        # Créer la figure
        plt.figure(figsize=(12, 8))
        
        # Positionner les nœuds
        pos = nx.spring_layout(G)
        
        # Déterminer les couleurs des nœuds selon leur état
        node_colors = []
        for name in G.nodes():
            target = self.targets[name]
            if target.status == 'completed':
                node_colors.append('green')
            elif target.status == 'running':
                node_colors.append('orange')
            elif target.status == 'failed':
                node_colors.append('red')
            else:
                # Vérifier si la cible est invalidée
                if target.is_invalidated():
                    node_colors.append('lightcoral')
                else:
                    node_colors.append('lightblue')
        
        # Dessiner les nœuds
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        
        # Ajouter les labels
        nx.draw_networkx_labels(G, pos, font_weight='bold')
        
        plt.title(f"Workflow: {self.name}")
        plt.axis('off')
        
        # Ajouter une légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', label='À jour'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Invalidé'),
            Patch(facecolor='green', edgecolor='black', label='Complété'),
            Patch(facecolor='orange', edgecolor='black', label='En cours'),
            Patch(facecolor='red', edgecolor='black', label='Échoué')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Sauvegarder l'image
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Afficher l'image
        if show:
            plt.show()
        
        return plt.gcf()
    
    def _build_dependency_graph(self):
        """
        Construit un graphe orienté des dépendances entre les cibles.
        
        Returns:
            networkx.DiGraph: Graphe de dépendances
        """
        G = nx.DiGraph()
        
        # Ajouter les nœuds
        for name in self.targets:
            G.add_node(name)
        
        # Ajouter les arêtes
        for name, target in self.targets.items():
            for input_target in target.inputs:
                G.add_edge(input_target.name, name)
        
        return G
    
    def clean(self, target_name=None):
        """
        Supprime les métadonnées et les fichiers de résultats d'une ou toutes les cibles.
        
        Args:
            target_name (str, optional): Nom de la cible à nettoyer.
                Si None, nettoie toutes les cibles.
        """
        if target_name:
            # Nettoyer une cible spécifique
            if target_name not in self.targets:
                raise ValueError(f"La cible '{target_name}' n'existe pas")
            
            target = self.targets[target_name]
            self._clean_target(target)
        else:
            # Nettoyer toutes les cibles
            for target in self.targets.values():
                self._clean_target(target)
    
    def _clean_target(self, target):
        """
        Supprime les métadonnées et les fichiers de résultats d'une cible.
        
        Args:
            target (Target): Cible à nettoyer
        """
        # Supprimer le fichier de métadonnées
        if target.metadata_path.exists():
            os.remove(target.metadata_path)
        
        # Supprimer le fichier de résultat
        result_path = Path(str(target.metadata_path).replace('.json', '.pickle'))
        if result_path.exists():
            os.remove(result_path)
        
        # Réinitialiser l'état de la cible
        target.status = 'pending'
        target.result = None
        target.error = None
        target.elapsed_time = None


def target(workflow, name, inputs=None, file_inputs=None, file_outputs=None):
    """
    Décorateur pour ajouter facilement une fonction comme cible à un workflow.
    
    Args:
        workflow (Workflow): Workflow auquel ajouter la cible
        name (str): Nom de la cible
        inputs (list, optional): Liste des noms des cibles d'entrée
        file_inputs (list, optional): Liste des fichiers d'entrée
        file_outputs (list, optional): Liste des fichiers de sortie
    
    Returns:
        callable: Décorateur
    
    Example:
        >>> workflow = Workflow("mon_workflow")
        >>> @target(workflow, "ma_cible", file_outputs=["output.csv"])
        >>> def process_data():
        ...     # Traitement des données
        ...     return data
    """
    def decorator(func):
        workflow.add_target(
            name=name,
            func=func,
            inputs=inputs,
            file_inputs=file_inputs,
            file_outputs=file_outputs
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator