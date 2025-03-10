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

# Ajout des imports pour les nouvelles fonctionnalités
import shutil
import pandas as pd
from ..models.sam_model import load_sam_model, batch_detect_with_sam, combine_sam_detections
from ..models.yolo_model import load_yolo_model


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
        performance_data (dict): Données de performance de l'exécution
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
        self.performance_data = {}
    
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
            
            # Enregistrer le temps d'exécution
            self.elapsed_time = time.time() - start_time
            
            # Collecter des données de performance
            self.performance_data = {
                "name": self.name,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(time.time()).isoformat(),
                "elapsed_time": self.elapsed_time,
                "cpu_count": os.cpu_count(),
                "status": "completed"
            }
            
            # Sauvegarder les métadonnées
            self._save_metadata()
            
            # Sauvegarder le résultat
            result_path = Path(str(self.metadata_path).replace('.json', '.pickle'))
            with open(result_path, 'wb') as f:
                pickle.dump(self.result, f)
            
            # Sauvegarder les données de performance
            perf_dir = Path(str(self.metadata_path).parent) / "performance"
            perf_dir.mkdir(exist_ok=True)
            perf_path = perf_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(perf_path, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
            
            self.status = 'completed'
            print(f"Cible '{self.name}' terminée avec succès en {self.elapsed_time:.2f} secondes.")
        
        except Exception as e:
            self.error = e
            self.status = 'failed'
            self.elapsed_time = time.time() - start_time
            
            # Enregistrer les données d'échec
            self.performance_data = {
                "name": self.name,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(time.time()).isoformat(),
                "elapsed_time": self.elapsed_time,
                "status": "failed",
                "error": str(e)
            }
            
            # Sauvegarder les données de performance en cas d'échec
            perf_dir = Path(str(self.metadata_path).parent) / "performance"
            perf_dir.mkdir(exist_ok=True)
            perf_path = perf_dir / f"{self.name}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(perf_path, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
            
            print(f"Erreur lors de l'exécution de la cible '{self.name}': {str(e)}")
            raise
        
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
        config (dict): Configuration du workflow
    """
    
    def __init__(self, name, metadata_dir=None, config=None):
        """
        Initialise un nouveau workflow.
        
        Args:
            name (str): Nom du workflow
            metadata_dir (str, optional): Répertoire pour les métadonnées
            config (dict, optional): Configuration du workflow
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
        
        # Configuration du workflow
        self.config = config or {}
        
        # Sauvegarder la configuration
        config_file = self.metadata_dir / "workflow_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
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
        
        # Démarrer le chronomètre pour mesurer la performance globale
        workflow_start_time = time.time()
        
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
        
        # Calculer le temps total d'exécution du workflow
        workflow_elapsed_time = time.time() - workflow_start_time
        
        # Sauvegarder les performances du workflow
        workflow_perf = {
            "name": self.name,
            "start_time": datetime.fromtimestamp(workflow_start_time).isoformat(),
            "end_time": datetime.fromtimestamp(time.time()).isoformat(),
            "elapsed_time": workflow_elapsed_time,
            "parallel": parallel,
            "targets": {name: target.performance_data for name, target in self.targets.items()}
        }
        
        # Créer le répertoire de performance s'il n'existe pas
        perf_dir = self.metadata_dir / "performance"
        perf_dir.mkdir(exist_ok=True)
        
        # Sauvegarder les performances
        perf_file = perf_dir / f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(perf_file, 'w') as f:
            json.dump(workflow_perf, f, indent=2)
        
        print(f"Workflow '{self.name}' terminé en {workflow_elapsed_time:.2f} secondes")
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
        target.performance_data = {}

    def get_performance_summary(self, output_path=None):
        """
        Génère un résumé des performances du workflow.
        
        Args:
            output_path (str, optional): Chemin pour sauvegarder le résumé au format CSV
        
        Returns:
            pandas.DataFrame: DataFrame contenant les statistiques de performance
        """
        # Récupérer tous les fichiers de performance
        perf_dir = self.metadata_dir / "performance"
        if not perf_dir.exists():
            return pd.DataFrame()
        
        # Récupérer les fichiers de performance des cibles
        target_perf_files = list(perf_dir.glob("*_*.json"))
        if not target_perf_files:
            return pd.DataFrame()
        
        # Charger les données de performance
        perf_data = []
        for file in target_perf_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        perf_data.append(data)
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {file}: {str(e)}")
        
        # Créer un DataFrame avec les données de performance
        df = pd.DataFrame(perf_data)
        
        # Sauvegarder au format CSV si demandé
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_path, index=False)
        
        return df
    
    def create_report(self, report_type='html', output_dir=None):
        """
        Génère un rapport sur le workflow.
        
        Args:
            report_type (str, optional): Type de rapport ('html', 'pdf'). Par défaut 'html'.
            output_dir (str, optional): Répertoire de sortie pour le rapport
        
        Returns:
            str: Chemin du rapport généré
        """
        # Vérifier les résultats du workflow
        if not self.targets:
            raise ValueError("Aucune cible dans le workflow")
        
        # Collecter les informations sur les cibles
        targets_info = {}
        for name, target in self.targets.items():
            targets_info[name] = {
                'status': target.status,
                'elapsed_time': target.elapsed_time,
                'error': str(target.error) if target.error else None
            }
        
        # Créer un résumé de performance
        perf_summary = self.get_performance_summary()
        
        # Configurer le répertoire de sortie
        if output_dir is None:
            output_dir = self.metadata_dir / "reports"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Créer le rapport
        from ..reporting.reporting import generate_quarto_report
        
        # Préparer les données pour le rapport
        report_data = {
            'workflow_name': self.name,
            'targets': targets_info,
            'performance': perf_summary.to_dict(orient='records') if not perf_summary.empty else []
        }
        
        # Générer le rapport
        output_file = output_dir / f"workflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_type}"
        
        try:
            report_path = generate_quarto_report(
                report_data,
                str(output_file),
                title=f"Rapport du workflow {self.name}"
            )
            
            print(f"Rapport généré: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"Erreur lors de la génération du rapport: {str(e)}")
            raise


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

def create_sam_detection_workflow(config, project_dir=None):
    """
    Crée un workflow pour la détection avec SAM 2.
    
    Args:
        config (dict): Configuration du workflow
        project_dir (str or Path, optional): Répertoire racine du projet
    
    Returns:
        Workflow: Workflow configuré pour la détection SAM
    """
    # Configuration
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)
    
    # Créer le workflow
    workflow = Workflow("sam_detection", config=config)
    
    # Étape 1: Conversion TIFF → JPEG
    @target(
        workflow=workflow,
        name="convert_tiff_to_jpeg",
        file_inputs=[project_dir / "data" / "raw_data"],
        file_outputs=[project_dir / "data" / "proc_data" / "jpeg_images"]
    )
    def convert_images():
        """Convertit les images TIFF en JPEG"""
        from ..image_processing.conversion import process_directory
        
        input_dir = project_dir / "data" / "raw_data"
        output_dir = project_dir / "data" / "proc_data" / "jpeg_images"
        
        process_directory(str(input_dir), str(output_dir), recursive=True)
        
        return output_dir
    
    # Étape 2: Prétraitement des images
    @target(
        workflow=workflow,
        name="preprocess_images",
        inputs=["convert_tiff_to_jpeg"],
        file_outputs=[project_dir / "data" / "proc_data" / "preprocessed_images"]
    )
    def preprocess_images(jpeg_dir):
        """Prétraite les images pour améliorer la détection"""
        from ..image_processing.preprocessing import batch_preprocess_directory
        
        output_dir = project_dir / "data" / "proc_data" / "preprocessed_images"
        
        batch_preprocess_directory(str(jpeg_dir), str(output_dir))
        
        return output_dir
    
    # Étape 3: Détection avec SAM 2
    @target(
        workflow=workflow,
        name="detect_with_sam",
        inputs=["preprocess_images"],
        file_outputs=[project_dir / "outputs" / "sam_detections"]
    )
    def detect_with_sam(preprocessed_dir):
        """Détecte les spores avec le modèle SAM 2"""
        from ..models.sam_model import load_sam_model, batch_detect_with_sam, combine_sam_detections
        
        # Charger le modèle SAM 2
        sam_model_path = project_dir / "data" / "blank_models" / "sam2.1_l.pt"
        if not sam_model_path.exists():
            # Télécharger le modèle si nécessaire
            print(f"Téléchargement du modèle SAM 2...")
            # Ici, vous pourriez ajouter une fonction pour télécharger le modèle
            
        # Charger le modèle
        model = load_sam_model(str(sam_model_path))
        
        # Configurer le répertoire de sortie
        output_dir = project_dir / "outputs" / "sam_detections"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Effectuer la détection sur toutes les images
        all_results = {}
        
        for sample_dir in Path(preprocessed_dir).glob("*"):
            if sample_dir.is_dir():
                sample_name = sample_dir.name
                print(f"Détection des spores dans l'échantillon {sample_name}...")
                
                # Créer un sous-répertoire pour les détections de cet échantillon
                sample_output = output_dir / sample_name
                
                # Détection avec SAM
                results = batch_detect_with_sam(
                    model,
                    str(sample_dir),
                    str(sample_output)
                )
                
                # Convertir les résultats en détections standards
                detections = combine_sam_detections(results)
                
                # Sauvegarder les détections
                detection_file = sample_output / "sam_detections.json"
                with open(detection_file, 'w') as f:
                    json.dump(detections, f, indent=2)
                
                all_results[sample_name] = detections
        
        return all_results
    
    # Étape 4: Annotation manuelle (facultative si interactive)
    @target(
        workflow=workflow,
        name="prepare_annotations",
        inputs=["detect_with_sam"],
        file_outputs=[project_dir / "data" / "proc_data" / "yolo_annotations" / "data.yaml"]
    )
    def prepare_annotations(sam_detections):
        """Prépare les annotations pour l'entraînement de YOLO"""
        # Cette étape sert à préparer les annotations sans lancer l'interface interactive
        
        # Créer le répertoire d'annotations YOLO
        yolo_dir = project_dir / "data" / "proc_data" / "yolo_annotations"
        yolo_dir.mkdir(exist_ok=True, parents=True)
        
        # Créer les sous-répertoires pour l'entraînement et la validation
        train_images_dir = yolo_dir / "train" / "images"
        train_labels_dir = yolo_dir / "train" / "labels"
        val_images_dir = yolo_dir / "val" / "images"
        val_labels_dir = yolo_dir / "val" / "labels"
        
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        # Créer le fichier de classes
        spore_classes = config.get('spore_classes', [
            'ALAC',  # Alatospora acuminata
            'LUCU',  # Lunulospora curvula
            'HEST',  # Heliscella stellata
            'HELU',  # Heliscus lugdunensis
            'CLAQ',  # Clavatospora aquatica
            'ARTE',  # Articulospora tetracladia
            'TEMA',  # Tetracladium marchalanium
            'TRSP',  # Tricladium splendens
            'TUAC',  # Tumularia aquatica
            'Debris' # Débris divers
        ])
        
        with open(yolo_dir / "classes.txt", 'w') as f:
            for cls in spore_classes:
                f.write(f"{cls}\n")
        
        # Convertir les détections SAM en annotations YOLO
        # Note: Dans un workflow réel, cette étape serait interactive
        # Ici, nous simulons la création des annotations de base
        
        print("REMARQUE: Cette étape devrait normalement être interactive.")
        print("Utilisez l'outil d'annotation pour vérifier et corriger les annotations.")
        print(f"Les annotations de base seront générées dans {yolo_dir}")
        
        # Créer le fichier data.yaml pour YOLOv8
        data_yaml = {
            'path': str(yolo_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'nc': len(spore_classes),
            'names': spore_classes
        }
        
        yaml_path = yolo_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        return yaml_path
    
    # Étape 5: Entraînement de YOLO
    @target(
        workflow=workflow,
        name="train_yolo_model",
        inputs=["prepare_annotations"],
        file_outputs=[project_dir / "outputs" / "models" / "yolo_spores_model.pt"]
    )
    def train_yolo(data_yaml_path):
        """Entraîne un modèle YOLO sur les annotations préparées"""
        from ..models.yolo_model import load_yolo_model, train_yolo_model
        
        # Récupérer ou télécharger le modèle YOLO de base
        base_model_path = load_yolo_model(
            path=str(project_dir / "data" / "blank_models"),
            model="yolo11m.pt",
            download=True
        )
        
        if not base_model_path:
            raise FileNotFoundError("Impossible de trouver ou télécharger le modèle YOLO de base")
        
        # Configurer le répertoire de sortie
        models_dir = project_dir / "outputs" / "models"
        models_dir.mkdir(exist_ok=True, parents=True)
        
        # Entraîner le modèle
        print(f"Entraînement du modèle YOLO sur les annotations...")
        
        # Ajouter un timestamp pour le versionnage des modèles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = models_dir / f"yolo_model_{timestamp}"
        
        # Entraîner et retourner le chemin du modèle
        model_path = train_yolo_model(
            model_path=base_model_path,
            data_yaml_path=str(data_yaml_path),
            epochs=config.get('yolo_epochs', 100),
            img_size=config.get('yolo_image_size', 640),
            batch_size=config.get('yolo_batch_size', 16),
            output_dir=str(output_dir)
        )
        
        # Créer un lien symbolique vers la dernière version
        latest_model_path = models_dir / "yolo_spores_model_latest.pt"
        if os.path.exists(latest_model_path):
            os.remove(latest_model_path)
        
        # Créer une copie du modèle avec un nom standardisé
        shutil.copy(model_path, models_dir / "yolo_spores_model.pt")
        
        return models_dir / "yolo_spores_model.pt"
    
    # Étape 6: Validation du modèle
    @target(
        workflow=workflow,
        name="validate_model",
        inputs=["train_yolo_model", "prepare_annotations"],
        file_outputs=[project_dir / "outputs" / "validation"]
    )
    def validate_model(model_path, data_yaml_path):
        """Valide le modèle YOLO entraîné"""
        from ..models.yolo_model import validate_yolo_model
        
        # Configurer le répertoire de sortie
        validation_dir = project_dir / "outputs" / "validation"
        validation_dir.mkdir(exist_ok=True, parents=True)
        
        # Valider le modèle
        print(f"Validation du modèle {model_path}...")
        
        results = validate_yolo_model(
            model_path=str(model_path),
            data_yaml_path=str(data_yaml_path),
            output_dir=str(validation_dir)
        )
        
        # Sauvegarder les résultats
        result_file = validation_dir / "validation_results.json"
        with open(result_file, 'w') as f:
            # Convertir les résultats numpy en types Python standard
            serializable_results = {}
            for key, value in results.items():
                if hasattr(value, 'item'):  # Pour les scalaires numpy
                    serializable_results[key] = value.item()
                elif hasattr(value, 'tolist'):  # Pour les tableaux numpy
                    serializable_results[key] = value.tolist()
                else:
                    # Tenter de convertir, sinon convertir en chaîne
                    try:
                        json.dumps(value)  # Test de sérialisation
                        serializable_results[key] = value
                    except (TypeError, OverflowError):
                        serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, indent=2)
        
        return validation_dir
    
    # Étape 7: Détection sur tous les échantillons
    @target(
        workflow=workflow,
        name="process_all_samples",
        inputs=["train_yolo_model", "convert_tiff_to_jpeg"],
        file_outputs=[project_dir / "outputs" / "processed_samples"]
    )
    def process_all_samples(model_path, jpeg_dir):
        """Traite tous les échantillons avec le modèle YOLO entraîné"""
        from ..core.utils import get_sample_info_from_path
        from ..models.yolo_model import batch_predict_with_yolo
        
        # Configurer le répertoire de sortie
        output_dir = project_dir / "outputs" / "processed_samples"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Traiter chaque échantillon
        all_results = {}
        
        for sample_dir in Path(jpeg_dir).glob("*"):
            if sample_dir.is_dir():
                sample_name = sample_dir.name
                print(f"Traitement de l'échantillon {sample_name}...")
                
                # Extraire les informations de l'échantillon
                sample_info = get_sample_info_from_path(sample_dir)
                
                # Créer un dossier de sortie pour cet échantillon
                sample_output = output_dir / sample_name
                sample_output.mkdir(exist_ok=True, parents=True)
                
                # Effectuer les détections YOLO
                detections = batch_predict_with_yolo(
                    str(model_path),
                    str(sample_dir),
                    str(sample_output)
                )
                
                # Analyser les résultats
                from ..analysis.statistics import analyze_detections, save_analysis_results
                
                stats = analyze_detections(detections, sample_info)
                
                # Sauvegarder les résultats
                analysis_dir = sample_output / "analysis"
                csv_path, plot_paths = save_analysis_results(stats, str(analysis_dir))
                
                # Stocker les résultats pour l'étape suivante
                all_results[sample_name] = stats
        
        # Sauvegarder tous les résultats
        results_file = output_dir / "all_results.json"
        
        # Convertir les résultats numpy en types Python standard pour JSON
        serializable_results = {}
        for sample_name, stats in all_results.items():
            serializable_stats = {}
            for key, value in stats.items():
                if hasattr(value, 'item'):  # Pour les scalaires numpy
                    serializable_stats[key] = value.item()
                elif hasattr(value, 'tolist'):  # Pour les tableaux numpy
                    serializable_stats[key] = value.tolist()
                else:
                    # Tenter de convertir, sinon convertir en chaîne
                    try:
                        json.dumps(value)  # Test de sérialisation
                        serializable_stats[key] = value
                    except (TypeError, OverflowError):
                        serializable_stats[key] = str(value)
            
            serializable_results[sample_name] = serializable_stats
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return all_results
    
    # Étape 8: Comparaison des échantillons
    @target(
        workflow=workflow,
        name="compare_samples",
        inputs=["process_all_samples"],
        file_outputs=[project_dir / "outputs" / "comparison"]
    )
    def compare_samples(all_results):
        """Compare les résultats de tous les échantillons"""
        from ..analysis.statistics import compare_samples, perform_statistical_tests
        
        # Configurer le répertoire de sortie
        output_dir = project_dir / "outputs" / "comparison"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Convertir les résultats en DataFrame
        results_df, plot_paths = compare_samples(all_results, str(output_dir))
        
        # Effectuer des tests statistiques
        test_results = perform_statistical_tests(results_df, str(output_dir))
        
        # Sauvegarder les résultats
        results_df.to_csv(output_dir / "comparison_results.csv", index=False)
        
        return output_dir
    
    # Étape 9: Génération de rapport Quarto
    @target(
        workflow=workflow,
        name="generate_report",
        inputs=["compare_samples", "process_all_samples"],
        file_outputs=[project_dir / "outputs" / "reports"]
    )
    def generate_report(comparison_dir, all_results):
        """Génère un rapport Quarto avec les résultats d'analyse"""
        from ..reporting.reporting import generate_quarto_report
        
        # Configurer le répertoire de sortie
        reports_dir = project_dir / "outputs" / "reports"
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        # Nom du fichier de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = reports_dir / f"analysis_report_{timestamp}.html"
        
        # Préparer les données pour le rapport (s'assurer qu'elles sont sérialisables)
        serializable_results = {}
        for sample_name, stats in all_results.items():
            serializable_stats = {}
            for key, value in stats.items():
                if hasattr(value, 'item'):  # Pour les scalaires numpy
                    serializable_stats[key] = value.item()
                elif hasattr(value, 'tolist'):  # Pour les tableaux numpy
                    serializable_stats[key] = value.tolist()
                else:
                    # Tenter de convertir, sinon convertir en chaîne
                    try:
                        json.dumps(value)  # Test de sérialisation
                        serializable_stats[key] = value
                    except (TypeError, OverflowError):
                        serializable_stats[key] = str(value)
            
            serializable_results[sample_name] = serializable_stats
        
        # Générer le rapport
        report_path = generate_quarto_report(
            serializable_results,
            str(output_file),
            title="Analyse de spores d'hyphomycètes"
        )
        
        print(f"Rapport généré: {report_path}")
        return reports_dir
    
    return workflow


def create_custom_workflow(config, project_dir=None):
    """
    Crée un workflow personnalisé basé sur la configuration.
    
    Args:
        config (dict): Configuration du workflow
        project_dir (str or Path, optional): Répertoire racine du projet
    
    Returns:
        Workflow: Workflow personnalisé
    """
    # Configuration
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)
    
    # Choisir le workflow en fonction de la configuration
    workflow_type = config.get('workflow_type', 'complete')
    
    if workflow_type == 'sam':
        # Workflow de détection SAM uniquement
        workflow = Workflow("sam_detection", config=config)
        # Définir les étapes pour SAM uniquement...
        
    elif workflow_type == 'yolo':
        # Workflow d'entraînement YOLO uniquement
        workflow = Workflow("yolo_training", config=config)
        # Définir les étapes pour YOLO uniquement...
        
    elif workflow_type == 'analysis':
        # Workflow d'analyse uniquement
        workflow = Workflow("analysis", config=config)
        # Définir les étapes pour l'analyse uniquement...
        
    else:
        # Workflow complet par défaut
        workflow = create_sam_detection_workflow(config, project_dir)
    
    return workflow


def get_latest_model_version(models_dir, model_type='yolo'):
    """
    Récupère la dernière version du modèle dans le répertoire spécifié.
    
    Args:
        models_dir (str or Path): Répertoire contenant les modèles
        model_type (str): Type de modèle ('yolo', 'unet', 'sam')
    
    Returns:
        Path: Chemin du modèle le plus récent
    """
    models_dir = Path(models_dir)
    pattern = f"{model_type}_*"
    
    # Chercher tous les dossiers de modèles
    model_dirs = list(models_dir.glob(pattern))
    
    if not model_dirs:
        # Vérifier s'il existe un modèle standard
        if model_type == 'yolo':
            standard_model = models_dir / "yolo_spores_model.pt"
            if standard_model.exists():
                return standard_model
        elif model_type == 'unet':
            standard_model = models_dir / "unet_spores_model.h5"
            if standard_model.exists():
                return standard_model
        elif model_type == 'sam':
            standard_model = models_dir / "sam_model.pt"
            if standard_model.exists():
                return standard_model
        
        return None
    
    # Trier par date de modification (le plus récent en premier)
    model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Chercher le fichier de modèle dans le dossier le plus récent
    if model_type == 'yolo':
        model_file = model_dirs[0] / 'yolo_spores_model.pt'
    elif model_type == 'unet':
        model_file = model_dirs[0] / 'unet_spores_model.h5'
    elif model_type == 'sam':
        model_file = model_dirs[0] / 'sam_model.pt'
    
    if model_file.exists():
        return model_file
    
    return None