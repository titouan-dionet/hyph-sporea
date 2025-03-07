"""
Exemple de workflow pour le projet HYPH-SPOREA.

Ce script montre comment utiliser le module de workflow pour créer
un pipeline de traitement d'images et d'analyse de spores.
"""

import sys
import os
import json
import random
import yaml
import pandas as pd
from pathlib import Path

# Ajouter le répertoire parent au PATH pour importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Python.core import HyphSporeaProcessor
from Python.core.utils import get_sample_info_from_path
from Python.image_processing import batch_preprocess_directory, create_binary_masks
from Python.models import train_yolo_model

from workflow import Workflow, target
from pipeline_config import load_config


def main():
    """Exemple d'utilisation du workflow"""
    # Charger la configuration
    config = load_config()
    
    # Créer le workflow
    workflow = Workflow("hyph_sporea_pipeline")
    
    # Définir le processeur HYPH-SPOREA
    processor = HyphSporeaProcessor(config['PROJECT_ROOT'])
    
    # ===== Étape 1: Conversion des images TIFF en JPEG =====
    @target(
        workflow=workflow,
        name="convert_tiff_to_jpeg",
        file_inputs=[config['raw_data_dir']],
        file_outputs=[config['proc_data_dir'] / "jpeg_images"]
    )
    def convert_images():
        """
        Convertit les images TIFF en JPEG en préservant la structure des dossiers.
        Traite tous les dossiers d'échantillons dans le dossier d'expérience.
        """
        # Trouver tous les dossiers d'échantillons dans le dossier d'expérience
        experiment_dirs = [d for d in config['raw_data_dir'].glob("*") if d.is_dir()]
        
        if not experiment_dirs:
            print(f"Aucun dossier d'expérience trouvé dans {config['raw_data_dir']}")
            return config['proc_data_dir'] / "jpeg_images"
        
        # Pour chaque dossier d'expérience, convertir toutes les images
        for exp_dir in experiment_dirs:
            output_dir = config['proc_data_dir'] / "jpeg_images" / exp_dir.name
            os.makedirs(output_dir, exist_ok=True)
            
            # Rechercher les sous-dossiers d'échantillons
            sample_dirs = [d for d in exp_dir.glob("*") if d.is_dir()]
            
            if not sample_dirs:
                # Si pas de sous-dossiers, traiter le dossier d'expérience directement
                processor.convert_tiff_to_jpeg(str(exp_dir), str(output_dir))
            else:
                # Pour chaque dossier d'échantillon, convertir les images
                for sample_dir in sample_dirs:
                    sample_output = output_dir / sample_dir.name
                    processor.convert_tiff_to_jpeg(str(sample_dir), str(sample_output))
                    
        return config['proc_data_dir'] / "jpeg_images"
    
    # ===== Étape 2: Prétraitement des images =====
    @target(
        workflow=workflow,
        name="preprocess_images",
        inputs=["convert_tiff_to_jpeg"],
        file_outputs=[config['proc_data_dir'] / "preprocessed_images"]
    )
    def preprocess_images(jpeg_dir):
        """
        Prétraite les images JPEG en préservant la structure des dossiers.
        """
        # Trouver tous les dossiers d'expérience
        experiment_dirs = [d for d in jpeg_dir.glob("*") if d.is_dir()]
        
        if not experiment_dirs:
            print(f"Aucun dossier d'expérience trouvé dans {jpeg_dir}")
            return config['proc_data_dir'] / "preprocessed_images"
        
        # Pour chaque dossier d'expérience, prétraiter toutes les images
        for exp_dir in experiment_dirs:
            output_dir = config['proc_data_dir'] / "preprocessed_images" / exp_dir.name
            os.makedirs(output_dir, exist_ok=True)
            
            # Rechercher les sous-dossiers d'échantillons
            sample_dirs = [d for d in exp_dir.glob("*") if d.is_dir()]
            
            if not sample_dirs:
                # Si pas de sous-dossiers, traiter le dossier d'expérience directement
                batch_preprocess_directory(str(exp_dir), str(output_dir), intensity='light')
            else:
                # Pour chaque dossier d'échantillon, prétraiter les images
                for sample_dir in sample_dirs:
                    sample_output = output_dir / sample_dir.name
                    batch_preprocess_directory(str(sample_dir), str(sample_output), intensity='light')
        
        return config['proc_data_dir'] / "preprocessed_images"    
    
    # ===== Étape 3: Création des masques pour U-Net =====
    @target(
        workflow=workflow,
        name="create_masks",
        inputs=["preprocess_images"],
        file_outputs=[config['proc_data_dir'] / "masks"]
    )
    def create_masks(preprocessed_dir):
        """
        Crée des masques binaires pour tous les échantillons.
        """
        # Trouver tous les dossiers d'expérience
        experiment_dirs = [d for d in preprocessed_dir.glob("*") if d.is_dir()]
        
        if not experiment_dirs:
            print(f"Aucun dossier d'expérience trouvé dans {preprocessed_dir}")
            return config['proc_data_dir'] / "masks"
        
        # Pour chaque dossier d'expérience, créer des masques
        for exp_dir in experiment_dirs:
            output_dir = config['proc_data_dir'] / "masks" / exp_dir.name
            os.makedirs(output_dir, exist_ok=True)
            
            # Rechercher les sous-dossiers d'échantillons
            sample_dirs = [d for d in exp_dir.glob("*") if d.is_dir()]
            
            if not sample_dirs:
                # Si pas de sous-dossiers, traiter le dossier d'expérience directement
                create_binary_masks(str(exp_dir), str(output_dir))
            else:
                # Pour chaque dossier d'échantillon, créer des masques
                for sample_dir in sample_dirs:
                    sample_output = output_dir / sample_dir.name
                    create_binary_masks(str(sample_dir), str(sample_output))
        
        return config['proc_data_dir'] / "masks"
    
    # ===== Étape 3b: Première utilisation de l'interface =====
    @target(
        workflow=workflow,
        name="annotate_images",
        inputs=["preprocess_images"],
        file_outputs=[config['proc_data_dir'] / "yolo_annotations" / "data.yaml"]
    )
    def annotate_images(preprocessed_dir):
        """Lance l'interface graphique d'annotation pour tous les échantillons"""
        from Python.gui.annotation_tool import run_annotation_tool
        import shutil
        import yaml
        import os
        import random
        
        # Chemins des dossiers
        yolo_dir = config['proc_data_dir'] / "yolo_annotations"
        yaml_path = yolo_dir / "data.yaml"
        
        # Vérifier si le fichier data.yaml existe déjà
        if yaml_path.exists():
            print(f"Fichier d'annotations {yaml_path} trouvé, étape d'annotation ignorée.")
            return yaml_path
        
        print("\n" + "="*50)
        print("ÉTAPE INTERACTIVE: Veuillez annoter quelques images")
        print("1. Une interface graphique va s'ouvrir")
        print("2. Parcourez les dossiers d'échantillons à annoter dans:", preprocessed_dir)
        print("3. Annotez au moins quelques images de chaque échantillon")
        print("4. Enregistrez et convertissez en format YOLO avant de fermer")
        print("="*50 + "\n")
        
        # Lancer l'interface d'annotation
        input("Appuyez sur Entrée pour lancer l'interface d'annotation...")
        run_annotation_tool()
        
        # Créer la structure de dossiers pour YOLO
        train_images_dir = yolo_dir / "train" / "images"
        train_labels_dir = yolo_dir / "train" / "labels"
        val_images_dir = yolo_dir / "val" / "images"
        val_labels_dir = yolo_dir / "val" / "labels"
        
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        # Récupérer toutes les annotations à partir des dossiers d'échantillons
        all_annotations = []
        
        # Parcourir tous les dossiers d'expérience
        for exp_dir in preprocessed_dir.glob("*"):
            if not exp_dir.is_dir():
                continue
                
            # Chercher les annotations dans le dossier d'expérience
            yolo_annotations_dir = exp_dir / "yolo_annotations"
            if yolo_annotations_dir.exists():
                # Récupérer les fichiers .txt (annotations) sauf classes.txt
                for txt_file in yolo_annotations_dir.glob("*.txt"):
                    if txt_file.stem == 'classes':
                        continue
                    
                    # Chercher l'image correspondante
                    for ext in ['.jpeg', '.jpg', '.png']:
                        img_file = yolo_annotations_dir.parent / f"{txt_file.stem}{ext}"
                        if img_file.exists():
                            all_annotations.append((img_file, txt_file))
                            break
            
            # Chercher dans les sous-dossiers d'échantillons
            for sample_dir in exp_dir.glob("*"):
                if not sample_dir.is_dir():
                    continue
                    
                yolo_annotations_dir = sample_dir / "yolo_annotations"
                if yolo_annotations_dir.exists():
                    # Récupérer les fichiers .txt (annotations) sauf classes.txt
                    for txt_file in yolo_annotations_dir.glob("*.txt"):
                        if txt_file.stem == 'classes':
                            continue
                        
                        # Chercher l'image correspondante
                        for ext in ['.jpeg', '.jpg', '.png']:
                            img_file = yolo_annotations_dir.parent / f"{txt_file.stem}{ext}"
                            if img_file.exists():
                                all_annotations.append((img_file, txt_file))
                                break
        
        if not all_annotations:
            raise FileNotFoundError(
                "Aucune annotation trouvée. Assurez-vous d'avoir annoté des images et converti en format YOLO."
            )
        
        # Mélanger et diviser les annotations en ensembles d'entraînement et de validation
        random.shuffle(all_annotations)
        split_idx = int(len(all_annotations) * 0.8)
        train_set = all_annotations[:split_idx]
        val_set = all_annotations[split_idx:]
        
        # Copier les fichiers
        for img_file, txt_file in train_set:
            shutil.copy(img_file, train_images_dir / img_file.name)
            shutil.copy(txt_file, train_labels_dir / txt_file.name)
        
        for img_file, txt_file in val_set:
            shutil.copy(img_file, val_images_dir / img_file.name)
            shutil.copy(txt_file, val_labels_dir / txt_file.name)
        
        # Rechercher le fichier classes.txt dans les dossiers d'annotations
        classes_file = None
        for directory in preprocessed_dir.glob("**/yolo_annotations"):
            potential_file = directory / "classes.txt"
            if potential_file.exists():
                classes_file = potential_file
                break
        
        # Lire les classes
        classes = []
        if classes_file and classes_file.exists():
            shutil.copy(classes_file, yolo_dir / "classes.txt")
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
        else:
            classes = config.get('spore_classes', [
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
        
        # Créer le fichier data.yaml
        data = {
            'path': str(yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(classes),
            'names': classes
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        print(f"Annotation terminée avec succès! {len(all_annotations)} images annotées.")
        print(f"Structure de données YOLO créée dans {yolo_dir}")
        print(f"Ensembles : {len(train_set)} images d'entraînement, {len(val_set)} images de validation")
        
        return yaml_path
    
    # ===== Étape 4: Entraînement du modèle YOLO =====
    @target(
        workflow=workflow,
        name="train_model",
        inputs=["annotate_images"],
        file_outputs=[config['models_dir'] / "yolo_spores_model.pt"]
    )
    def train_model(data_yaml_path):
        """Entraîne le modèle YOLO"""
        return train_yolo_model(
            data_yaml_path,
            epochs=5,
            output_dir=config['models_dir']
        )
    
    # ===== Étape 5: Traitement de l'échantillon 1 =====
    @target(
        workflow=workflow,
        name="process_samples",
        inputs=["train_model"],
        file_outputs=[config['output_dir'] / "processed_samples"]
    )
    def process_samples(model_path):
        """
        Traite tous les échantillons avec le modèle entraîné.
        """
        from Python.core.utils import get_sample_info_from_path
        
        # Chercher tous les dossiers d'échantillons
        all_samples = []
        jpeg_dir = config['proc_data_dir'] / "jpeg_images"
        
        # Parcourir tous les dossiers d'expérience
        for exp_dir in jpeg_dir.glob("*"):
            if not exp_dir.is_dir():
                continue
            
            # Chercher les sous-dossiers d'échantillons
            sample_dirs = [d for d in exp_dir.glob("*") if d.is_dir()]
            
            if not sample_dirs:
                # Si pas de sous-dossiers, le dossier d'expérience est l'échantillon
                all_samples.append(exp_dir)
            else:
                # Ajouter tous les dossiers d'échantillons
                all_samples.extend(sample_dirs)
        
        # Si aucun échantillon n'est trouvé, utiliser les dossiers d'expérience comme échantillons
        if not all_samples:
            all_samples = [d for d in jpeg_dir.glob("*") if d.is_dir()]
        
        # Créer le répertoire de sortie
        output_dir = config['output_dir'] / "processed_samples"
        os.makedirs(output_dir, exist_ok=True)
        
        # Traiter chaque échantillon
        for sample_dir in all_samples:
            # Extraire les informations de l'échantillon
            sample_info = get_sample_info_from_path(sample_dir)
            
            # Créer un dossier de sortie spécifique pour cet échantillon
            sample_output = output_dir / sample_dir.name
            
            # Traiter l'échantillon
            processor.process_sample(
                str(sample_dir),
                model_path,
                str(sample_output),
                sample_info=sample_info
            )
            
            print(f"Échantillon {sample_dir.name} traité avec succès")
        
        return output_dir
    
    # # ===== Étape 6: Traitement de l'échantillon 2 =====
    # @target(
    #     workflow=workflow,
    #     name="process_sample_2",
    #     inputs=["train_model"],
    #     file_inputs=[config['proc_data_dir'] / "jpeg_images" / "sample_2"],
    #     file_outputs=[config['output_dir'] / "processed_sample_2"]
    # )
    # def process_sample_2(model_path):
    #     """Traite l'échantillon 2"""
    #     # Détection avec YOLO
    #     detections = batch_predict_with_yolo(
    #         model_path,
    #         config['proc_data_dir'] / "jpeg_images" / "sample_2",
    #         config['output_dir'] / "processed_sample_2"
    #     )
        
    #     # Analyse des résultats
    #     sample_info = {
    #         'sample': 'T9_TEMA_MA01_C6',
    #         'strain': 'T9_TEMA',
    #         'condition': 'MA01_C6',
    #         'replicate': '1'
    #     }
        
    #     stats = analyze_detections(detections, sample_info)
        
    #     # Sauvegarder les résultats
    #     save_analysis_results(
    #         stats,
    #         config['output_dir'] / "processed_sample_2" / "analysis"
    #     )
        
    #     return {
    #         'detections': detections,
    #         'stats': stats
    #     }
    
    # ===== Étape 7: Comparaison des échantillons =====
    @target(
        workflow=workflow,
        name="compare_samples",
        inputs=["process_samples"],
        file_outputs=[config['output_dir'] / "comparison"]
    )
    def compare_samples(processed_dir):
        """
        Compare les résultats de tous les échantillons traités.
        """
        # Créer le répertoire de sortie
        output_dir = config['output_dir'] / "comparison"
        os.makedirs(output_dir, exist_ok=True)
        
        # Charger les résultats de tous les échantillons
        all_results = {}
        
        # Parcourir tous les dossiers d'échantillons traités
        for sample_dir in processed_dir.glob("*"):
            if not sample_dir.is_dir():
                continue
            
            # Charger les statistiques de l'échantillon
            stats_file = sample_dir / "analysis" / "sample_stats.csv"
            
            if stats_file.exists():
                try:
                    stats_df = pd.read_csv(stats_file)
                    if not stats_df.empty:
                        sample_name = sample_dir.name
                        all_results[sample_name] = stats_df.iloc[0].to_dict()
                except Exception as e:
                    print(f"Erreur lors du chargement des statistiques de {sample_dir.name}: {str(e)}")
        
        if not all_results:
            print("Aucun résultat trouvé pour la comparaison")
            return output_dir
        
        # Comparer les échantillons
        processor.results = all_results
        processor.compare_samples(output_dir)
        
        return output_dir
    
    # Visualiser le workflow
    workflow.visualize(config['output_dir'] / "visualizations" / "workflow_graph.png")
    
    # Exécuter le workflow
    print("\nExécution du workflow complet...")
    results = workflow.execute(parallel=False)
    
    print("\nRésultats du workflow:")
    for target_name, result in results.items():
        if target_name == "compare_samples":
            print(f"\nComparaison des échantillons:")
            # Vérifier le type de résultat
            if isinstance(result, pd.DataFrame):
                print(result[['sample', 'total_spores_detected']])
            else:
                print(f"Résultats de comparaison disponibles dans: {result}")
    
    print("\nWorkflow terminé.")


if __name__ == "__main__":
    main()