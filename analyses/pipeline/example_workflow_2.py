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
import time
from datetime import datetime
import shutil

from Python.core import HyphSporeaProcessor
from Python.core.utils import get_sample_info_from_path
from Python.image_processing import batch_preprocess_directory, create_binary_masks
from Python.models import train_yolo_model, load_yolo_model
from Python.models.sam_model import load_sam_model, batch_detect_with_sam, combine_sam_detections

from Python.workflow import Workflow, target
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
    
    # ===== Étape 2b: Détection initiale avec SAM 2 =====
    @target(
        workflow=workflow,
        name="detect_with_sam",
        inputs=["preprocess_images"],
        file_outputs=[config['output_dir'] / "sam_detections"]
    )
    def detect_with_sam(preprocessed_dir):
        """
        Effectue une détection initiale des spores avec le modèle SAM 2.
        """
        # Vérifier si le modèle SAM 2 existe, sinon utiliser le modèle par défaut
        sam_model_path = config['blank_model_dir'] / config['default_sam_model']
        
        if not sam_model_path.exists():
            print(f"Avertissement: Modèle SAM {sam_model_path} non trouvé. Utilisation du modèle par défaut.")
            # Dans une implémentation réelle, il faudrait télécharger le modèle
        
        # Charger le modèle SAM
        sam_model = load_sam_model(str(sam_model_path))
        
        # Configurer le répertoire de sortie
        output_dir = config['output_dir'] / "sam_detections"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Traiter chaque dossier d'échantillon
        all_detections = {}
        
        for exp_dir in preprocessed_dir.glob("*"):
            if not exp_dir.is_dir():
                continue
                
            # Rechercher les sous-dossiers d'échantillons
            sample_dirs = [d for d in exp_dir.glob("*") if d.is_dir()]
            
            # Si aucun sous-dossier, traiter le dossier d'expérience directement
            if not sample_dirs:
                sample_dirs = [exp_dir]
            
            # Traiter chaque échantillon
            for sample_dir in sample_dirs:
                sample_name = sample_dir.name
                print(f"Détection des spores avec SAM dans {sample_name}...")
                
                # Créer un répertoire de sortie pour cet échantillon
                sample_output_dir = output_dir / sample_name
                sample_output_dir.mkdir(exist_ok=True, parents=True)
                
                # Effectuer la détection avec SAM
                sam_results = batch_detect_with_sam(
                    sam_model,
                    str(sample_dir),
                    str(sample_output_dir)
                )
                
                # Convertir les résultats en détections standardisées
                detections = combine_sam_detections(sam_results)
                
                # Sauvegarder les détections
                detection_file = sample_output_dir / "sam_detections.json"
                with open(detection_file, 'w') as f:
                    json.dump(detections, f, indent=2)
                
                all_detections[sample_name] = detections
        
        return output_dir
    
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
        inputs=["preprocess_images", "detect_with_sam"],
        file_outputs=[config['proc_data_dir'] / "yolo_annotations" / "data.yaml"]
    )
    def annotate_images(preprocessed_dir, sam_detections_dir):
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
        print("3. Les détections SAM sont disponibles dans:", sam_detections_dir)
        print("4. Annotez au moins quelques images de chaque échantillon")
        print("5. Enregistrez et convertissez en format YOLO avant de fermer")
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
        
        # Puis, chercher aussi dans jpeg_dir
        jpeg_dir = config['proc_data_dir'] / "jpeg_images"
        for exp_dir in jpeg_dir.glob("*"):
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
    
    # ===== Étape 4a: Chargement du modèle YOLO =====
    @target(
        workflow=workflow,
        name="load_model",
        file_inputs=[config['blank_model_dir']],
        file_outputs=[config['blank_model_dir'] / "yolov8n.pt"]  # Changé à yolov8n.pt
    )
    def load_model():
        """Vérifie l'existence du modèle YOLO, le télécharge si nécessaire, et renvoie son chemin"""
        # Utiliser un modèle plus petit et plus fiable (yolov8n au lieu de yolo11m)
        model_name = "yolov8n.pt"
        model_path = config['blank_model_dir'] / model_name
        
        # Vérifier si le modèle existe déjà
        if model_path.exists():
            print(f"Modèle trouvé: {model_path}")
            return str(model_path)
        
        # Si le modèle n'existe pas, utiliser l'API YOLO directement
        try:
            print(f"Le modèle {model_path} n'existe pas. Téléchargement automatique...")
            
            # Créer le répertoire si nécessaire
            model_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Utiliser l'API YOLO directement
            from ultralytics import YOLO
            
            # Initialiser le modèle (cela le télécharge automatiquement)
            model = YOLO(model_name)
            
            # Sauvegarder le modèle au chemin souhaité
            model_file = str(model_path)
            print(f"Sauvegarde du modèle dans {model_file}")
            
            # Exporter/sauvegarder le modèle
            success = model.export(format="pt", imgsz=640)
            
            # Copier le fichier vers l'emplacement souhaité
            import shutil
            source_path = Path(model_name)
            if source_path.exists():
                shutil.copy(source_path, model_path)
                print(f"Modèle copié dans {model_path}")
            else:
                # Si le fichier n'est pas au chemin racine, le chercher dans le répertoire courant
                for file in Path().glob("*.pt"):
                    if file.name == model_name:
                        shutil.copy(file, model_path)
                        print(f"Modèle trouvé et copié dans {model_path}")
                        break
            
            # Vérifier que le modèle existe maintenant
            if model_path.exists():
                print(f"Vérification réussie: modèle disponible dans {model_path}")
                return str(model_path)
            else:
                # Si le modèle n'a pas été sauvegardé au bon endroit, utiliser le modèle chargé en mémoire
                print("Le modèle n'a pas été sauvegardé au chemin spécifié, mais est chargé en mémoire.")
                return model_name  # Retourner le nom qui peut être utilisé directement par YOLO
            
        except Exception as e:
            print(f"Erreur lors du téléchargement du modèle: {str(e)}")
            
            # En dernier recours, utiliser directement le nom du modèle
            print("Tentative d'utilisation directe du modèle via le nom...")
            return model_name  # Retourner simplement le nom du modèle que YOLO recherchera

    # ===== Étape 4b: Entraînement du modèle YOLO =====
    @target(
        workflow=workflow,
        name="train_model",
        inputs=["load_model", "annotate_images"],
        file_outputs=[config['models_dir'] / "yolo_spores_model.pt"]
    )
    def train_model(model_path, data_yaml_path):
        """Entraîne le modèle YOLO"""
        
        # Ajouter un timestamp pour le versionnage des modèles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config['models_dir'] / f"yolo_model_{timestamp}"
        
        # Vérifier si model_path est un chemin de fichier ou un nom de modèle
        if os.path.exists(model_path):
            print(f"Utilisation du modèle local: {model_path}")
        else:
            print(f"Utilisation du modèle par nom: {model_path}")
        
        # Entraîner le modèle YOLO
        final_model_path = train_yolo_model(
            model_path = model_path,
            data_yaml_path = data_yaml_path,
            epochs=config.get('yolo_epochs', 100),
            output_dir=str(output_dir)
        )
        
        # Créer une copie avec un nom standardisé
        standard_model_path = config['models_dir'] / "yolo_spores_model.pt"
        shutil.copy(final_model_path, standard_model_path)
        
        print(f"Modèle entraîné et sauvegardé dans {standard_model_path}")
        
        return standard_model_path
    
    # ===== Étape 4c: Validation du modèle YOLO =====
    @target(
        workflow=workflow,
        name="validate_model",
        inputs=["train_model"],
        file_outputs=[config['output_dir'] / "validation"]
    )
    def validate_model(model_path):
        """Valide le modèle YOLO et sauvegarde les résultats"""
        from Python.models import validate_yolo_model
        
        # Définir le répertoire de sortie pour la validation
        validation_dir = config['output_dir'] / "validation"
        os.makedirs(validation_dir, exist_ok=True)
        
        # Charger le fichier data.yaml
        data_yaml_path = config['proc_data_dir'] / "yolo_annotations" / "data.yaml"
        
        # Valider le modèle
        results = validate_yolo_model(
            model_path=model_path,
            data_yaml_path=str(data_yaml_path),
            output_dir=str(validation_dir)
        )
        
        # Convertir les résultats en dictionnaire 
        # en utilisant la méthode results_dict() de DetMetrics
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict()
        else:
            # Fallback si results est déjà un dictionnaire ou un autre type
            results_dict = {}
            try:
                # Essayer d'obtenir les valeurs importantes
                if hasattr(results, 'mean_results'):
                    # Créer un dictionnaire à partir des résultats moyens
                    keys = results.keys if hasattr(results, 'keys') else ["precision", "recall", "mAP50", "mAP"]
                    values = results.mean_results()
                    results_dict = dict(zip(keys, values))
                elif hasattr(results, 'maps'):
                    # Récupérer les mAP values
                    results_dict = {'maps': results.maps}
                elif hasattr(results, 'fitness'):
                    # Au moins récupérer le score de fitness
                    results_dict = {'fitness': results.fitness()}
            except Exception as e:
                print(f"Impossible d'extraire les résultats de validation: {str(e)}")
                results_dict = {"error": str(e)}
        
        # Sauvegarder les résultats en format texte
        metrics_path = validation_dir / "validation_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write(f"# Validation du modèle YOLO\n\n")
            f.write(f"Modèle: {model_path}\n")
            f.write(f"Dataset: {data_yaml_path}\n\n")
            
            # Écrire les métriques principales
            f.write("## Métriques\n\n")
            for key, value in results_dict.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Sauvegarder également au format JSON
        import json
        json_path = validation_dir / "validation_metrics.json"
        
        # Convertir tous les valeurs non-sérialisables en chaînes
        serializable_results = {}
        for key, value in results_dict.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return validation_dir
    
    # ===== Étape 5: Traitement de tous les échantillons =====
    @target(
        workflow=workflow,
        name="process_samples",
        inputs=["validate_model"],
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
    
    # ===== Étape 6: Comparaison des échantillons =====
    @target(
        workflow=workflow,
        name="compare_samples",
        inputs=["process_samples"],
        file_outputs=[config['output_dir'] / "comparison"]
    )
    def compare_samples(processed_dir):
        """
        Compare les résultats de plusieurs échantillons traités.
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
    
    # ===== Étape 7: Génération d'un rapport Quarto =====
    @target(
        workflow=workflow,
        name="generate_report",
        inputs=["compare_samples"],
        file_outputs=[config['output_dir'] / "reports"]
    )
    def generate_report(comparison_dir):
        """
        Génère un rapport d'analyse au format Quarto.
        """
        # Vérifier que le processeur a des résultats
        if not processor.results:
            print("Aucun résultat disponible pour générer un rapport.")
            return
        
        # Générer le rapport
        reports_dir = config['output_dir'] / "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Générer différents formats de rapport
        print("Génération des rapports d'analyse...")
        
        try:
            html_report = processor.generate_report(output_format='html')
            print(f"Rapport HTML généré: {html_report}")
            
            # Tenter de générer également un PDF si Quarto le supporte
            try:
                pdf_report = processor.generate_report(output_format='pdf')
                print(f"Rapport PDF généré: {pdf_report}")
            except Exception as e:
                print(f"Impossible de générer le rapport PDF: {str(e)}")
        
        except Exception as e:
            print(f"Erreur lors de la génération des rapports: {str(e)}")
        
        return reports_dir
    
    # Visualiser le workflow
    workflow.visualize(config['output_dir'] / "visualizations" / "workflow_graph.png")
    
    # Exécuter le workflow
    print("\nExécution du workflow complet...")
    start_time = time.time()
    results = workflow.execute(parallel=False)
    elapsed_time = time.time() - start_time
    
    print(f"\nWorkflow terminé en {elapsed_time:.2f} secondes")
    print("\nRésultats du workflow:")
    for target_name, result in results.items():
        if target_name == "compare_samples":
            print(f"\nComparaison des échantillons:")
            # Vérifier le type de résultat
            if isinstance(result, pd.DataFrame):
                print(result[['sample', 'total_spores_detected']])
            else:
                print(f"Résultats de comparaison disponibles dans: {result}")
    
    # Générer et afficher les performances du workflow
    try:
        perf_summary = workflow.get_performance_summary()
        print("\nRésumé des performances:")
        if not perf_summary.empty:
            print(perf_summary[['name', 'elapsed_time', 'status']])
        
        # Sauvegarder le résumé des performances
        perf_file = config['output_dir'] / "performance" / f"workflow_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs(perf_file.parent, exist_ok=True)
        perf_summary.to_csv(perf_file, index=False)
        
        print(f"Résumé des performances sauvegardé dans {perf_file}")
    except Exception as e:
        print(f"Erreur lors de la génération du résumé des performances: {str(e)}")
    
    print("\nWorkflow terminé.")


if __name__ == "__main__":
    main()