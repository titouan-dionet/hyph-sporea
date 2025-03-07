"""
Module principal contenant la classe HyphSporeaProcessor qui intègre toutes les fonctionnalités
pour l'analyse des spores d'hyphomycètes.

Ce module sert de point d'entrée pour utiliser les fonctionnalités du projet HYPH-SPOREA.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from pathlib import Path
import torch

from ultralytics import YOLO
from tensorflow.keras.models import load_model

from ..image_processing.conversion import process_directory
from ..image_processing.preprocessing import enhanced_preprocess_image
from ..models.unet_model import train_unet, predict_with_unet
from ..gui.annotation_tool import run_annotation_tool


class HyphSporeaProcessor:
    """
    Classe principale pour le traitement des images de spores d'hyphomycètes.
    
    Cette classe intègre toutes les fonctionnalités du projet HYPH-SPOREA :
    - Conversion des images TIFF en JPEG
    - Prétraitement des images
    - Annotation des images
    - Entraînement des modèles (YOLO, U-Net)
    - Traitement des échantillons
    - Analyse et comparaison des résultats
    
    Attributes:
        project_root (Path): Chemin racine du projet.
        data_dir (Path): Répertoire de données.
        output_dir (Path): Répertoire de sortie.
        models_dir (Path): Répertoire des modèles entraînés.
        yolo_model: Instance du modèle YOLO chargé.
        unet_model: Instance du modèle U-Net chargé.
        results (dict): Dictionnaire pour stocker les résultats d'analyse.
    """
    
    def __init__(self, project_root):
        """
        Initialise l'instance HyphSporeaProcessor.
        
        Args:
            project_root (str or Path): Chemin racine du projet.
        """
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "outputs"
        self.models_dir = self.project_root / "outputs" / "models"
        
        # Création des répertoires si nécessaires
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Modèles
        self.yolo_model = None
        self.unet_model = None
        
        # Dictionnaire pour stocker les résultats
        self.results = {}
    
    def convert_tiff_to_jpeg(self, input_dir, output_dir=None):
        """
        Convertit les fichiers TIFF d'un répertoire en JPEG.
        
        Args:
            input_dir (str): Répertoire contenant les images TIFF.
            output_dir (str, optional): Répertoire de sortie pour les images JPEG.
                Si None, crée un sous-dossier "jpeg_images" dans data_dir.
        
        Returns:
            Path: Chemin du répertoire de sortie contenant les images JPEG.
        
        Example:
            >>> processor = HyphSporeaProcessor("/chemin/vers/projet")
            >>> jpeg_dir = processor.convert_tiff_to_jpeg("/chemin/vers/tiff")
        """
        if output_dir is None:
            output_dir = self.data_dir / "proc_data" / "jpeg_images"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Utilisation de la fonction de conversion TIFF vers JPEG
        process_directory(str(input_dir), str(output_dir), recursive=True)
        
        return output_dir
    
    def preprocess_dataset(self, input_dir, output_dir=None):
        """
        Prétraite toutes les images d'un répertoire.
        
        Applique le prétraitement d'image amélioré à chaque image JPEG du répertoire.
        Préserve la structure des sous-répertoires.
        
        Args:
            input_dir (str): Répertoire contenant les images JPEG.
            output_dir (str, optional): Répertoire de sortie pour les images prétraitées.
                Si None, crée un sous-dossier "preprocessed_images" dans data_dir.
        
        Returns:
            Path: Chemin du répertoire de sortie contenant les images prétraitées.
        
        Example:
            >>> processor = HyphSporeaProcessor("/chemin/vers/projet")
            >>> preprocessed_dir = processor.preprocess_dataset("data/proc_data/jpeg_images")
        """
        input_dir = Path(input_dir)
        
        if output_dir is None:
            output_dir = self.data_dir / "proc_data" / "preprocessed_images"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Liste des fichiers JPEG à traiter
        jpeg_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    jpeg_path = os.path.join(root, file)
                    jpeg_files.append(jpeg_path)
        
        print(f"Prétraitement de {len(jpeg_files)} images...")
        
        # Traitement des images
        for i, jpeg_path in enumerate(jpeg_files):
            # Extraction du nom de fichier et de la structure de répertoires
            rel_path = os.path.relpath(os.path.dirname(jpeg_path), input_dir)
            filename = os.path.basename(jpeg_path)
            
            # Création du répertoire de sortie
            target_dir = output_dir / rel_path
            target_dir.mkdir(exist_ok=True, parents=True)
            
            # Chemin de sortie
            output_path = target_dir / filename
            
            # Prétraitement de l'image
            enhanced_preprocess_image(jpeg_path, save=True, output_path=str(output_path))
            
            # Affichage de la progression
            if (i + 1) % 10 == 0 or i == len(jpeg_files) - 1:
                print(f"Progression: {i + 1}/{len(jpeg_files)} images traitées")
        
        print("Prétraitement terminé.")
        return output_dir
    
    def create_annotation_tool(self):
        """
        Lance l'interface graphique d'annotation.
        
        Lance une interface tkinter permettant d'annoter les spores sur les images.
        L'interface permet de:
        - Charger un dossier d'images
        - Annoter les spores avec leur classe
        - Sauvegarder les annotations au format JSON et YOLO
        
        Returns:
            None
        
        Example:
            >>> processor = HyphSporeaProcessor("/chemin/vers/projet")
            >>> processor.create_annotation_tool()
        """
        run_annotation_tool()
    
    def train_yolo_model(self, data_yaml_path, epochs=100, img_size=640, batch_size=16):
        """
        Entraîne un modèle YOLO sur les données annotées.
        
        Args:
            data_yaml_path (str): Chemin du fichier data.yaml contenant les configurations d'entraînement.
            epochs (int, optional): Nombre d'époques d'entraînement. Par défaut 100.
            img_size (int, optional): Taille des images d'entraînement. Par défaut 640.
            batch_size (int, optional): Taille des batchs d'entraînement. Par défaut 16.
        
        Returns:
            Path: Chemin du modèle entraîné.
        
        Example:
            >>> processor = HyphSporeaProcessor("/chemin/vers/projet")
            >>> model_path = processor.train_yolo_model("data/proc_data/annotations/data.yaml")
        """
        # Créer un dossier pour le modèle avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"yolo_model_{timestamp}"
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialiser le modèle YOLO (YOLOv8 nano)
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Entraîner le modèle
        print(f"Démarrage de l'entraînement du modèle YOLO...")
        results = self.yolo_model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=20,
            project=str(model_dir),
            name='train',
            device='0' if torch.cuda.is_available() else 'cpu'
        )
        
        # Sauvegarder le modèle
        best_model_path = model_dir / 'train' / 'weights' / 'best.pt'
        final_model_path = model_dir / 'yolo_spores_model.pt'
        shutil.copy(str(best_model_path), str(final_model_path))
        
        print(f"Modèle YOLO entraîné et sauvegardé dans {final_model_path}")
        return final_model_path
    
    def train_unet_model(self, image_dir, mask_dir, epochs=50, img_size=(256, 256)):
        """
        Entraîne un modèle U-Net pour la segmentation des spores.
        
        Args:
            image_dir (str): Répertoire contenant les images d'entraînement.
            mask_dir (str): Répertoire contenant les masques de segmentation.
            epochs (int, optional): Nombre d'époques d'entraînement. Par défaut 50.
            img_size (tuple, optional): Taille des images d'entraînement. Par défaut (256, 256).
        
        Returns:
            Path: Chemin du modèle entraîné.
        
        Example:
            >>> processor = HyphSporeaProcessor("/chemin/vers/projet")
            >>> model_path = processor.train_unet_model("data/proc_data/images", "data/proc_data/masks")
        """
        # Créer un dossier pour le modèle avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"unet_model_{timestamp}"
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Chemin pour sauvegarder le modèle
        model_path = model_dir / 'unet_spores_model.h5'
        
        # Entraîner le modèle U-Net
        print(f"Démarrage de l'entraînement du modèle U-Net...")
        self.unet_model = train_unet(
            image_dir=str(image_dir),
            mask_dir=str(mask_dir),
            output_model_path=str(model_path)
        )
        
        print(f"Modèle U-Net entraîné et sauvegardé dans {model_path}")
        return model_path
    
    def process_sample(self, sample_dir, model_path=None, output_dir=None):
        """
        Traite un échantillon complet avec le modèle YOLO ou U-Net.
        
        Traite toutes les images d'un échantillon en appliquant le modèle spécifié,
        extrait les détections et génère des statistiques.
        
        Args:
            sample_dir (str): Répertoire contenant les images de l'échantillon.
            model_path (str, optional): Chemin du modèle à utiliser. Si None, utilise le modèle chargé.
            output_dir (str, optional): Répertoire de sortie pour les résultats.
        
        Returns:
            Path: Chemin du répertoire contenant les résultats.
        
        Example:
            >>> processor = HyphSporeaProcessor("/chemin/vers/projet")
            >>> results_dir = processor.process_sample("data/raw_data/T1_ALAC_C6_1", "models/yolo_model.pt")
        """
        sample_dir = Path(sample_dir)
        
        # Vérifier si un modèle est spécifié ou utiliser le modèle par défaut
        if model_path:
            model_path = Path(model_path)
            if str(model_path).endswith('.pt'):
                model = YOLO(model_path)
                model_type = 'yolo'
            elif str(model_path).endswith('.h5'):
                model = load_model(model_path)
                model_type = 'unet'
            else:
                raise ValueError("Type de modèle non reconnu")
        elif self.yolo_model:
            model = self.yolo_model
            model_type = 'yolo'
        elif self.unet_model:
            model = self.unet_model
            model_type = 'unet'
        else:
            raise ValueError("Aucun modèle disponible")
        
        # Configurer le répertoire de sortie
        if output_dir is None:
            sample_name = sample_dir.name
            output_dir = self.output_dir / f"processed_{sample_name}"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Récupérer tous les fichiers d'images dans le répertoire de l'échantillon
        image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        # Dictionnaire pour stocker les résultats de détection
        detections = {}
        
        print(f"Traitement de {len(image_files)} images dans {sample_dir}...")
        
        # Traiter chaque image
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(sample_dir, img_file)
            
            # Prétraiter l'image
            preprocessed_img, _ = enhanced_preprocess_image(img_path)
            
            # Détecter les spores selon le type de modèle
            if model_type == 'yolo':
                results = model.predict(preprocessed_img, conf=0.25, save=True, project=str(output_dir), name=img_file.split('.')[0])
                
                # Extraire les détections
                detections[img_file] = []
                for det in results[0].boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = det
                    detections[img_file].append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class': int(cls),
                        'width': float(x2) - float(x1),
                        'height': float(y2) - float(y1),
                        'area': (float(x2) - float(x1)) * (float(y2) - float(y1))
                    })
            
            elif model_type == 'unet':
                result_img, mask, contours = predict_with_unet(model, img_path, output_dir=str(output_dir))
                
                # Extraire les détections
                detections[img_file] = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    detections[img_file].append({
                        'bbox': [float(x), float(y), float(x+w), float(y+h)],
                        'width': float(w),
                        'height': float(h),
                        'area': float(cv2.contourArea(contour))
                    })
            
            # Afficher la progression
            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"Progression: {i + 1}/{len(image_files)} images traitées")
        
        # Sauvegarder les résultats de détection
        detection_file = output_dir / 'detections.json'
        with open(detection_file, 'w') as f:
            json.dump(detections, f, indent=2)
        
        # Analyser les résultats
        self.analyze_results(detections, output_dir)
        
        print(f"Traitement terminé. Résultats sauvegardés dans {output_dir}")
        return output_dir
    
    def analyze_results(self, detections, output_dir):
        """
        Analyse les résultats de détection et génère des statistiques.
        
        Args:
            detections (dict): Dictionnaire contenant les détections par image.
            output_dir (Path): Répertoire de sortie pour les analyses.
        
        Returns:
            dict: Statistiques calculées pour l'échantillon.
        
        Example:
            >>> processor = HyphSporeaProcessor("/chemin/vers/projet")
            >>> with open('detections.json', 'r') as f:
            ...     detections = json.load(f)
            >>> stats = processor.analyze_results(detections, Path("outputs/analysis"))
        """
        output_dir = Path(output_dir)
        
        # Créer un répertoire pour les analyses
        analysis_dir = output_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        # Extraire les informations de l'échantillon depuis le nom du répertoire
        sample_name = output_dir.name.replace('processed_', '')
        parts = sample_name.split('_')
        
        strain = parts[0] + '_' + parts[1] if len(parts) > 1 else parts[0]
        condition = parts[2] if len(parts) > 2 else 'unknown'
        replicate = parts[3] if len(parts) > 3 else '1'
        
        # Compter le nombre de spores détectées par classe
        class_counts = {}
        total_spores = 0
        
        for img_file, img_detections in detections.items():
            for det in img_detections:
                # Si la classe est disponible (YOLO)
                if 'class' in det:
                    cls = det['class']
                    if cls not in class_counts:
                        class_counts[cls] = 0
                    class_counts[cls] += 1
                    total_spores += 1
                else:
                    # Si seule la détection est disponible (U-Net)
                    if 'unknown' not in class_counts:
                        class_counts['unknown'] = 0
                    class_counts['unknown'] += 1
                    total_spores += 1
        
        # Créer un DataFrame pour les statistiques
        stats = {
            'sample': sample_name,
            'strain': strain,
            'condition': condition,
            'replicate': replicate,
            'total_images': len(detections),
            'total_spores_detected': total_spores
        }
        
        # Ajouter les comptages par classe
        for cls, count in class_counts.items():
            stats[f'class_{cls}_count'] = count
        
        # Calculer les statistiques morphologiques
        widths = []
        heights = []
        areas = []
        
        for img_detections in detections.values():
            for det in img_detections:
                widths.append(det['width'])
                heights.append(det['height'])
                areas.append(det['area'])
        
        if widths:
            stats['avg_width'] = np.mean(widths)
            stats['avg_height'] = np.mean(heights)
            stats['avg_area'] = np.mean(areas)
            stats['median_area'] = np.median(areas)
            stats['min_area'] = np.min(areas)
            stats['max_area'] = np.max(areas)
        
        # Sauvegarder les statistiques
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(analysis_dir / 'sample_stats.csv', index=False)
        
        # Créer un graphique de distribution des tailles
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(areas, bins=30, alpha=0.7, color='blue')
        plt.title('Distribution des surfaces de spores')
        plt.xlabel('Surface (pixels²)')
        plt.ylabel('Nombre de spores')
        
        plt.subplot(1, 2, 2)
        plt.scatter(widths, heights, alpha=0.5)
        plt.title('Rapport largeur/hauteur des spores')
        plt.xlabel('Largeur (pixels)')
        plt.ylabel('Hauteur (pixels)')
        
        plt.tight_layout()
        plt.savefig(analysis_dir / 'size_distribution.png')
        
        # Créer un graphique de comptage par classe
        if class_counts and not all(k == 'unknown' for k in class_counts.keys()):
            plt.figure(figsize=(10, 6))
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            plt.bar(classes, counts, color='skyblue')
            plt.title('Nombre de spores par classe')
            plt.xlabel('Classe')
            plt.ylabel('Nombre de spores')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(analysis_dir / 'class_distribution.png')
        
        print(f"Analyse terminée. Statistiques sauvegardées dans {analysis_dir}")
        
        # Ajouter les résultats au dictionnaire global
        self.results[sample_name] = stats
        
        return stats
    
    def compare_samples(self, output_dir=None):
        """
        Compare les résultats de plusieurs échantillons.
        
        Génère des graphiques et tableaux comparatifs entre différents échantillons.
        
        Args:
            output_dir (Path, optional): Répertoire de sortie pour la comparaison.
        
        Returns:
            Path: Chemin du répertoire contenant les résultats de la comparaison.
        
        Example:
            >>> processor = HyphSporeaProcessor("/chemin/vers/projet")
            >>> # Après avoir traité plusieurs échantillons
            >>> comparison_dir = processor.compare_samples()
        """
        if not self.results:
            print("Aucun résultat à comparer.")
            return None
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_dir / f"comparison_{timestamp}"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Créer un DataFrame avec tous les résultats
        all_results = pd.DataFrame(self.results.values())
        
        # Sauvegarder les résultats combinés
        all_results.to_csv(output_dir / 'all_samples_results.csv', index=False)
        
        # Créer un graphique comparatif du nombre de spores par échantillon
        plt.figure(figsize=(12, 6))
        
        samples = all_results['sample'].tolist()
        spore_counts = all_results['total_spores_detected'].tolist()
        
        plt.bar(samples, spore_counts, color='skyblue')
        plt.title('Nombre total de spores par échantillon')
        plt.xlabel('Échantillon')
        plt.ylabel('Nombre de spores')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'samples_comparison.png')
        
        # Créer un graphique comparatif par condition
        if 'condition' in all_results.columns:
            conditions = all_results['condition'].unique()
            
            plt.figure(figsize=(12, 6))
            
            for condition in conditions:
                condition_data = all_results[all_results['condition'] == condition]
                plt.scatter(
                    condition_data['strain'],
                    condition_data['total_spores_detected'],
                    label=condition,
                    alpha=0.7,
                    s=100
                )
            
            plt.title('Nombre de spores par souche et condition')
            plt.xlabel('Souche')
            plt.ylabel('Nombre de spores')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'strain_condition_comparison.png')
        
        print(f"Comparaison terminée. Résultats sauvegardés dans {output_dir}")
        return output_dir
