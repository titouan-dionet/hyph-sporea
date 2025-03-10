"""
Module pour la gestion des modèles YOLO dans le projet HYPH-SPOREA.

Ce module contient des fonctions pour entraîner, valider et utiliser
les modèles YOLO pour la détection de spores d'hyphomycètes.
"""

import os
import torch
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def load_yolo_model(path, model="yolo11m.pt", download=True):
    """
    Vérifie si un modèle YOLO existe, le télécharge si nécessaire, et renvoie son chemin.
    
    Args:
        path (str): Chemin du répertoire contenant le modèle YOLO.
        model (str, optional): Nom du modèle YOLO. Par défaut "yolo11m.pt".
        download (bool, optional): Si True et que le modèle n'existe pas, le télécharge.
            Par défaut True.
    
    Returns:
        str: Chemin complet du modèle YOLO
    """
    
    # Créer le répertoire s'il n'existe pas
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Répertoire créé: {path}")
    
    # Chemin complet du modèle
    model_path = os.path.join(path, model)
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print(f"Le modèle {model} n'existe pas dans {path}")
        
        if download:
            print(f"Téléchargement du modèle {model}...")
            try:
                # Utiliser l'API YOLO pour télécharger le modèle
                # Pour YOLO v8, on peut utiliser YOLO(model) qui téléchargera automatiquement si besoin
                temp_model = YOLO(model)
                # Sauvegarder le modèle au bon endroit
                temp_model.model.save(model_path)
                print(f"Modèle téléchargé et sauvegardé dans {model_path}")
            except Exception as e:
                print(f"Erreur lors du téléchargement du modèle: {str(e)}")
                return None
        else:
            print("Téléchargement désactivé, impossible de continuer sans le modèle.")
            return None
    
    return model_path


def train_yolo_model(model_path, data_yaml_path, epochs=100, img_size=640, batch_size=16, project_dir=None,
                    patience=20, device=None, output_dir=None):
    """
    Entraîne un modèle YOLO sur les données annotées.
    
    Args:
        model_path (str): Chemin du modèle YOLO à utiliser. 
        data_yaml_path (str): Chemin du fichier data.yaml contenant les configurations d'entraînement.
        epochs (int, optional): Nombre d'époques d'entraînement. Par défaut 100.
        img_size (int, optional): Taille des images d'entraînement. Par défaut 640.
        batch_size (int, optional): Taille des batchs d'entraînement. Par défaut 16.
        project_dir (str, optional): Répertoire du projet pour les sorties. Si None, utilise 'outputs'.
        patience (int, optional): Patience pour l'arrêt anticipé. Par défaut 20.
        device (str, optional): Dispositif d'entraînement ('0', 'cpu', etc.). Si None, détecte automatiquement.
        output_dir (str, optional): Répertoire spécifique pour le modèle de sortie.
    
    Returns:
        str: Chemin du modèle entraîné
    
    Example:
        >>> model_path = train_yolo_model("data/proc_data/yolo_annotations/data.yaml", epochs=50)
        >>> print(f"Modèle entraîné: {model_path}")
    """
    # Configuration du répertoire du projet
    if output_dir:
        project_dir = output_dir
    
    if project_dir is None:
        project_dir = "outputs/models"
    
    project_dir = Path(project_dir)
    project_dir.mkdir(exist_ok=True, parents=True)
    
    # Choix du dispositif d'entraînement
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Initialiser le modèle YOLO (YOLO11m)
    model = YOLO(model_path)
    
    # Entraîner le modèle
    print(f"Démarrage de l'entraînement du modèle YOLO...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience,
        project=str(project_dir),
        name='yolo_train',
        device=device
    )
    
    # Récupérer le chemin du meilleur modèle
    best_model_path = project_dir / 'yolo_train' / 'weights' / 'best.pt'
    final_model_path = project_dir / 'yolo_spores_model.pt'
    
    # Copier le meilleur modèle
    if best_model_path.exists():
        shutil.copy(str(best_model_path), str(final_model_path))
        print(f"Modèle YOLO entraîné et sauvegardé dans {final_model_path}")
    else:
        print(f"Attention: Le fichier {best_model_path} n'existe pas. L'entraînement a-t-il été interrompu?")
        # Utiliser le dernier modèle si le meilleur n'est pas disponible
        last_model_path = project_dir / 'yolo_train' / 'weights' / 'last.pt'
        if last_model_path.exists():
            shutil.copy(str(last_model_path), str(final_model_path))
            print(f"Modèle YOLO (dernier) sauvegardé dans {final_model_path}")
        else:
            raise FileNotFoundError(f"Aucun modèle trouvé dans {project_dir / 'yolo_train' / 'weights'}")
    
    return str(final_model_path)


def validate_yolo_model(model_path, data_yaml_path, output_dir=None):
    """
    Valide un modèle YOLO sur un ensemble de données.
    
    Args:
        model_path (str): Chemin du modèle YOLO
        data_yaml_path (str): Chemin du fichier data.yaml
        output_dir (str, optional): Répertoire de sortie pour les résultats de validation
    
    Returns:
        dict: Résultats de la validation
    
    Example:
        >>> results = validate_yolo_model(
        ...     "outputs/models/yolo_spores_model.pt",
        ...     "data/proc_data/yolo_annotations/data.yaml",
        ...     "outputs/validation"
        ... )
        >>> print(f"mAP50: {results['metrics/mAP50(B)']:.4f}")
    """
    # Charger le modèle
    model = YOLO(model_path)
    
    # Configuration du répertoire de sortie
    if output_dir is None:
        output_dir = "outputs/validation"
    
    # Charger le fichier data.yaml pour vérifier les chemins
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Validation
    results = model.val(
        data=data_yaml_path,
        imgsz=640,
        batch=16,
        project=output_dir,
        name='yolo_validation'
    )
    
    return results


def predict_with_yolo(model_path, image_path, conf=0.25, output_dir=None, save=True):
    """
    Effectue des prédictions sur une image avec un modèle YOLO.
    
    Args:
        model_path (str): Chemin du modèle YOLO
        image_path (str): Chemin de l'image à analyser
        conf (float, optional): Seuil de confiance pour les détections. Par défaut 0.25.
        output_dir (str, optional): Répertoire de sortie pour les résultats
        save (bool, optional): Si True, sauvegarde les résultats. Par défaut True.
    
    Returns:
        tuple: (résultats YOLO, liste de détections formatées)
    
    Example:
        >>> results, detections = predict_with_yolo(
        ...     "outputs/models/yolo_spores_model.pt",
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     output_dir="outputs/predictions"
        ... )
        >>> print(f"Nombre de détections: {len(detections)}")
    """
    # Charger le modèle
    model = YOLO(model_path)
    
    # Prédiction
    results = model.predict(
        source=image_path,
        conf=conf,
        save=save,
        project=output_dir if output_dir else "outputs/predictions",
        name=Path(image_path).stem
    )
    
    # Formater les détections
    detections = []
    
    if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class': int(cls),
                'width': float(x2) - float(x1),
                'height': float(y2) - float(y1),
                'area': (float(x2) - float(x1)) * (float(y2) - float(y1))
            })
    
    return results, detections


def batch_predict_with_yolo(model_path, image_dir, output_dir=None, conf=0.25, pattern="*.jpeg"):
    """
    Effectue des prédictions sur un lot d'images avec un modèle YOLO.
    
    Args:
        model_path (str): Chemin du modèle YOLO
        image_dir (str): Répertoire contenant les images à analyser
        output_dir (str, optional): Répertoire de sortie pour les résultats
        conf (float, optional): Seuil de confiance pour les détections. Par défaut 0.25.
        pattern (str, optional): Motif pour filtrer les fichiers. Par défaut "*.jpeg".
    
    Returns:
        dict: Dictionnaire de détections par image
    
    Example:
        >>> all_detections = batch_predict_with_yolo(
        ...     "outputs/models/yolo_spores_model.pt",
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1",
        ...     "outputs/predictions/T1_ALAC_C6_1"
        ... )
        >>> total_detections = sum(len(dets) for dets in all_detections.values())
        >>> print(f"Total des détections: {total_detections}")
    """
    # Charger le modèle
    model = YOLO(model_path)
    
    # Configuration du répertoire de sortie
    if output_dir is None:
        output_dir = "outputs/predictions"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Lister les images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob(pattern))
    
    if not image_files:
        print(f"Aucune image correspondant au motif '{pattern}' trouvée dans {image_dir}")
        return {}
    
    print(f"Traitement de {len(image_files)} images...")
    
    # Dictionnaire pour stocker les résultats
    all_detections = {}
    
    # Traiter chaque image
    for i, image_file in enumerate(image_files):
        try:
            # Afficher la progression
            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"Progression: {i + 1}/{len(image_files)} images traitées")
            
            # Effectuer les prédictions
            img_output_dir = output_dir / image_file.stem
            _, detections = predict_with_yolo(
                model_path, 
                str(image_file),
                conf=conf,
                output_dir=str(img_output_dir)
            )
            
            # Stocker les détections
            all_detections[image_file.name] = detections
            
        except Exception as e:
            print(f"Erreur lors du traitement de {image_file}: {str(e)}")
    
    return all_detections


def visualize_yolo_training_results(training_dir, output_path=None):
    """
    Visualise les résultats d'entraînement d'un modèle YOLO.
    
    Args:
        training_dir (str): Répertoire contenant les résultats d'entraînement
        output_path (str, optional): Chemin pour sauvegarder la figure
    
    Returns:
        matplotlib.figure.Figure: Figure créée
    
    Example:
        >>> fig = visualize_yolo_training_results(
        ...     "outputs/models/yolo_train",
        ...     "outputs/visualizations/training_results.png"
        ... )
    """
    results_file = Path(training_dir) / "results.csv"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Fichier de résultats non trouvé: {results_file}")
    
    # Charger les résultats
    results = np.loadtxt(results_file, delimiter=",", skiprows=1)
    
    # Extraire les données
    epochs = results[:, 0]
    train_loss = results[:, 1]
    val_loss = results[:, 2]
    precision = results[:, 3]
    recall = results[:, 4]
    map50 = results[:, 5]
    map50_95 = results[:, 6]
    
    # Créer la figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Pertes
    axs[0, 0].plot(epochs, train_loss, label='Train Loss')
    axs[0, 0].plot(epochs, val_loss, label='Validation Loss')
    axs[0, 0].set_title('Pertes')
    axs[0, 0].set_xlabel('Époques')
    axs[0, 0].set_ylabel('Perte')
    axs[0, 0].legend()
    axs[0, 0].grid(alpha=0.3)
    
    # Précision et Rappel
    axs[0, 1].plot(epochs, precision, label='Précision')
    axs[0, 1].plot(epochs, recall, label='Rappel')
    axs[0, 1].set_title('Précision et Rappel')
    axs[0, 1].set_xlabel('Époques')
    axs[0, 1].set_ylabel('Valeur')
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)
    
    # mAP
    axs[1, 0].plot(epochs, map50, label='mAP@0.5')
    axs[1, 0].plot(epochs, map50_95, label='mAP@0.5:0.95')
    axs[1, 0].set_title('Mean Average Precision (mAP)')
    axs[1, 0].set_xlabel('Époques')
    axs[1, 0].set_ylabel('mAP')
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3)
    
    # Convergence globale
    axs[1, 1].plot(epochs, train_loss/train_loss[0], label='Train Loss (normalisé)')
    axs[1, 1].plot(epochs, val_loss/val_loss[0], label='Val Loss (normalisé)')
    axs[1, 1].plot(epochs, map50/max(map50), label='mAP@0.5 (normalisé)')
    axs[1, 1].set_title('Convergence')
    axs[1, 1].set_xlabel('Époques')
    axs[1, 1].set_ylabel('Valeur normalisée')
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def parse_yolo_predictions(results):
    """
    Parse les résultats de prédiction YOLO en un format facilement utilisable.
    
    Args:
        results: Résultats de prédiction YOLO
    
    Returns:
        list: Liste de dictionnaires contenant les informations de détection
    
    Example:
        >>> model = YOLO("outputs/models/yolo_spores_model.pt")
        >>> results = model.predict("path/to/image.jpeg")
        >>> detections = parse_yolo_predictions(results)
        >>> for det in detections:
        ...     print(f"Classe: {det['class']}, Confiance: {det['confidence']:.2f}")
    """
    detections = []
    
    if not results or len(results) == 0:
        return detections
    
    for result in results:
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Extraire les coordonnées
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Classe et confiance
                cls = int(box.cls)
                conf = float(box.conf)
                
                # Créer un dictionnaire de détection
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': int(cls),
                    'width': float(x2) - float(x1),
                    'height': float(y2) - float(y1),
                    'area': (float(x2) - float(x1)) * (float(y2) - float(y1))
                }
                
                detections.append(detection)
    
    return detections
