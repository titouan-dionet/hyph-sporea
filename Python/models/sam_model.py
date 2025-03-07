"""
Module pour l'utilisation du modèle SAM (Segment Anything Model) dans le projet HYPH-SPOREA.

Ce module contient des fonctions pour utiliser le modèle SAM pour la segmentation
des spores d'hyphomycètes.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage import measure
from skimage.feature import peak_local_max
from ultralytics import SAM

from ..image_processing.preprocessing import enhanced_preprocess_image


def load_sam_model(model_path="sam2.1_l.pt"):
    """
    Charge un modèle SAM.
    
    Args:
        model_path (str, optional): Chemin du modèle SAM.
            Par défaut "sam2.1_l.pt" (modèle large).
    
    Returns:
        SAM: Modèle SAM chargé
    
    Example:
        >>> model = load_sam_model("data/blank_models/sam2.1_b.pt")
        >>> print(model.model.names)  # Affiche les classes du modèle
    """
    model = SAM(model_path)
    return model


def detect_with_sam(image_path, model, output_dir=None, save=True, conf=0.1):
    """
    Détecte les spores dans une image avec le modèle SAM en utilisant des points d'ancrage
    basés sur le prétraitement de l'image.
    
    Args:
        image_path (str): Chemin de l'image à analyser
        model (SAM or str): Modèle SAM chargé ou chemin du modèle
        output_dir (str, optional): Répertoire de sortie pour les résultats
        save (bool, optional): Si True, sauvegarde les résultats. Par défaut True.
        conf (float, optional): Seuil de confiance pour les détections. Par défaut 0.1.
    
    Returns:
        tuple: (résultats SAM, image prétraitée, points d'ancrage)
    
    Example:
        >>> model = load_sam_model()
        >>> results, image, points = detect_with_sam(
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     model,
        ...     output_dir="outputs/predictions/sam"
        ... )
        >>> print(f"Nombre de points d'ancrage: {len(points)}")
    """
    # Charger le modèle si c'est un chemin
    if isinstance(model, str):
        model = load_sam_model(model)
    
    # Configuration du répertoire de sortie
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Prétraitement de l'image
    final_image, mask = enhanced_preprocess_image(image_path)
    
    # Utilisation du masque pour trouver des centres d'objets (points d'ancrage pour SAM)
    # Extraction des composantes connexes
    labeled_mask = measure.label(mask)
    props = measure.regionprops(labeled_mask)
    
    # Création de points d'ancrage pour SAM
    points = []
    labels = []
    
    for prop in props:
        # Filtrage par taille (ajustez selon la taille de vos spores)
        if prop.area > 30 and prop.area < 5000:
            y, x = prop.centroid
            points.append([int(x), int(y)])
            labels.append(1)  # 1 pour "c'est un objet"
    
    # Si aucun point n'est trouvé, utilisez une autre approche
    if len(points) == 0:
        # Utilisation de maxima locaux comme points d'ancrage
        distance = 10  # distance minimale entre les pics
        coordinates = peak_local_max(255 - mask, min_distance=distance)
        points = [[int(x), int(y)] for y, x in coordinates[:20]]  # limiter à 20 points
        labels = [1] * len(points)
    
    # Convertir en tableaux NumPy si nécessaire
    if points:
        points = np.array(points)
        labels = np.array(labels)
        
        # Prédiction avec SAM en utilisant les points comme prompts
        results = model.predict(
            source=final_image,
            save=save,
            project=output_dir if output_dir else "outputs/predictions/sam",
            name=Path(image_path).stem,
            points=points,
            point_labels=labels,
            conf=conf,  # Réduire le seuil de confiance
            retina_masks=True  # Pour une meilleure précision
        )
    else:
        # Pas de points trouvés, SAM ne peut pas fonctionner correctement
        print(f"Aucun point d'ancrage trouvé dans l'image {image_path}")
        results = None
    
    return results, final_image, points


def batch_detect_with_sam(model, image_dir, output_dir=None, pattern="*.jpeg", conf=0.1):
    """
    Effectue des détections sur un lot d'images avec le modèle SAM.
    
    Args:
        model (SAM or str): Modèle SAM chargé ou chemin du modèle
        image_dir (str): Répertoire contenant les images à analyser
        output_dir (str, optional): Répertoire de sortie pour les résultats
        pattern (str, optional): Motif pour filtrer les fichiers. Par défaut "*.jpeg".
        conf (float, optional): Seuil de confiance pour les détections. Par défaut 0.1.
    
    Returns:
        dict: Dictionnaire contenant les résultats par image
    
    Example:
        >>> results = batch_detect_with_sam(
        ...     "data/blank_models/sam2.1_b.pt",
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1",
        ...     "outputs/predictions/sam/T1_ALAC_C6_1"
        ... )
    """
    # Charger le modèle si c'est un chemin
    if isinstance(model, str):
        model = load_sam_model(model)
    
    # Configuration du répertoire de sortie
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Lister les images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob(pattern))
    
    if not image_files:
        print(f"Aucune image correspondant au motif '{pattern}' trouvée dans {image_dir}")
        return {}
    
    print(f"Traitement de {len(image_files)} images avec SAM...")
    
    # Dictionnaire pour stocker les résultats
    all_results = {}
    
    # Traiter chaque image
    for i, image_file in enumerate(image_files):
        try:
            # Créer un sous-répertoire pour cette image
            img_output_dir = output_dir / image_file.stem if output_dir else None
            
            # Détection
            results, _, points = detect_with_sam(
                str(image_file),
                model,
                output_dir=img_output_dir,
                conf=conf
            )
            
            # Stocker les résultats
            all_results[image_file.name] = {
                'results': results,
                'num_points': len(points) if points is not None else 0
            }
            
            # Afficher la progression
            if (i + 1) % 5 == 0 or i == len(image_files) - 1:
                print(f"Progression: {i + 1}/{len(image_files)} images traitées")
                
        except Exception as e:
            print(f"Erreur lors du traitement de {image_file}: {str(e)}")
    
    return all_results


def parse_sam_results(results):
    """
    Convertit les résultats du modèle SAM en un format standardisé de détections.
    
    Args:
        results: Résultats du modèle SAM
    
    Returns:
        list: Liste de dictionnaires contenant les informations de détection
    
    Example:
        >>> model = load_sam_model()
        >>> results, _, _ = detect_with_sam(
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     model
        ... )
        >>> detections = parse_sam_results(results)
        >>> for det in detections:
        ...     print(f"Surface: {det['area']:.2f} pixels²")
    """
    detections = []
    
    if not results or len(results) == 0:
        return detections
    
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            
            for i in range(len(masks)):
                # Obtenir le masque
                mask = masks[i].data[0].cpu().numpy().astype(np.uint8)
                
                # Trouver les contours du masque
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Créer une détection pour chaque contour
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    
                    # Filtrer par taille
                    if area > 30 and area < 5000:
                        detection = {
                            'bbox': [float(x), float(y), float(x+w), float(y+h)],
                            'width': float(w),
                            'height': float(h),
                            'area': float(area)
                        }
                        
                        detections.append(detection)
    
    return detections


def combine_sam_detections(images_results):
    """
    Combine les détections SAM de plusieurs images en un seul dictionnaire.
    
    Args:
        images_results (dict): Dictionnaire avec les résultats SAM par image
    
    Returns:
        dict: Dictionnaire des détections par image
    
    Example:
        >>> all_results = batch_detect_with_sam(...)
        >>> detections = combine_sam_detections(all_results)
        >>> total_detections = sum(len(dets) for dets in detections.values())
        >>> print(f"Total des détections: {total_detections}")
    """
    detections_dict = {}
    
    for image_name, result_data in images_results.items():
        results = result_data.get('results')
        
        if results:
            detections = parse_sam_results(results)
            detections_dict[image_name] = detections
        else:
            detections_dict[image_name] = []
    
    return detections_dict
