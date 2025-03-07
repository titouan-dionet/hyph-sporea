"""
Module de visualisation pour le projet HYPH-SPOREA.

Ce module contient des fonctions pour visualiser les résultats 
de détection et de segmentation des spores.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def draw_detections(image, detections, class_names=None, output_path=None):
    """
    Dessine les détections de spores sur une image.
    
    Args:
        image (numpy.ndarray or str): Image ou chemin de l'image
        detections (list): Liste de dictionnaires contenant les détections:
            - 'bbox': [x1, y1, x2, y2]
            - 'class': ID de classe (optionnel)
            - 'confidence': Score de confiance (optionnel)
        class_names (list, optional): Liste des noms de classes correspondant aux IDs.
            Si None, utilise les IDs numériques.
        output_path (str, optional): Chemin pour sauvegarder l'image.
            Si None, ne sauvegarde pas l'image.
    
    Returns:
        numpy.ndarray: Image avec les détections
    
    Example:
        >>> detections = [
        ...     {'bbox': [100, 150, 200, 250], 'class': 0, 'confidence': 0.95},
        ...     {'bbox': [300, 200, 400, 300], 'class': 1, 'confidence': 0.87}
        ... ]
        >>> class_names = ['ALAC', 'ANFI']
        >>> result = draw_detections(
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     detections,
        ...     class_names,
        ...     "outputs/detections/T1_ALAC_C6_1/T1_ALAC_C6_1_000156_det.jpeg"
        ... )
    """
    # Si l'image est un chemin, la charger
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        
        if image is None:
            raise ValueError(f"Impossible de lire l'image: {image}")
    
    # Créer une copie de l'image
    result = image.copy()
    
    # Couleurs pour les différentes classes (format BGR)
    colors = [
        (255, 0, 0),     # Bleu
        (0, 255, 0),     # Vert
        (0, 0, 255),     # Rouge
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Jaune
        (128, 128, 128)  # Gris
    ]
    
    # Dessiner chaque détection
    for detection in detections:
        bbox = detection['bbox']
        
        # Extraire la classe et le score si disponibles
        cls = detection.get('class', 0)
        conf = detection.get('confidence', None)
        
        # Déterminer la couleur
        color = colors[cls % len(colors)]
        
        # Extraire les coordonnées
        x1, y1, x2, y2 = [int(c) for c in bbox]
        
        # Dessiner le rectangle englobant
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Préparer le texte d'étiquette
        if class_names and cls < len(class_names):
            label = class_names[cls]
        else:
            label = f"Class {cls}"
            
        if conf is not None:
            label = f"{label}: {conf:.2f}"
        
        # Dessiner l'étiquette avec un fond
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Sauvegarder l'image si un chemin est spécifié
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), result)
    
    return result


def create_detection_grid(image_paths, detections_list, class_names=None, output_path=None, 
                         grid_size=(2, 2), figsize=(12, 12)):
    """
    Crée une grille d'images avec leurs détections pour visualisation.
    
    Args:
        image_paths (list): Liste des chemins d'images
        detections_list (list): Liste de listes de détections pour chaque image
        class_names (list, optional): Liste des noms de classes
        output_path (str, optional): Chemin pour sauvegarder la figure
        grid_size (tuple, optional): Taille de la grille (lignes, colonnes)
        figsize (tuple, optional): Taille de la figure en pouces
    
    Returns:
        matplotlib.figure.Figure: Figure créée
    
    Example:
        >>> image_paths = [f"data/proc_data/jpeg_images/sample_{i}.jpeg" for i in range(4)]
        >>> detections_list = [detections for _ in range(4)]  # Utiliser les mêmes détections
        >>> class_names = ['ALAC', 'ANFI', 'CLAQ', 'Debris']
        >>> fig = create_detection_grid(
        ...     image_paths,
        ...     detections_list,
        ...     class_names,
        ...     "outputs/visualizations/detection_grid.png"
        ... )
    """
    # Limiter le nombre d'images à la taille de la grille
    max_images = grid_size[0] * grid_size[1]
    image_paths = image_paths[:max_images]
    detections_list = detections_list[:max_images]
    
    # Créer la figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten()
    
    # Traiter chaque image
    for i, (image_path, detections) in enumerate(zip(image_paths, detections_list)):
        if i >= len(axes):
            break
            
        # Dessiner les détections
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                axes[i].text(0.5, 0.5, f"Image non trouvée:\n{image_path}", 
                           ha='center', va='center', fontsize=10)
                continue
                
            result = draw_detections(image, detections, class_names)
            
            # Convertir de BGR à RGB pour matplotlib
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            # Afficher l'image
            axes[i].imshow(result)
            axes[i].set_title(os.path.basename(image_path))
            axes[i].axis('off')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Erreur: {str(e)}", 
                       ha='center', va='center', fontsize=10)
    
    # Masquer les axes inutilisés
    for i in range(len(image_paths), len(axes)):
        axes[i].axis('off')
    
    # Ajuster l'espacement
    plt.tight_layout()
    
    # Sauvegarder si un chemin est spécifié
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_visualization(original_image, preprocessed_image, detections, 
                                  class_names=None, output_path=None):
    """
    Crée une visualisation comparative montrant l'image originale, 
    l'image prétraitée et les détections.
    
    Args:
        original_image (numpy.ndarray or str): Image originale ou son chemin
        preprocessed_image (numpy.ndarray or str): Image prétraitée ou son chemin
        detections (list): Liste des détections
        class_names (list, optional): Liste des noms de classes
        output_path (str, optional): Chemin pour sauvegarder la figure
    
    Returns:
        matplotlib.figure.Figure: Figure créée
    
    Example:
        >>> fig = create_comparison_visualization(
        ...     "data/raw_data/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     "data/proc_data/preprocessed/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     detections,
        ...     class_names,
        ...     "outputs/visualizations/comparison_T1_ALAC_C6_1_000156.png"
        ... )
    """
    # Charger les images si ce sont des chemins
    if isinstance(original_image, (str, Path)):
        original_image = cv2.imread(str(original_image))
        
        if original_image is None:
            raise ValueError(f"Impossible de lire l'image originale")
    
    if isinstance(preprocessed_image, (str, Path)):
        preprocessed_image = cv2.imread(str(preprocessed_image))
        
        if preprocessed_image is None:
            raise ValueError(f"Impossible de lire l'image prétraitée")
    
    # Dessiner les détections
    detections_image = draw_detections(preprocessed_image.copy(), detections, class_names)
    
    # Créer la figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Afficher l'image originale
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    # Afficher l'image prétraitée
    axes[1].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image prétraitée")
    axes[1].axis('off')
    
    # Afficher l'image avec détections
    axes[2].imshow(cv2.cvtColor(detections_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Détections")
    axes[2].axis('off')
    
    # Ajuster l'espacement
    plt.tight_layout()
    
    # Sauvegarder si un chemin est spécifié
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_detections_heatmap(image_shape, detections_list, output_path=None, alpha=0.7):
    """
    Crée une carte de chaleur des détections sur toutes les images.
    
    Args:
        image_shape (tuple): Forme de l'image (hauteur, largeur)
        detections_list (list): Liste de listes de détections pour chaque image
        output_path (str, optional): Chemin pour sauvegarder la figure
        alpha (float, optional): Transparence de la carte de chaleur
    
    Returns:
        matplotlib.figure.Figure: Figure créée
    
    Example:
        >>> shape = (1200, 1600)  # Forme de l'image (hauteur, largeur)
        >>> heatmap = plot_detections_heatmap(
        ...     shape,
        ...     all_detections,
        ...     "outputs/visualizations/detections_heatmap.png"
        ... )
    """
    # Créer une image vide pour la carte de chaleur
    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Ajouter chaque détection à la carte de chaleur
    for detections in detections_list:
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(c) for c in bbox]
            
            # Limiter les coordonnées à la taille de l'image
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Ajouter 1 pour chaque pixel de la boîte englobante
            heatmap[y1:y2, x1:x2] += 1
    
    # Normaliser la carte de chaleur
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Afficher la carte de chaleur
    im = ax.imshow(heatmap, cmap='hot', alpha=alpha)
    
    # Ajouter une barre de couleur
    cbar = plt.colorbar(im)
    cbar.set_label('Densité normalisée des détections')
    
    ax.set_title('Carte de chaleur des détections')
    ax.set_xlabel('Largeur (pixels)')
    ax.set_ylabel('Hauteur (pixels)')
    
    # Sauvegarder si un chemin est spécifié
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
