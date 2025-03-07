"""
Module de prétraitement d'images pour le projet HYPH-SPOREA.

Ce module contient des fonctions pour améliorer les images de spores d'hyphomycètes
afin de faciliter leur détection et segmentation.
"""

import cv2
import numpy as np
from skimage import exposure, filters, morphology
import os
from pathlib import Path


def enhanced_preprocess_image(image_path, save=False, output_path=None, intensity='medium'):
    """
    Prétraite une image pour améliorer la visibilité des spores.
    
    Applique plusieurs techniques de traitement d'image pour isoler les spores
    sur fond violet:
    1. Séparation des canaux de couleur pour détecter les teintes violettes
    2. Amélioration du contraste avec CLAHE
    3. Réduction du bruit
    4. Binarisation adaptative
    5. Opérations morphologiques
    6. Suppression des petits objets
    
    Args:
        image_path (str): Chemin de l'image à prétraiter
        save (bool, optional): Si True, sauvegarde l'image prétraitée. Par défaut False.
        output_path (str, optional): Chemin de sortie pour l'image prétraitée.
            Requis si save=True.
        intensity (str, optional): Intensité du prétraitement ('light', 'medium', 'strong'). 
            Par défaut 'medium'.
    
    Returns:
        tuple: (image prétraitée, masque binaire)
    
    Example:
        >>> processed_img, mask = enhanced_preprocess_image(
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     save=True,
        ...     output_path="data/proc_data/preprocessed/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg"
        ... )
    """
    # Chargement de l'image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
    
    # Paramètres selon l'intensité
    if intensity == 'very_light':
        blur_size = 3
        thresh_blocksize = 30
        thresh_c = 7
        morph_iterations = 0
        min_size = 15
    elif intensity == 'light':
        blur_size = 3
        thresh_blocksize = 25
        thresh_c = 7
        morph_iterations = 0
        min_size = 20
    elif intensity == 'medium':
        blur_size = 3
        thresh_blocksize = 15
        thresh_c = 5
        morph_iterations = 1
        min_size = 30
    else:  # strong
        blur_size = 5
        thresh_blocksize = 11
        thresh_c = 3
        morph_iterations = 2
        min_size = 50
    
    # Séparation des canaux de couleur (pour mieux détecter les teintes violettes)
    b, g, r = cv2.split(image)
    
    # Utilisation du canal bleu et rouge pour améliorer la détection des teintes violettes
    # (le violet est un mélange de bleu et rouge)
    violet_channel = cv2.addWeighted(b, 0.6, r, 0.4, 0)
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(violet_channel)
    
    # Réduction du bruit
    denoised = cv2.GaussianBlur(enhanced, (blur_size, blur_size), 0)
    
    # Binarisation adaptative pour gérer les variations de luminosité
    binary = cv2.adaptiveThreshold(
        denoised, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        thresh_blocksize, 
        thresh_c
    )
    
    # Opérations morphologiques (optionnelles selon l'intensité)
    if morph_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    
    # Suppression des petits objets
    cleaned = morphology.remove_small_objects(
        binary.astype(bool), 
        min_size=min_size, 
        connectivity=2
    ).astype(np.uint8) * 255
    
    # Résultat final avec fond original conservé pour éviter la fragmentation
    # Utiliser le masque uniquement pour sélectionner les zones d'intérêt
    result = image.copy()
    
    # Fond blanc
    white_background = np.ones_like(image) * 255
    final_image = np.where(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR) == 0, white_background, result)
    
    if save:
        if output_path is None:
            raise ValueError("output_path doit être spécifié si save=True")
        
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Sauvegarder l'image prétraitée
        cv2.imwrite(output_path, final_image)
    
    # Retourner à la fois l'image finale et le masque
    return final_image, cleaned


def batch_preprocess_directory(input_dir, output_dir, file_pattern="*.jpeg", intensity='medium'):
    """
    Prétraite toutes les images d'un répertoire correspondant au motif spécifié.
    
    Args:
        input_dir (str or Path): Répertoire contenant les images à prétraiter
        output_dir (str or Path): Répertoire de sortie pour les images prétraitées
        file_pattern (str, optional): Motif de fichier à traiter. Par défaut "*.jpeg"
        intensity (str, optional): Intensité du prétraitement ('light', 'medium', 'strong'). Par défaut 'medium'.
    
    Returns:
        Path: Chemin du répertoire de sortie
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Créer le répertoire de sortie
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Liste des fichiers à traiter
    files = list(input_dir.glob(file_pattern))
    
    if not files:
        print(f"Aucun fichier correspondant au motif '{file_pattern}' trouvé dans {input_dir}")
        return output_dir
    
    print(f"Prétraitement de {len(files)} images...")
    
    # Prétraitement de chaque image
    for i, file_path in enumerate(files):
        # Chemin de sortie
        rel_path = file_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        # Créer le répertoire parent si nécessaire
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Prétraiter et sauvegarder l'image
        try:
            # Passer le paramètre intensity à enhanced_preprocess_image
            enhanced_preprocess_image(
                str(file_path), 
                save=True, 
                output_path=str(output_path),
                intensity=intensity
            )
            
            # Afficher la progression
            if (i + 1) % 10 == 0 or i == len(files) - 1:
                print(f"Progression: {i + 1}/{len(files)} images traitées")
                
        except Exception as e:
            print(f"Erreur lors du prétraitement de {file_path}: {str(e)}")
    
    print(f"Prétraitement terminé. Images sauvegardées dans {output_dir}")
    return output_dir


def create_binary_masks(input_dir, output_dir, file_pattern="*.jpeg"):
    """
    Crée des masques binaires à partir des images prétraitées.
    
    Ces masques peuvent être utilisés pour l'entraînement du modèle U-Net.
    
    Args:
        input_dir (str or Path): Répertoire contenant les images prétraitées
        output_dir (str or Path): Répertoire de sortie pour les masques
        file_pattern (str, optional): Motif de fichier à traiter. Par défaut "*.jpeg"
    
    Returns:
        int: Nombre de masques générés
    
    Example:
        >>> n_masks = create_binary_masks(
        ...     "data/proc_data/preprocessed/T1_ALAC_C6_1",
        ...     "data/proc_data/masks/T1_ALAC_C6_1"
        ... )
        >>> print(f"{n_masks} masques générés")
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Créer le répertoire de sortie
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Liste des fichiers à traiter
    files = list(input_dir.glob(file_pattern))
    
    if not files:
        print(f"Aucun fichier correspondant au motif '{file_pattern}' trouvé dans {input_dir}")
        return 0
    
    print(f"Génération de {len(files)} masques binaires...")
    
    # Génération de masques pour chaque image
    for i, file_path in enumerate(files):
        try:
            # Charger l'image
            image = cv2.imread(str(file_path))
            
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Appliquer un seuillage pour obtenir un masque binaire
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Appliquer des opérations morphologiques pour améliorer le masque
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Chemin de sortie
            rel_path = file_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.png')
            
            # Créer le répertoire parent si nécessaire
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Sauvegarder le masque
            cv2.imwrite(str(output_path), mask)
            
            # Afficher la progression
            if (i + 1) % 10 == 0 or i == len(files) - 1:
                print(f"Progression: {i + 1}/{len(files)} masques générés")
                
        except Exception as e:
            print(f"Erreur lors de la génération du masque pour {file_path}: {str(e)}")
    
    print(f"Génération de masques terminée. Masques sauvegardés dans {output_dir}")
    return len(files)


def create_visualization(image_path, mask_path=None, output_path=None):
    """
    Crée une visualisation de l'image avec son masque superposé.
    
    Args:
        image_path (str or Path): Chemin de l'image originale
        mask_path (str or Path, optional): Chemin du masque binaire.
            Si None, génère le masque à partir de l'image.
        output_path (str or Path, optional): Chemin de sortie pour la visualisation.
            Si None, ne sauvegarde pas l'image.
    
    Returns:
        numpy.ndarray: Image avec masque superposé
    
    Example:
        >>> visualization = create_visualization(
        ...     "data/proc_data/preprocessed/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.jpeg",
        ...     "data/proc_data/masks/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.png",
        ...     "outputs/visualizations/T1_ALAC_C6_1_000156_vis.jpeg"
        ... )
    """
    # Charger l'image
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
    
    # Charger ou générer le masque
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Impossible de lire le masque: {mask_path}")
    else:
        # Générer le masque à partir de l'image
        _, mask = enhanced_preprocess_image(str(image_path))
    
    # Créer une copie de l'image
    visualization = image.copy()
    
    # Appliquer une teinte verte sur les objets détectés
    green_overlay = np.zeros_like(image)
    green_overlay[:, :, 1] = mask  # Canal vert
    
    # Superposer le masque avec transparence
    alpha = 0.3
    visualization = cv2.addWeighted(visualization, 1, green_overlay, alpha, 0)
    
    # Dessiner les contours en vert vif
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
    
    # Sauvegarder la visualisation si un chemin de sortie est spécifié
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), visualization)
    
    return visualization