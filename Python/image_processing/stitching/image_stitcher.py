"""
Module d'assemblage d'images pour le projet HYPH-SPOREA.

Ce module contient des fonctions pour assembler des images de filtres circulaires
acquises par microscopie, en tenant compte du chevauchement entre les images et
de leur disposition spatiale spécifique.
"""

import os
import cv2
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime


def parse_grid_file(grid_file_path):
    """
    Parse le fichier de grille pour déterminer la position de chaque image.
    
    Args:
        grid_file_path (str): Chemin vers le fichier de grille
        
    Returns:
        tuple: (grid, dimensions) où:
            - grid est un dictionnaire {position_number: (row, col)}
            - dimensions est un tuple (rows, cols) représentant les dimensions de la grille
    """
    with open(grid_file_path, 'r') as f:
        lines = f.readlines()
    
    # Créer la grille
    grid = []
    for line in lines:
        row = line.strip().split('\t')
        grid.append(row)
    
    # Dimensions de la grille
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Créer un dictionnaire pour stocker les positions des images
    position_dict = {}
    
    # Parcourir la grille pour récupérer les positions
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 'x':
                position_dict[int(grid[i][j])] = (i, j)
    
    return position_dict, (rows, cols)


def detect_overlap(image1, image2, direction='horizontal', max_overlap=200, method=cv2.TM_CCOEFF_NORMED):
    """
    Détecte le chevauchement entre deux images adjacentes avec une précision améliorée.
    
    Args:
        image1 (numpy.ndarray): Première image
        image2 (numpy.ndarray): Deuxième image
        direction (str): Direction du chevauchement ('horizontal' ou 'vertical')
        max_overlap (int): Chevauchement maximal à vérifier (en pixels)
        method (int): Méthode de correspondance de modèle à utiliser
        
    Returns:
        int: Valeur estimée du chevauchement en pixels
    """
    h, w = image1.shape[:2]
    
    # Appliquer un prétraitement pour améliorer la précision de la détection
    # Convertir en niveaux de gris
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1.copy()
        gray2 = image2.copy()
    
    # Appliquer une égalisation d'histogramme pour améliorer le contraste
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    # Appliquer un filtre Sobel pour détecter les bords (améliore la correspondance)
    # Le filtre Sobel rend les caractéristiques structurelles plus distinctes
    sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    
    # Utiliser principalement le gradient dans la direction perpendiculaire au chevauchement
    if direction == 'horizontal':
        sobel1 = cv2.convertScaleAbs(sobelx1)  # Gradient horizontal pour chevauchement horizontal
        sobel2 = cv2.convertScaleAbs(sobelx2)
    else:  # vertical
        sobel1 = cv2.convertScaleAbs(sobely1)  # Gradient vertical pour chevauchement vertical
        sobel2 = cv2.convertScaleAbs(sobely2)
    
    # Définir les régions à comparer
    if direction == 'horizontal':
        # Prendre la partie droite de la première image
        # Réduire légèrement la taille du template pour plus de robustesse
        template_size = int(max_overlap * 0.95)  # Réduction légère pour éviter la surestimation
        template = sobel1[:, w-template_size:w]
        # Chercher dans la partie gauche de la deuxième image
        search_area = sobel2[:, 0:max_overlap*2]
    else:  # vertical
        # Prendre la partie inférieure de la première image
        template_size = int(max_overlap * 0.95)
        template = sobel1[h-template_size:h, :]
        # Chercher dans la partie supérieure de la deuxième image
        search_area = sobel2[0:max_overlap*2, :]
    
    # Affiner la correspondance à l'aide de plusieurs méthodes
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    best_overlap = 0
    best_confidence = -1
    
    for mtd in methods:
        # Appliquer la correspondance de modèle
        result = cv2.matchTemplate(search_area, template, mtd)
        _, confidence, _, max_loc = cv2.minMaxLoc(result)
        
        # Calculer l'overlap
        if direction == 'horizontal':
            overlap = template_size - max_loc[0]
        else:  # vertical
            overlap = template_size - max_loc[1]
        
        # Prendre l'overlap avec la meilleure confiance de correspondance
        if confidence > best_confidence:
            best_confidence = confidence
            best_overlap = overlap
    
    # Ajuster le résultat avec un facteur de correction pour compenser la réduction
    # du template et corriger toute sous-estimation
    correction_factor = 0.95  # Compenser la réduction du template (légèrement inférieur à 1 pour réduire l'overlap)
    adjusted_overlap = int(best_overlap * correction_factor)
    
    # S'assurer que l'overlap est dans les limites raisonnables
    if adjusted_overlap < 0:
        adjusted_overlap = 0
    elif adjusted_overlap > max_overlap:
        adjusted_overlap = max_overlap
    
    return adjusted_overlap


def estimate_overlap_from_sample(image_folder, pattern, num_samples=10, max_overlap=200):
    """
    Estime le chevauchement moyen en utilisant un échantillon d'images.
    
    Args:
        image_folder (str): Dossier contenant les images
        pattern (str): Modèle de nom de fichier (ex: 'T1_ALAC_*.tiff')
        num_samples (int): Nombre de paires d'images à échantillonner
        max_overlap (int): Chevauchement maximal à vérifier
        
    Returns:
        tuple: (horizontal_overlap, vertical_overlap) estimations moyennes
    """
    # Lister toutes les images
    if '*.' in pattern:
        # Si le pattern est explicite (ex: '*.tiff')
        extension = pattern.split('*.')[1]
        image_files = sorted([f for f in os.listdir(image_folder) 
                             if os.path.isfile(os.path.join(image_folder, f)) and 
                             f.lower().endswith(extension)])
    else:
        # Si le pattern est un format plus général (ex: "T1_ALAC_*")
        # Chercher à la fois les extensions .tif et .tiff pour les images TIFF
        if pattern.endswith('*.tif') or pattern.endswith('*.tiff'):
            extensions = ['.tif', '.tiff']
            base_pattern = pattern.split('*')[0]
            image_files = []
            for ext in extensions:
                image_files.extend(sorted([f for f in os.listdir(image_folder) 
                                 if os.path.isfile(os.path.join(image_folder, f)) and 
                                 f.startswith(base_pattern) and
                                 f.lower().endswith(ext)]))
        # Pour les autres types de patterns
        else:
            image_files = sorted([f for f in os.listdir(image_folder) 
                                 if os.path.isfile(os.path.join(image_folder, f)) and 
                                 re.match(pattern, f)])
    
    # Expression régulière pour extraire le numéro d'image
    num_pattern = re.compile(r'.*_(\d{6})\.')
    image_nums = {}
    
    for f in image_files:
        match = num_pattern.match(f)
        if match:
            num = int(match.group(1))
            image_nums[num] = f
    
    # Charger la grille pour trouver des images adjacentes
    # Nous utilisons le premier fichier comme base pour déterminer le motif
    base_name = os.path.basename(image_files[0]).split('_')[0:2]
    base_name = '_'.join(base_name)
    grid_file = os.path.join(os.path.dirname(image_folder), f"{base_name}_grid.txt")
    
    # Si le fichier de grille n'existe pas, utiliser un autre moyen de trouver les paires
    if not os.path.exists(grid_file):
        print(f"Fichier de grille '{grid_file}' non trouvé. Utilisation de paires consécutives.")
        
        # Utiliser des paires consécutives comme approximation
        h_overlaps = []
        v_overlaps = []
        
        # Pour les chevauchements horizontaux (supposer que les images consécutives sont adjacentes)
        for i in range(num_samples):
            idx = np.random.randint(0, len(image_files) - 1)
            img1 = cv2.imread(os.path.join(image_folder, image_files[idx]))
            img2 = cv2.imread(os.path.join(image_folder, image_files[idx + 1]))
            
            if img1 is not None and img2 is not None:
                overlap = detect_overlap(img1, img2, 'horizontal', max_overlap)
                if 0 < overlap < max_overlap:
                    h_overlaps.append(overlap)
        
        # Pour les chevauchements verticaux (supposer que les images espacées de ~18 sont verticalement adjacentes)
        step = 18  # Estimation basée sur la grille 18x24
        for i in range(num_samples):
            idx = np.random.randint(0, len(image_files) - step)
            img1 = cv2.imread(os.path.join(image_folder, image_files[idx]))
            img2 = cv2.imread(os.path.join(image_folder, image_files[idx + step]))
            
            if img1 is not None and img2 is not None:
                overlap = detect_overlap(img1, img2, 'vertical', max_overlap)
                if 0 < overlap < max_overlap:
                    v_overlaps.append(overlap)
    else:
        # Utiliser la grille pour trouver des paires adjacentes
        position_dict, (rows, cols) = parse_grid_file(grid_file)
        
        # Créer une matrice pour représenter la grille spatiale
        grid_matrix = np.full((rows, cols), -1, dtype=int)
        for img_num, (row, col) in position_dict.items():
            grid_matrix[row, col] = img_num
        
        # Échantillonner des paires horizontales
        h_overlaps = []
        v_overlaps = []
        
        # Pour les chevauchements horizontaux
        horizontal_pairs = []
        for r in range(rows):
            for c in range(cols-1):
                if grid_matrix[r, c] >= 0 and grid_matrix[r, c+1] >= 0:
                    horizontal_pairs.append((grid_matrix[r, c], grid_matrix[r, c+1]))
        
        if horizontal_pairs:
            # Échantillonner aléatoirement parmi les paires horizontales
            sample_indices = np.random.choice(len(horizontal_pairs), 
                                            min(num_samples, len(horizontal_pairs)), 
                                            replace=False)
            
            for idx in sample_indices:
                num1, num2 = horizontal_pairs[idx]
                if num1 in image_nums and num2 in image_nums:
                    img1_path = os.path.join(image_folder, image_nums[num1])
                    img2_path = os.path.join(image_folder, image_nums[num2])
                    
                    img1 = cv2.imread(img1_path)
                    img2 = cv2.imread(img2_path)
                    
                    if img1 is not None and img2 is not None:
                        overlap = detect_overlap(img1, img2, 'horizontal', max_overlap)
                        if 0 < overlap < max_overlap:
                            h_overlaps.append(overlap)
        
        # Pour les chevauchements verticaux
        vertical_pairs = []
        for r in range(rows-1):
            for c in range(cols):
                if grid_matrix[r, c] >= 0 and grid_matrix[r+1, c] >= 0:
                    vertical_pairs.append((grid_matrix[r, c], grid_matrix[r+1, c]))
        
        if vertical_pairs:
            # Échantillonner aléatoirement parmi les paires verticales
            sample_indices = np.random.choice(len(vertical_pairs), 
                                            min(num_samples, len(vertical_pairs)), 
                                            replace=False)
            
            for idx in sample_indices:
                num1, num2 = vertical_pairs[idx]
                if num1 in image_nums and num2 in image_nums:
                    img1_path = os.path.join(image_folder, image_nums[num1])
                    img2_path = os.path.join(image_folder, image_nums[num2])
                    
                    img1 = cv2.imread(img1_path)
                    img2 = cv2.imread(img2_path)
                    
                    if img1 is not None and img2 is not None:
                        overlap = detect_overlap(img1, img2, 'vertical', max_overlap)
                        if 0 < overlap < max_overlap:
                            v_overlaps.append(overlap)
    
    # Calculer les moyennes
    horizontal_overlap = int(np.mean(h_overlaps)) if h_overlaps else 105
    vertical_overlap = int(np.mean(v_overlaps)) if v_overlaps else 93
    
    print(f"Estimation du chevauchement horizontal: {horizontal_overlap} pixels (basé sur {len(h_overlaps)} échantillons)")
    print(f"Estimation du chevauchement vertical: {vertical_overlap} pixels (basé sur {len(v_overlaps)} échantillons)")
    
    # Visualiser la distribution des chevauchements
    if h_overlaps and v_overlaps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(h_overlaps, bins=10, alpha=0.7, color='blue')
        ax1.axvline(horizontal_overlap, color='red', linestyle='dashed', linewidth=2)
        ax1.set_title('Distribution du chevauchement horizontal')
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('Fréquence')
        
        ax2.hist(v_overlaps, bins=10, alpha=0.7, color='green')
        ax2.axvline(vertical_overlap, color='red', linestyle='dashed', linewidth=2)
        ax2.set_title('Distribution du chevauchement vertical')
        ax2.set_xlabel('Pixels')
        
        plt.tight_layout()
        plt.show()
    
    return horizontal_overlap, vertical_overlap


def stitch_images(image_folder, grid_file_path, output_path, h_overlap=105, v_overlap=93, 
                 sample_name=None, pixel_size_mm=0.264, use_tiff=True, 
                 show_grid=False, grid_color=(0, 0, 0), grid_alpha=0.4, grid_thickness=2,
                 show_numbers=False, numbers_color=(0, 0, 0), numbers_alpha=0.4):
    """
    Assemble les images en une seule image complète en utilisant la grille spécifiée.
    
    Args:
        image_folder (str): Dossier contenant les images
        grid_file_path (str): Chemin vers le fichier de grille
        output_path (str): Chemin de sortie pour l'image assemblée
        h_overlap (int): Chevauchement horizontal entre images adjacentes (en pixels)
        v_overlap (int): Chevauchement vertical entre images adjacentes (en pixels)
        sample_name (str, optional): Nom de l'échantillon, extrait du nom de fichier si None
        pixel_size_mm (float, optional): Taille du pixel en mm (0.0104 pouces = 0.264 mm)
        use_tiff (bool, optional): Si True, cherche des fichiers TIFF, sinon cherche des JPEG
        show_grid (bool, optional): Si True, affiche une grille entre les images
        grid_color (tuple, optional): Couleur de la grille (B, G, R)
        grid_alpha (float, optional): Transparence de la grille (0.0-1.0)
        grid_thickness (int, optional): Épaisseur des lignes de la grille en pixels
        show_numbers (bool, optional): Si True, affiche les numéros d'image
        numbers_color (tuple, optional): Couleur des numéros (B, G, R)
        numbers_alpha (float, optional): Transparence des numéros (0.0-1.0)
        
    Returns:
        numpy.ndarray: Image assemblée
    """
    # Déterminer les extensions en fonction du format d'image
    file_ext_list = [".tif", ".tiff"] if use_tiff else [".jpeg", ".jpg"]
    
    # Si le nom de l'échantillon n'est pas spécifié, essayer de le déterminer
    if sample_name is None:
        # Trouver le premier fichier pour déterminer le motif du nom
        files = []
        for ext in file_ext_list:
            files.extend([f for f in os.listdir(image_folder) if f.lower().endswith(ext)])
            
        if not files:
            raise ValueError(f"Aucun fichier {', '.join(file_ext_list)} trouvé dans le dossier {image_folder}")
        
        # Extraire le nom de l'échantillon (ex: "T1_ALAC")
        first_file = files[0]
        sample_name = "_".join(first_file.split("_")[:-1])
    
    # Lister toutes les images correspondant au motif pour chaque extension
    image_files = {}
    for ext in file_ext_list:
        for f in os.listdir(image_folder):
            if not f.lower().endswith(ext):
                continue
                
            # Vérifier si le fichier correspond au motif du nom d'échantillon
            name_parts = f.split('_')
            if len(name_parts) < 2 or "_".join(name_parts[:-1]) != sample_name:
                continue
            
            # Extraire le numéro d'image
            match = re.search(r'_(\d{6})\.', f)
            if match:
                num = int(match.group(1))
                image_files[num] = os.path.join(image_folder, f)
    
    # Vérifier si nous avons trouvé des images
    if not image_files:
        raise ValueError(f"Aucune image correspondant au motif {sample_name}_* trouvée dans {image_folder}")
    
    # Parser le fichier de grille
    position_dict, (rows, cols) = parse_grid_file(grid_file_path)
    
    # Charger une image pour obtenir ses dimensions
    first_image = cv2.imread(next(iter(image_files.values())))
    img_height, img_width = first_image.shape[:2]
    
    # Calcul du chevauchement total à partir des valeurs par côté
    total_h_overlap = h_overlap * 2  # Chevauchement horizontal total
    total_v_overlap = v_overlap * 2  # Chevauchement vertical total
    
    # Calculer les dimensions de l'image finale
    effective_width = img_width - total_h_overlap  # Largeur effective après chevauchement total
    effective_height = img_height - total_v_overlap  # Hauteur effective après chevauchement total
    
    # Trouver les limites de la grille (min et max row/col)
    min_row = min(pos[0] for pos in position_dict.values())
    max_row = max(pos[0] for pos in position_dict.values())
    min_col = min(pos[1] for pos in position_dict.values())
    max_col = max(pos[1] for pos in position_dict.values())
    
    # Calculer les dimensions finales de l'image
    final_width = effective_width * (max_col - min_col + 1) + total_h_overlap
    final_height = effective_height * (max_row - min_row + 1) + total_v_overlap
    
    # Créer une image vide pour le résultat
    # Utiliser des valeurs de fond blanc
    result = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255
    
    # Fonction pour afficher la progression
    def display_progress(text, count, total):
        percent = (count / total) * 100
        print(f"{text}: {count}/{total} ({percent:.1f}%)")
    
    # Parcourir toutes les positions et placer les images
    print(f"Assemblage de {len(image_files)} images...")
    print(f"Dimensions de la grille: {rows}x{cols}")
    print(f"Dimensions de l'image finale: {final_width}x{final_height} pixels")
    print(f"Taille de pixel: {pixel_size_mm} mm")
    print(f"Dimensions réelles: {final_width * pixel_size_mm:.1f} x {final_height * pixel_size_mm:.1f} mm")
    
    # Dictionnaire pour stocker les positions de la grille pour l'ajout ultérieur
    grid_positions = {}
    
    processed = 0
    for img_num, (row, col) in position_dict.items():
        if img_num not in image_files:
            continue
        
        # Charger l'image
        img = cv2.imread(image_files[img_num])
        if img is None:
            continue
        
        # Calculer la position dans l'image finale (ajustée pour les positions min)
        adj_row = row - min_row
        adj_col = col - min_col
        
        y_pos = adj_row * effective_height
        x_pos = adj_col * effective_width
        
        # Stocker la position pour l'ajout de la grille ultérieurement
        grid_positions[img_num] = (x_pos, y_pos, img_width, img_height)
        
        # Placer l'image dans le résultat
        result[y_pos:y_pos+img_height, x_pos:x_pos+img_width] = img
        
        processed += 1
        if processed % 20 == 0 or processed == len(image_files):
            display_progress("Assemblage", processed, len(image_files))
    
    # Ajouter la grille et les numéros d'image si demandé
    if show_grid or show_numbers:
        # Créer une copie de l'image pour dessiner la grille et les numéros
        overlay = result.copy()
        
        # Dessiner la grille
        if show_grid:
            # Créer un tableau pour représenter la grille
            grid_matrix = np.zeros((max_row + 1, max_col + 1), dtype=int)
            for img_num, (row, col) in position_dict.items():
                if img_num in image_files:
                    grid_matrix[row, col] = 1
            
            for img_num, (x_pos, y_pos, width, height) in grid_positions.items():
                # Déterminer si cette image a des voisins à droite et en bas
                row, col = position_dict[img_num]
                has_right_neighbor = col < max_col and grid_matrix[row, col+1] == 1
                has_bottom_neighbor = row < max_row and grid_matrix[row+1, col] == 1
                
                # Dessiner le rectangle avec les bords ajustés pour l'overlap
                x2 = x_pos + width if not has_right_neighbor else x_pos + effective_width
                y2 = y_pos + height if not has_bottom_neighbor else y_pos + effective_height
                
                cv2.rectangle(overlay, (x_pos, y_pos), (x2, y2), grid_color, grid_thickness)
                
        # Ajouter les numéros d'image
        if show_numbers:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            for img_num, (x_pos, y_pos, width, height) in grid_positions.items():
                # Extraire les 3 derniers chiffres du numéro d'image
                num_str = f"{img_num % 1000:03d}"  # Format: 001, 002, etc.
                
                # Calculer la taille du texte pour le positionner correctement
                text_size = cv2.getTextSize(num_str, font, font_scale, font_thickness)[0]
                
                # Position du texte (coin supérieur gauche avec un petit décalage)
                text_x = x_pos + 10
                text_y = y_pos + text_size[1] + 10
                
                # Dessiner un fond semi-transparent pour le texte
                rect_padding = 5
                cv2.rectangle(overlay, 
                             (text_x - rect_padding, text_y - text_size[1] - rect_padding),
                             (text_x + text_size[0] + rect_padding, text_y + rect_padding),
                             numbers_color, -1)  # -1 pour remplir le rectangle
                
                # Dessiner le texte en blanc
                cv2.putText(overlay, num_str, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        # Fusionner l'overlay avec l'image originale
        # Utiliser grid_alpha comme valeur alpha pour la grille et les numéros
        cv2.addWeighted(overlay, grid_alpha, result, 1 - grid_alpha, 0, result)
    
    # Créer le répertoire de sortie si nécessaire
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sauvegarder l'image finale
    print(f"Sauvegarde de l'image assemblée vers {output_path}...")
    cv2.imwrite(output_path, result)
    
    # Ajouter des métadonnées au nom du fichier pour se souvenir de la taille
    output_info = {
        "sample_name": sample_name,
        "original_count": len(image_files),
        "processed_count": processed,
        "grid_size": f"{rows}x{cols}",
        "image_dimensions": f"{final_width}x{final_height}",
        "physical_dimensions": f"{final_width * pixel_size_mm:.1f}x{final_height * pixel_size_mm:.1f}mm",
        "h_overlap": h_overlap,
        "v_overlap": v_overlap,
        "pixel_size_mm": pixel_size_mm,
        "source_directory": image_folder,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "show_grid": show_grid,
        "show_numbers": show_numbers
    }
    
    # Sauvegarder les informations dans un fichier JSON
    info_path = output_path.replace(".jpeg", ".json").replace(".jpg", ".json").replace(".png", ".json")
    with open(info_path, 'w') as f:
        json.dump(output_info, f, indent=2)
    
    print("Assemblage terminé avec succès!")
    return result


def run_overlap_detection(image_folder, output_file=None, num_samples=20, max_overlap=200, use_tiff=True):
    """
    Exécute la détection de chevauchement et sauvegarde les résultats.
    
    Args:
        image_folder (str): Dossier contenant les images
        output_file (str, optional): Fichier de sortie pour les résultats
        num_samples (int, optional): Nombre de paires d'images à échantillonner
        max_overlap (int, optional): Chevauchement maximal à vérifier
        use_tiff (bool, optional): Si True, cherche des fichiers TIFF, sinon cherche des JPEG
        
    Returns:
        tuple: (horizontal_overlap, vertical_overlap) estimations moyennes
    """
    # Déterminer les extensions en fonction du format d'image
    file_extensions = [".tif", ".tiff"] if use_tiff else [".jpeg", ".jpg"]
    
    # Trouver le premier fichier pour déterminer le motif du nom
    files = []
    for ext in file_extensions:
        files.extend([f for f in os.listdir(image_folder) if f.lower().endswith(ext)])
    
    if not files:
        raise ValueError(f"Aucun fichier {', '.join(file_extensions)} trouvé dans le dossier {image_folder}")
    
    # Extraire le nom de l'échantillon (ex: "T1_ALAC")
    first_file = files[0]
    sample_name = "_".join(first_file.split("_")[:-1])
    
    pattern = f"{sample_name}_*"
    
    print(f"Détection du chevauchement pour l'échantillon {sample_name}...")
    h_overlap, v_overlap = estimate_overlap_from_sample(
        image_folder, pattern, num_samples=num_samples, max_overlap=max_overlap
    )
    
    # Sauvegarder les résultats si un fichier de sortie est spécifié
    if output_file:
        results = {
            "sample_name": sample_name,
            "horizontal_overlap": h_overlap,
            "vertical_overlap": v_overlap,
            "num_samples": num_samples,
            "image_format": "TIFF" if use_tiff else "JPEG",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_directory": image_folder
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Résultats sauvegardés dans {output_file}")
    
    return h_overlap, v_overlap