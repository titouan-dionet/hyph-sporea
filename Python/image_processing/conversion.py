"""
Module de conversion d'images pour le projet HYPH-SPOREA.

Ce module contient les fonctions pour convertir les images TIFF en JPEG
tout en préservant les métadonnées importantes.
"""

import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import piexif
import json
from multiprocessing import Pool, cpu_count
import time
from pathlib import Path
from ..core.utils import get_sample_info_from_path


def extract_tiff_metadata(tiff_path):
    """
    Extrait les métadonnées d'un fichier TIFF.
    
    Récupère les métadonnées EXIF et les informations de base de l'image.
    Extrait également les informations du nom de fichier selon la convention:
    "T1_ALAC_C6_1_000156.tiff" où:
    - "T1_ALAC" est la souche
    - "C6" est la condition
    - "1" est le numéro de réplicat
    - "000156" est le numéro d'image
    
    Args:
        tiff_path (str): Chemin du fichier TIFF
    
    Returns:
        dict: Dictionnaire contenant les métadonnées extraites
    
    Example:
        >>> metadata = extract_tiff_metadata("data/raw_data/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.tiff")
        >>> print(metadata['Strain'])
        T1_ALAC
    """
    try:
        # Ouverture de l'image avec PIL
        img = Image.open(tiff_path)
        
        # Extraction des métadonnées
        meta_dict = {}
        
        # Ajout des propriétés de base de l'image
        meta_dict['Width'] = img.width
        meta_dict['Height'] = img.height
        meta_dict['Format'] = img.format
        meta_dict['Mode'] = img.mode
        
        # Tentative de récupération des métadonnées EXIF, sans erreur si absentes
        try:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    meta_dict[tag] = str(value)
        except (AttributeError, TypeError, KeyError):
            # Si pas de métadonnées EXIF, on continue sans erreur
            pass
        
        # Ajout d'informations supplémentaires extraites du nom de fichier
        filename = os.path.basename(tiff_path)
        name_parts = filename.split('_')
        
        # Analyse du nom de fichier pour extraire les informations
        filepath = Path(tiff_path)
        sample_info = get_sample_info_from_path(filepath)
        
        # Ajouter les informations d'échantillon aux métadonnées
        meta_dict.update(sample_info)
        
        return meta_dict
    except Exception as e:
        print(f"Erreur lors de l'extraction des métadonnées de {tiff_path}: {str(e)}")
        return {}


def convert_tiff_to_jpeg(tiff_path, output_dir, quality=85):
    """
    Convertit un fichier TIFF en JPEG en préservant les métadonnées.
    
    La fonction convertit l'image, sauvegarde les métadonnées dans un
    fichier JSON associé et tente également de les injecter dans les
    métadonnées EXIF du JPEG.
    
    Args:
        tiff_path (str): Chemin du fichier TIFF à convertir
        output_dir (str): Répertoire de sortie pour les fichiers JPEG
        quality (int, optional): Qualité de compression JPEG (1-100). Par défaut 85.
    
    Returns:
        bool: True si la conversion a réussi, False sinon
    
    Example:
        >>> success = convert_tiff_to_jpeg(
        ...     "data/raw_data/T1_ALAC_C6_1/T1_ALAC_C6_1_000156.tiff",
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1"
        ... )
    """
    try:
        # Extraction des métadonnées
        metadata = extract_tiff_metadata(tiff_path)
        
        # Lecture de l'image avec OpenCV
        img = cv2.imread(tiff_path)
        
        # Vérification que l'image a été correctement chargée
        if img is None:
            print(f"Impossible de lire l'image {tiff_path}")
            return False
        
        # Création du nom de fichier JPEG
        filename = os.path.basename(tiff_path)
        base_name = os.path.splitext(filename)[0]
        jpeg_filename = base_name + '.jpeg'
        jpeg_path = os.path.join(output_dir, jpeg_filename)
        
        # Sauvegarde de l'image au format JPEG
        cv2.imwrite(jpeg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        
        # Sauvegarde des métadonnées dans un fichier JSON associé
        meta_filename = base_name + '.json'
        meta_path = os.path.join(output_dir, meta_filename)
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Tentative d'ajout de métadonnées EXIF au JPEG
        try:
            # Création d'un dictionnaire EXIF vide
            zeroth_ifd = {}
            exif_ifd = {}
            gps_ifd = {}
            
            # Ajout de métadonnées de base
            if 'Artist' in metadata:
                zeroth_ifd[piexif.ImageIFD.Artist] = metadata['Artist']
            if 'Copyright' in metadata:
                zeroth_ifd[piexif.ImageIFD.Copyright] = metadata['Copyright']
            
            # Ajout de métadonnées personnalisées dans le champ UserComment
            user_comment = json.dumps(metadata).encode('utf-8')
            exif_ifd[piexif.ExifIFD.UserComment] = user_comment
            
            # Assemblage du dictionnaire EXIF
            exif_dict = {
                "0th": zeroth_ifd,
                "Exif": exif_ifd,
                "GPS": gps_ifd,
                "1st": {},
                "thumbnail": None
            }
            
            # Conversion en bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Ajout au fichier JPEG
            piexif.insert(exif_bytes, jpeg_path)
        except Exception as e:
            print(f"Erreur lors de l'ajout des métadonnées EXIF: {str(e)}")
        
        return True
    except Exception as e:
        print(f"Erreur lors de la conversion de {tiff_path}: {str(e)}")
        return False


def process_directory(input_dir, output_dir, recursive=True, parallel=True):
    """
    Traite un répertoire de fichiers TIFF et les convertit en JPEG.
    
    Args:
        input_dir (str): Répertoire contenant les images TIFF
        output_dir (str): Répertoire de sortie pour les images JPEG
        recursive (bool, optional): Si True, parcourt les sous-répertoires. Par défaut True.
        parallel (bool, optional): Si True, utilise le traitement multiprocessus. Par défaut True.
    
    Returns:
        tuple: (nombre de fichiers traités, temps d'exécution)
    
    Example:
        >>> n_files, duration = process_directory(
        ...     "data/raw_data",
        ...     "data/proc_data/jpeg_images"
        ... )
        >>> print(f"Conversion de {n_files} fichiers en {duration:.2f} secondes")
    """
    # Création du répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Liste des fichiers TIFF à traiter
    tiff_files = []
    
    # Parcours du répertoire
    if recursive:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    tiff_path = os.path.join(root, file)
                    
                    # Création de la structure de répertoires correspondante dans output_dir
                    rel_path = os.path.relpath(root, input_dir)
                    target_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    tiff_files.append((tiff_path, target_dir))
    else:
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.tif', '.tiff')):
                tiff_path = os.path.join(input_dir, file)
                tiff_files.append((tiff_path, output_dir))
    
    # Affichage du nombre de fichiers à traiter
    print(f"Conversion de {len(tiff_files)} fichiers TIFF en JPEG...")
    
    # Traitement parallèle ou séquentiel
    start_time = time.time()
    
    if parallel and len(tiff_files) > 10:
        # Utilisation de multiprocessing pour accélérer le traitement
        with Pool(processes=cpu_count()) as pool:
            results = []
            for tiff_path, target_dir in tiff_files:
                results.append(pool.apply_async(convert_tiff_to_jpeg, (tiff_path, target_dir)))
            
            # Attente des résultats
            for i, result in enumerate(results):
                result.get()
                if (i + 1) % 10 == 0 or i == len(results) - 1:
                    print(f"Progression: {i + 1}/{len(tiff_files)} fichiers traités")
    else:
        # Traitement séquentiel
        for i, (tiff_path, target_dir) in enumerate(tiff_files):
            convert_tiff_to_jpeg(tiff_path, target_dir)
            if (i + 1) % 10 == 0 or i == len(tiff_files) - 1:
                print(f"Progression: {i + 1}/{len(tiff_files)} fichiers traités")
    
    elapsed_time = time.time() - start_time
    print(f"Conversion terminée en {elapsed_time:.2f} secondes")
    
    return len(tiff_files), elapsed_time


def verify_conversion(tiff_dir, jpeg_dir):
    """
    Vérifie que tous les fichiers TIFF ont été correctement convertis en JPEG.
    
    Args:
        tiff_dir (str): Répertoire contenant les images TIFF originales
        jpeg_dir (str): Répertoire contenant les images JPEG converties
    
    Returns:
        tuple: (nombre de fichiers vérifiés, liste des fichiers manquants)
    
    Example:
        >>> n_verified, missing = verify_conversion(
        ...     "data/raw_data/T1_ALAC_C6_1",
        ...     "data/proc_data/jpeg_images/T1_ALAC_C6_1"
        ... )
        >>> if missing:
        ...     print(f"Fichiers manquants: {missing}")
        ... else:
        ...     print(f"Tous les {n_verified} fichiers ont été convertis avec succès")
    """
    # Liste de tous les fichiers TIFF
    tiff_files = []
    for root, _, files in os.walk(tiff_dir):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                rel_path = os.path.relpath(root, tiff_dir)
                tiff_files.append(os.path.join(rel_path, file))
    
    # Liste des fichiers manquants
    missing_files = []
    
    for tiff_file in tiff_files:
        # Chemin correspondant du fichier JPEG
        jpeg_file = os.path.splitext(tiff_file)[0] + '.jpeg'
        jpeg_path = os.path.join(jpeg_dir, jpeg_file)
        
        # Vérification de l'existence du fichier JPEG
        if not os.path.exists(jpeg_path):
            missing_files.append(tiff_file)
    
    return len(tiff_files), missing_files
