"""
Configuration du pipeline pour le projet HYPH-SPOREA.

Ce module contient les paramètres de configuration pour le workflow du pipeline.
"""

import os
from pathlib import Path
import pyprojroot

# Répertoire du projet
PROJECT_ROOT = pyprojroot.here()

# Configurations des chemins
DEFAULT_CONFIG = {
    # Répertoire du projet
    'PROJECT_ROOT': PROJECT_ROOT,
    
    # Répertoires de données
    'data_dir': PROJECT_ROOT / 'data',
    'raw_data_dir': PROJECT_ROOT / 'data' / 'raw_data',
    'proc_data_dir': PROJECT_ROOT / 'data' / 'proc_data',
    'blank_model_dir': PROJECT_ROOT / 'data' / 'blank_models',
    
    # Répertoires de sortie
    'output_dir': PROJECT_ROOT / 'outputs',
    'models_dir': PROJECT_ROOT / 'outputs' / 'models',
    'analysis_dir': PROJECT_ROOT / 'outputs' / 'analysis',
    'visualization_dir': PROJECT_ROOT / 'outputs' / 'visualizations',
    
    # Fichiers de modèles par défaut
    'default_yolo_model': 'yolo_spores_model.pt',
    'default_unet_model': 'unet_spores_model.h5',
    'default_sam_model': 'sam2.1_b.pt',
    
    # Paramètres d'entrainement
    'yolo_epochs': 100,
    'yolo_batch_size': 16,
    'yolo_image_size': 640,
    
    'unet_epochs': 50,
    'unet_batch_size': 16,
    'unet_image_size': 256,
    
    # Paramètres de prétraitement
    'jpeg_quality': 90,
    'min_object_size': 30,
    'max_object_size': 5000,
    
    # Paramètres de détection
    'detection_threshold': 0.25,
    
    # Classification des spores
    'spore_classes': [
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
    ]
}


def load_config(config_file=None):
    """
    Charge la configuration à partir d'un fichier ou utilise la configuration par défaut.
    
    Args:
        config_file (str, optional): Chemin vers un fichier de configuration JSON.
    
    Returns:
        dict: Configuration chargée
    """
    config = DEFAULT_CONFIG.copy()
    
    # Si un fichier de configuration est spécifié, le charger et fusionner avec la config par défaut
    if config_file:
        import json
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                
            # Convertir les chemins en objets Path
            for key, value in user_config.items():
                if key.endswith('_dir') or key.endswith('_path'):
                    user_config[key] = Path(value)
                    
            # Fusionner avec la config par défaut
            config.update(user_config)
            
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {str(e)}")
    
    # Créer les répertoires nécessaires
    for key, path in config.items():
        if key.endswith('_dir') and isinstance(path, Path):
            os.makedirs(path, exist_ok=True)
    
    return config


def save_config(config, output_file):
    """
    Sauvegarde la configuration dans un fichier JSON.
    
    Args:
        config (dict): Configuration à sauvegarder
        output_file (str): Chemin du fichier de sortie
    
    Returns:
        str: Chemin du fichier sauvegardé
    """
    import json
    
    # Convertir les objets Path en chaînes de caractères
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    # Enregistrer au format JSON
    with open(output_file, 'w') as f:
        json.dump(serializable_config, f, indent=4)
    
    return output_file


def generate_default_config(output_file):
    """
    Génère un fichier de configuration par défaut.
    
    Args:
        output_file (str): Chemin du fichier de sortie
    
    Returns:
        str: Chemin du fichier généré
    """
    config = DEFAULT_CONFIG.copy()
    return save_config(config, output_file)