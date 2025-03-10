"""
Fonctions utilitaires générales pour le projet HYPH-SPOREA.

Ce module contient des fonctions d'utilité générale qui peuvent être utilisées
dans différentes parties du projet.
"""

import os
import json
from pathlib import Path


def ensure_dir(directory):
    """
    Assure qu'un répertoire existe, le crée si nécessaire.
    
    Args:
        directory (str or Path): Chemin du répertoire à vérifier/créer
    
    Returns:
        Path: Chemin du répertoire
    
    Example:
        >>> data_dir = ensure_dir("data/raw_data")
        >>> print(data_dir)
        PosixPath('data/raw_data')
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_sample_info(sample_path):
    """
    Extrait les informations d'un échantillon à partir de son nom.
    
    Format attendu: "T1_ALAC_C6_1" où:
    - "T1_ALAC" est la souche
    - "C6" est la condition
    - "1" est le numéro de réplicat
    
    Args:
        sample_path (str or Path): Chemin vers l'échantillon ou nom de l'échantillon
    
    Returns:
        dict: Dictionnaire contenant les informations de l'échantillon
    
    Example:
        >>> info = get_sample_info("T1_ALAC_C6_1")
        >>> print(info)
        {'sample': 'T1_ALAC_C6_1', 'strain': 'T1_ALAC', 'condition': 'C6', 'replicate': '1'}
    """
    # Extraction du nom d'échantillon si c'est un chemin
    if isinstance(sample_path, (str, Path)) and os.path.exists(sample_path):
        sample_name = os.path.basename(sample_path)
    else:
        sample_name = str(sample_path)
    
    # Suppression du préfixe "processed_" s'il existe
    sample_name = sample_name.replace('processed_', '')
    
    # Décomposition du nom de l'échantillon
    parts = sample_name.split('_')
    
    # Initialisation du dictionnaire d'informations
    info = {'sample': sample_name}
    
    # Extraction de la souche, condition et réplicat
    if len(parts) >= 4:
        info['strain'] = parts[0] + '_' + parts[1]
        info['condition'] = parts[2]
        info['replicate'] = parts[3]
    elif len(parts) == 3:
        # Cas où la souche n'a qu'un seul segment
        info['strain'] = parts[0]
        info['condition'] = parts[1]
        info['replicate'] = parts[2]
    elif len(parts) == 2:
        # Cas minimal
        info['strain'] = parts[0]
        info['condition'] = parts[1]
        info['replicate'] = '1'  # Réplicat par défaut
    else:
        # Cas inconnu
        info['strain'] = sample_name
        info['condition'] = 'unknown'
        info['replicate'] = '1'
    
    return info

def get_sample_info_from_path(path):
    """
    Analyse le chemin d'un fichier ou dossier pour extraire les informations d'échantillon.
    
    Args:
        path (str or Path): Chemin du fichier ou dossier
    
    Returns:
        dict: Informations de l'échantillon
    
    Example:
        >>> info = get_sample_info_from_path("T2_LUCU_MA01_C6/T2_LUCU_MA01_C6_000156.tif")
        >>> print(info)
        {'sample': 'T2_LUCU_MA01_C6', 'strain': 'T2_LUCU', 'condition': 'MA01_C6', 'replicate': '1'}
    """
    path = Path(path)
    
    # Déterminer le nom de l'échantillon
    if path.is_file():
        # Si c'est un fichier, utiliser le nom du dossier parent
        sample_dir = path.parent.name
        # Si le nom du fichier contient le nom du dossier, on l'extrait
        filename_parts = path.stem.split('_')
        if len(filename_parts) > 3:  # Format attendu: T2_LUCU_MA01_C6_000156
            sample_name = '_'.join(filename_parts[:-1])  # Ignorer le dernier segment (numéro d'image)
        else:
            sample_name = sample_dir
    else:
        # Si c'est un dossier, utiliser son nom
        sample_name = path.name
    
    # Diviser le nom de l'échantillon en parties
    parts = sample_name.split('_')
    
    # Initialisation des informations
    info = {'sample': sample_name}
    
    # Décider comment extraire strain et condition en fonction du nombre de parties
    if len(parts) >= 4:  # ex: T2_LUCU_MA01_C6
        # Les deux premières parties forment la souche
        info['strain'] = parts[0] + '_' + parts[1]
        # Les parties restantes forment la condition
        info['condition'] = '_'.join(parts[2:])
    elif len(parts) == 3:  # ex: T2_LUCU_C6
        info['strain'] = parts[0] + '_' + parts[1]
        info['condition'] = parts[2]
    elif len(parts) == 2:  # ex: T2_LUCU
        info['strain'] = sample_name
        info['condition'] = 'unknown'
    else:
        info['strain'] = sample_name
        info['condition'] = 'unknown'
    
    # Vérifier si la dernière partie est un nombre (réplicat)
    last_part = parts[-1]
    if last_part.isdigit():
        info['replicate'] = last_part
        # Ajuster strain et condition
        if len(parts) > 2:
            info['strain'] = '_'.join(parts[:-2])
            info['condition'] = parts[-2]
    else:
        # Si pas de réplicat explicite, utiliser 1 par défaut
        info['replicate'] = '1'
    
    return info

def load_metadata(image_path):
    """
    Charge les métadonnées associées à une image.
    
    Cherche un fichier JSON de même nom que l'image et charge son contenu.
    
    Args:
        image_path (str or Path): Chemin vers l'image
    
    Returns:
        dict: Métadonnées chargées ou dictionnaire vide si aucun fichier trouvé
    
    Example:
        >>> metadata = load_metadata("data/raw_data/T1_ALAC_C6_1/T1_ALAC_C6_1_000001.jpeg")
        >>> print(metadata.get('Strain'))
        T1_ALAC
    """
    image_path = Path(image_path)
    metadata_path = image_path.with_suffix('.json')
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Erreur lors de la lecture du fichier de métadonnées {metadata_path}")
            return {}
    else:
        return {}


def count_files_by_extension(directory, extension='.jpeg'):
    """
    Compte les fichiers d'une extension donnée dans un répertoire.
    
    Args:
        directory (str or Path): Répertoire à explorer
        extension (str, optional): Extension de fichier à rechercher. Par défaut '.jpeg'
    
    Returns:
        int: Nombre de fichiers trouvés
    
    Example:
        >>> n_images = count_files_by_extension("data/raw_data/T1_ALAC_C6_1", ".jpeg")
        >>> print(f"Échantillon contenant {n_images} images")
        Échantillon contenant 356 images
    """
    directory = Path(directory)
    count = 0
    
    for file in directory.glob(f"*{extension}"):
        if file.is_file():
            count += 1
    
    return count


def save_config(config, filepath):
    """
    Sauvegarde un dictionnaire de configuration dans un fichier JSON.
    
    Args:
        config (dict): Configuration à sauvegarder
        filepath (str or Path): Chemin du fichier de sortie
    
    Returns:
        Path: Chemin du fichier sauvegardé
    
    Example:
        >>> config = {"model_type": "yolo", "epochs": 100}
        >>> save_config(config, "outputs/configs/train_config.json")
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filepath


def load_config(filepath):
    """
    Charge un fichier de configuration JSON.
    
    Args:
        filepath (str or Path): Chemin du fichier de configuration
    
    Returns:
        dict: Configuration chargée
    
    Example:
        >>> config = load_config("outputs/configs/train_config.json")
        >>> print(config["model_type"])
        yolo
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier de configuration {filepath} introuvable")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def get_latest_model_version(models_dir, model_type='yolo'):
    """
    Récupère la dernière version du modèle dans le répertoire spécifié.
    
    Args:
        models_dir (str or Path): Répertoire contenant les modèles
        model_type (str): Type de modèle ('yolo', 'unet', 'sam')
    
    Returns:
        Path: Chemin du modèle le plus récent
    """
    models_dir = Path(models_dir)
    pattern = f"{model_type}_*"
    
    # Chercher tous les dossiers de modèles
    model_dirs = list(models_dir.glob(pattern))
    
    if not model_dirs:
        # Vérifier s'il existe un modèle standard
        if model_type == 'yolo':
            standard_model = models_dir / "yolo_spores_model.pt"
            if standard_model.exists():
                return standard_model
        elif model_type == 'unet':
            standard_model = models_dir / "unet_spores_model.h5"
            if standard_model.exists():
                return standard_model
        elif model_type == 'sam':
            standard_model = models_dir / "sam_model.pt"
            if standard_model.exists():
                return standard_model
        
        return None
    
    # Trier par date de modification (le plus récent en premier)
    model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Chercher le fichier de modèle dans le dossier le plus récent
    if model_type == 'yolo':
        model_file = model_dirs[0] / 'yolo_spores_model.pt'
    elif model_type == 'unet':
        model_file = model_dirs[0] / 'unet_spores_model.h5'
    elif model_type == 'sam':
        model_file = model_dirs[0] / 'sam_model.pt'
    
    if model_file.exists():
        return model_file
    
    return None

def measure_performance(func):
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.
    
    Args:
        func (callable): Fonction à mesurer
    
    Returns:
        callable: Fonction décorée
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Démarrage de {func.__name__}...")
        
        try:
            result = func(*args, **kwargs)
            
            elapsed_time = time.time() - start_time
            print(f"{func.__name__} terminé en {elapsed_time:.2f} secondes")
            
            # Si le premier argument est un objet avec un attribut output_dir,
            # enregistrer les performances
            if args and hasattr(args[0], 'output_dir'):
                obj = args[0]
                perf_dir = Path(obj.output_dir) / "performance"
                perf_dir.mkdir(exist_ok=True, parents=True)
                
                perf_file = perf_dir / f"{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # Préparer les données de performance
                perf_data = {
                    'function': func.__name__,
                    'start_time': datetime.fromtimestamp(start_time).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'elapsed_time': elapsed_time,
                    'args': str([str(arg) for arg in args[1:]]),
                    'kwargs': str(kwargs)
                }
                
                # Sauvegarder les performances
                with open(perf_file, 'w') as f:
                    json.dump(perf_data, f, indent=2)
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"Erreur dans {func.__name__} après {elapsed_time:.2f} secondes: {str(e)}")
            raise
    
    return wrapper