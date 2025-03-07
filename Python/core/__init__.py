"""
Core du projet HYPH-SPOREA.

Ce package contient les fonctionnalités centrales du projet:
- Classe principale pour le traitement des spores d'hyphomycètes
- Fonctions utilitaires générales
"""

from .processor import HyphSporeaProcessor
from .utils import (
    ensure_dir, 
    get_sample_info, 
    load_metadata, 
    count_files_by_extension,
    save_config,
    load_config
)

__all__ = [
    'HyphSporeaProcessor',
    'ensure_dir',
    'get_sample_info',
    'load_metadata',
    'count_files_by_extension',
    'save_config',
    'load_config'
]
