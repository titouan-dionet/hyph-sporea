"""
Package principal du projet HYPH-SPOREA.

HYPH-SPOREA (Hyphomycetes Spores Recognition with Enhanced Algorithms) est un projet
pour l'analyse des spores d'hyphomycètes aquatiques.

Ce package contient tous les modules et fonctionnalités du projet.
"""

# Import des sous-packages principaux
from . import core
from . import image_processing
from . import models
from . import gui
from . import analysis

# Import des classes et fonctions principales
from .core import HyphSporeaProcessor

__version__ = '0.1.0'
__author__ = 'HYPH-SPOREA Team'

__all__ = [
    'core',
    'image_processing',
    'models',
    'gui',
    'analysis',
    'HyphSporeaProcessor'
]
