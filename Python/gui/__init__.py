"""
Module d'interfaces graphiques pour le projet HYPH-SPOREA.

Ce package contient les interfaces graphiques pour l'interaction avec l'utilisateur:
- Outil d'annotation de spores
- Interface principale
"""

from .annotation_tool import SporeAnnotationTool, run_annotation_tool
from .main_gui import HyphSporeaGUI, run_gui  # Ajouter l'import de la nouvelle interface

__all__ = [
    'SporeAnnotationTool',
    'run_annotation_tool',
    'HyphSporeaGUI',  # Ajouter la classe de l'interface principale
    'run_gui'  # Ajouter la fonction pour lancer l'interface
]

def __getattr__(name):
    if name == 'HyphSporeaGUI':
        from .main_gui import HyphSporeaGUI
        return HyphSporeaGUI
    elif name == 'run_gui':
        from .main_gui import run_gui
        return run_gui
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")