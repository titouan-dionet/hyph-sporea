"""
Module de génération de rapports pour le projet HYPH-SPOREA.

Ce module contient des fonctions pour générer des rapports au format Quarto
à partir des résultats d'analyse.
"""

from .reporting import generate_quarto_report

__all__ = [
    'generate_quarto_report'
]