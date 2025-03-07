"""
Module d'analyse pour le projet HYPH-SPOREA.

Ce package contient des fonctions pour analyser les résultats
de détection et de segmentation des spores.
"""

from .statistics import (
    analyze_detections,
    save_analysis_results,
    compare_samples,
    perform_statistical_tests,
    load_and_merge_results,
    analyze_spatial_distribution
)

__all__ = [
    'analyze_detections',
    'save_analysis_results',
    'compare_samples',
    'perform_statistical_tests',
    'load_and_merge_results',
    'analyze_spatial_distribution'
]
