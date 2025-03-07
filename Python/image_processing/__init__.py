"""
Module de traitement d'images pour le projet HYPH-SPOREA.

Ce package contient les fonctionnalités de traitement d'images:
- Conversion des images TIFF en JPEG
- Prétraitement pour améliorer la détection des spores
- Visualisation des résultats
"""

from .conversion import (
    extract_tiff_metadata,
    convert_tiff_to_jpeg,
    process_directory,
    verify_conversion
)

from .preprocessing import (
    enhanced_preprocess_image,
    batch_preprocess_directory,
    create_binary_masks,
    create_visualization
)

__all__ = [
    'extract_tiff_metadata',
    'convert_tiff_to_jpeg',
    'process_directory',
    'verify_conversion',
    'enhanced_preprocess_image',
    'batch_preprocess_directory',
    'create_binary_masks',
    'create_visualization'
]
