"""
Module des modèles pour le projet HYPH-SPOREA.

Ce package contient les implémentations de différents modèles
pour la détection et la segmentation des spores:
- YOLO: pour la détection et classification des spores
- U-Net: pour la segmentation des spores
- SAM: pour la segmentation automatique des objets
"""

from .yolo_model import (
    load_yolo_model,
    train_yolo_model,
    validate_yolo_model,
    predict_with_yolo,
    batch_predict_with_yolo,
    visualize_yolo_training_results,
    parse_yolo_predictions
)

from .unet_model import (
    create_unet_model,
    train_unet,
    predict_with_unet,
    batch_predict_with_unet,
    evaluate_unet_model,
    plot_training_history
)

from .sam_model import (
    load_sam_model,
    detect_with_sam,
    batch_detect_with_sam,
    parse_sam_results,
    combine_sam_detections
)

__all__ = [
    # YOLO
    'load_yolo_model',
    'train_yolo_model',
    'validate_yolo_model',
    'predict_with_yolo',
    'batch_predict_with_yolo',
    'visualize_yolo_training_results',
    'parse_yolo_predictions',
    
    # U-Net
    'create_unet_model',
    'train_unet',
    'predict_with_unet',
    'batch_predict_with_unet',
    'evaluate_unet_model',
    'plot_training_history',
    
    # SAM
    'load_sam_model',
    'detect_with_sam',
    'batch_detect_with_sam',
    'parse_sam_results',
    'combine_sam_detections'
]
