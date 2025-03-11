"""
Module de workflow pour HYPH-SPOREA.

Ce module fournit des fonctionnalités pour définir et exécuter des workflows.
"""

from .workflow import Workflow, target, create_sam_detection_workflow, create_custom_workflow, get_latest_model_version

__all__ = [
    'Workflow',
    'target',
    'create_sam_detection_workflow',
    'create_custom_workflow',
    'get_latest_model_version'
]