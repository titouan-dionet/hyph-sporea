# HYPH-SPOREA

**Hyphomycetes Spores Recognition with Enhanced Algorithms**

HYPH-SPOREA est un projet dédié à l'analyse d'images de spores d'hyphomycètes aquatiques. Ce système permet d'identifier et de classifier automatiquement différentes espèces de spores à partir d'images microscopiques.

## Structure du projet

```
hyph-sporea/
├── analyses/              # Codes d'analyse et pipeline
│   └── pipeline/          # Scripts du pipeline de traitement
├── data/                  # Données
│   ├── blank_models/      # Modèles de base (non modifiés)
│   ├── raw_data/          # Données brutes
│   └── proc_data/         # Données prétraitées
├── documents/             # Documentation du projet
├── outputs/               # Sorties des analyses
│   ├── models/            # Modèles générés
│   ├── analysis/          # Résultats d'analyse
│   └── visualizations/    # Visualisations
├── Python/                # Code Python
│   ├── core/              # Classes et fonctions principales
│   ├── image_processing/  # Traitement d'images
│   ├── models/            # Modèles de détection/segmentation
│   ├── gui/               # Interfaces graphiques
│   └── analysis/          # Analyse statistique
├── R/                     # Code R
├── syntheses/             # Rapports et synthèses
├── venv/                  # Environnement virtuel Python
├── renv/                  # Environnement virtuel R
└── README.md              # Ce fichier
```

## Fonctionnalités

HYPH-SPOREA permet de:

1. **Convertir les images TIFF en JPEG** tout en préservant les métadonnées importantes
2. **Prétraiter les images** pour améliorer la détection des spores
3. **Annoter les spores** via une interface graphique dédiée
4. **Entraîner des modèles** de détection et segmentation:
   - YOLOv8 pour la détection et classification
   - U-Net pour la segmentation
   - SAM (Segment Anything Model) pour la segmentation automatique
5. **Analyser les échantillons** pour obtenir des statistiques sur les spores
6. **Comparer les résultats** entre différents échantillons

## Installation

### Prérequis

- Python 3.8+
- R 4.0+ (optionnel, pour les analyses statistiques avancées)
- Bibliothèques Python requises (voir `requirements.txt`)

### Installation des dépendances

```bash
# Création et activation de l'environnement virtuel Python
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installation des dépendances
pip install -r requirements.txt
```

## Utilisation

### Via l'interface en ligne de commande

```bash
# Conversion des images TIFF en JPEG
python analyses/pipeline/main.py --project_dir /chemin/vers/projet convert \
    --input_dir /chemin/vers/tiff \
    --output_dir /chemin/vers/jpeg

# Prétraitement des images
python analyses/pipeline/main.py --project_dir /chemin/vers/projet preprocess \
    --input_dir /chemin/vers/jpeg \
    --output_dir /chemin/vers/preprocessed

# Lancement de l'outil d'annotation
python analyses/pipeline/main.py --project_dir /chemin/vers/projet annotate

# Entraînement d'un modèle YOLO
python analyses/pipeline/main.py --project_dir /chemin/vers/projet train \
    --model_type yolo \
    --data_yaml /chemin/vers/data.yaml

# Entraînement d'un modèle U-Net
python analyses/pipeline/main.py --project_dir /chemin/vers/projet train \
    --model_type unet \
    --image_dir /chemin/vers/images \
    --mask_dir /chemin/vers/masques

# Traitement d'échantillons
python analyses/pipeline/main.py --project_dir /chemin/vers/projet process \
    --sample_dirs /chemin/vers/echantillon1 /chemin/vers/echantillon2 \
    --model_path /chemin/vers/modele.pt

# Comparaison d'échantillons
python analyses/pipeline/main.py --project_dir /chemin/vers/projet compare \
    --result_dirs /chemin/vers/resultats1 /chemin/vers/resultats2 \
    --output_dir /chemin/vers/comparaison

# Pipeline complet
python analyses/pipeline/main.py --project_dir /chemin/vers/projet pipeline \
    --tiff_dir /chemin/vers/tiff \
    --model_type yolo \
    --sample_dirs /chemin/vers/echantillon1 /chemin/vers/echantillon2
```

### Via le workflow

```python
# Exemple d'utilisation du workflow
from analyses.pipeline.workflow import Workflow, target

# Créer le workflow
workflow = Workflow("mon_workflow")

# Ajouter des cibles
@target(workflow, "ma_cible", file_outputs=["output.csv"])
def ma_fonction():
    # Traitement
    return resultat

# Visualiser le workflow
workflow.visualize("workflow_graph.png")

# Exécuter le workflow
results = workflow.execute()
```

## Exemple de workflow complet

Un exemple complet de workflow est disponible dans le fichier `analyses/pipeline/example_workflow.py`.

## Contribuer

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
