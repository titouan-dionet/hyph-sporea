# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:38:03 2025

@author: p05421
"""

import os
import cv2
import numpy as np
import pyprojroot


PROJECT_ROOT = pyprojroot.here()

# Dossiers d'entrée et sortie
img_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/images"
label_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/labels"
mask_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/masks"  # Où sauvegarder les masques
os.makedirs(mask_dir, exist_ok=True)

# Taille des images
#IMG_WIDTH, IMG_HEIGHT = 1200, 1600  # Adapter à tes images

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    img_path = os.path.join(img_dir, label_file.replace(".txt", ".jpeg"))
    label_path = os.path.join(label_dir, label_file)

    # Charger l'image pour connaître ses dimensions
    img = cv2.imread(img_path)
    H, W, _ = img.shape

    # Créer un masque vide
    mask = np.zeros((H, W), dtype=np.uint8)

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)

            # Convertir YOLO en coordonnées absolues
            x1 = int((x_center - width / 2) * W)
            y1 = int((y_center - height / 2) * H)
            x2 = int((x_center + width / 2) * W)
            y2 = int((y_center + height / 2) * H)

            # Dessiner le masque
            mask[y1:y2, x1:x2] = 255  # 255 = objet

    # Sauvegarder le masque
    cv2.imwrite(os.path.join(mask_dir, label_file.replace(".txt", ".png")), mask)
