# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:37:26 2025

@author: p05421
"""

import os
import cv2
import numpy as np
from ultralytics import SAM

import pyprojroot

PROJECT_ROOT = pyprojroot.here()

# 📂 Dossier des images et annotations YOLO
images_dir = "C:/Users/p05421/Documents/CLAQ/images"
labels_dir = "C:/Users/p05421/Documents/CLAQ/labels"
sam_model_path = "data/blank_models/sam2.1_l.pt"

# Charger le modèle SAM
model = SAM(sam_model_path)

# 🔄 Boucle sur toutes les images du dataset
for img_file in os.listdir(images_dir):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        # Chemins de l'image et du label YOLO
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Lire l'image
        image = cv2.imread(img_path)
        height, width, _ = image.shape

        # 📌 Lire le fichier d'annotation YOLO
        points = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = line.strip().split()
                    x_center, y_center = float(values[1]), float(values[2])  # YOLO format (normalisé)

                    # Convertir en pixels
                    x_pixel = int(x_center * width)
                    y_pixel = int(y_center * height)

                    points.append([x_pixel, y_pixel])  # Ajouter le point

        # 🟢 Exécuter SAM avec ces points
        if points:
            results = model(image, points=points, project="outputs/SAM_test", save=True)
            print(f"SAM exécuté sur {img_file} avec {len(points)} points d'aide.")
        else:
            print(f"Aucune annotation pour {img_file}, SAM tourne en mode auto.")
