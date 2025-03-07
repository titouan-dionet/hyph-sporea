# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 19:41:44 2025

@author: p05421
"""

import os
import cv2
import numpy as np
from ultralytics import SAM
import pyprojroot
import torch

#%% Racine du projet 
PROJECT_ROOT = pyprojroot.here()

# 📂 Dossier des images et annotations YOLO
images_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/images"
labels_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/labels"
sam_model_path = PROJECT_ROOT / "data/blank_models/sam2.1_l.pt"

# Charger le modèle SAM
model = SAM(sam_model_path)

# 🔄 Boucle sur toutes les images du dataset
for img_file in os.listdir(images_dir):
    if img_file.endswith(".jpeg"):
        # Chemins de l'image et du label YOLO
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, img_file.replace(".jpeg", ".txt"))

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
            results = model(image, points=points, project = PROJECT_ROOT / "outputs/SAM_test", save=True)
            print(f"SAM exécuté sur {img_file} avec {len(points)} points d'aide.")
        else:
            print(f"Aucune annotation pour {img_file}, SAM tourne en mode auto.")
        

#%% Racine du projet 
PROJECT_ROOT = pyprojroot.here()

# 📂 Dossier des images et annotations YOLO
images_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/images"
labels_dir = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/train/labels"
sam_model_path = PROJECT_ROOT / "data/blank_models/sam2.1_l.pt"
sam_model_path = PROJECT_ROOT / "outputs/SAM_tune" / "sam_finetuned_spores.pt"

# Charger le modèle SAM
# model = SAM(sam_model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(sam_model_path, map_location=device)

# Charger une image
img_path = "C:/Users/p05421/Documents/CLAQ/T5_CLAQ_C6_2/T5_CLAQ_C6_2_000079.jpeg"
image = cv2.imread(img_path)

# Détection des objets avec SAM
results = model(img_path, project = PROJECT_ROOT / "outputs/SAM_test", save = True)
results = model(img_path, project = PROJECT_ROOT / "outputs/SAM_test", save = True, conf = 0.2)
results = model(img_path, project = PROJECT_ROOT / "outputs/SAM_test", save = True, mode = "everything")

# Récupérer les bounding boxes des objets détectés
bboxes = []
for mask in results[0].masks.xy:
    if not len(mask) == 0:
        x_min = int(np.min(mask[:, 0]))
        y_min = int(np.min(mask[:, 1]))
        x_max = int(np.max(mask[:, 0]))
        y_max = int(np.max(mask[:, 1]))
        bboxes.append((x_min, y_min, x_max, y_max))

# Afficher les bounding boxes sur l'image
for bbox in bboxes:
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

cv2.imshow("Detections SAM", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

