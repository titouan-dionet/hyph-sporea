# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 13:27:00 2025

@author: p05421
"""

import os
import cv2
import numpy as np
import torch
import pyprojroot
from segment_anything import sam_model_registry, SamPredictor

#%% 📌 Définition des chemins
PROJECT_ROOT = pyprojroot.here()

# 📌 Charger SAM pré-entraîné
sam_checkpoint = PROJECT_ROOT / "data/blank_models" / "sam_vit_h_4b8939.pth"  # Modèle de base
model_type = "vit_h"  # Architecture de SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

# 🔹 Charger les poids entraînés
sam_model_path = PROJECT_ROOT / "outputs/SAM_tune/sam_finetuned_spores.pth"
sam.load_state_dict(torch.load(sam_model_path, map_location=device))
sam.eval()  # Mettre en mode évaluation

# 🔹 Créer un prédicteur SAM
sam_predictor = SamPredictor(sam)

# 🔹 Charger une image
img_path = "C:/Users/p05421/Documents/CLAQ/T5_CLAQ_C6_2/T5_CLAQ_C6_2_000079.jpeg"
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # SAM attend une image RGB

# 🔹 Passer l'image à SAM
sam_predictor.set_image(image_rgb)

# 🔹 Définir un point d'interaction (exemple : centre de l'image)
input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])  # (x, y)
input_label = np.array([1])  # 1 = objet

# 🔹 Obtenir la segmentation
masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)

# 🔹 Récupérer les bounding boxes des objets détectés
bboxes = []
for mask in masks:
    y_indices, x_indices = np.where(mask)  # Trouver les pixels non nuls du masque
    if len(y_indices) > 0 and len(x_indices) > 0:
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        bboxes.append((x_min, y_min, x_max, y_max))

# 🔹 Dessiner les bounding boxes sur l'image
for bbox in bboxes:
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

cv2.imshow("Detections SAM", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
