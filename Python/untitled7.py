# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:12:30 2025

@author: Titouan Dionet
"""

#%% Packages

import torch
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import pyprojroot

#%% Racine du projet
PROJECT_ROOT = pyprojroot.here()

#%% Configuration du modèle et des chemins
sam_checkpoint = PROJECT_ROOT / "outputs/SAM_tune" / "sam_finetuned_spores.pth"  
model_type = "vit_h"  # Architecture de SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
img_dir = "C:/Users/p05421/Documents/CLAQ/T5_CLAQ_C6_2"  # Dossier contenant les images à tester

#%% Charger SAM et le modèle fine-tuné
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
sam.eval()

sam_predictor = SamPredictor(sam)

#%% Sélectionner 10 images au hasard
image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpeg')]
random_images = random.sample(image_files, 10)

#%% Afficher les résultats
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
axs = axs.flatten()

for i, image_name in enumerate(random_images):
    # Charger et préparer l'image
    img_path = os.path.join(img_dir, image_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #  Passer l'image au prédicteur SAM
    sam_predictor.set_image(image_rgb)

    #  Définir un point d'interaction (par exemple, le centre de l'image)
    input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])  # (x, y)
    input_label = np.array([1])  # 1 = objet

    #  Obtenir la segmentation
    masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)

    #  Obtenir le masque prédit et appliquer un seuil
    mask_pred_np = masks[0]
    mask_pred_np = (mask_pred_np > 0.5).astype(np.uint8) * 255
    
    # Convertir le masque en image RGB pour l'affichage
    mask_pred_rgb = cv2.merge([mask_pred_np] * 3)  # (H, W) -> (H, W, 3)
    
    # Assurez-vous que les tailles des images correspondent
    if image_rgb.shape[:2] != mask_pred_rgb.shape[:2]:
        mask_pred_rgb = cv2.resize(mask_pred_rgb, (image_rgb.shape[1], image_rgb.shape[0]))

    #  Convertir l'image en RGB pour l'affichage
    combined = cv2.addWeighted(image_rgb, 0.7, mask_pred_rgb, 0.3, 0)

    #  Dessiner une boîte autour de l'objet détecté (si applicable)
    contours, _ = cv2.findContours(mask_pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Seuillage de la taille de la zone pour éviter de petites détections
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(combined, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Dessiner la boîte

    #  Afficher l'image avec la boîte de détection dans le graphique
    axs[i].imshow(combined)
    axs[i].set_title(f"Image {i+1}: {image_name}")
    axs[i].axis('off')
    
    print(f"Image {i+1} done.")

#%% Afficher le graphique combiné
plt.tight_layout()
plt.show()
