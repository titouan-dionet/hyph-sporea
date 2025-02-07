# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:12:56 2025

@author: Titouan Dionet
"""

#%% Packages

from ultralytics import YOLO
import torch
import pandas as pd
import os

#%% Modèle

# 1. Charger le modèle YOLO pré-entraîné 
model = YOLO('../outputs/model/yolo11m.pt')  # Remplace par ton propre modèle si nécessaire

# 2. Entraînement du modèle (ajuste les paramètres selon tes besoins)
data_yaml = "../data/hypho_data/data.yaml"
model.train(data = data_yaml, epochs = 5, imgsz = 640, batch = 16, name = "hypho_model", project="../outputs/model/runs/train")

# 3. Validation du modèle
model.val(project = "../outputs/model/runs/val")

# 4. Analyse des résultats sur un ensemble d'images
dataset_path = "C:/Users/p05421/Documents/hypho/TRSP_21_MS_1"
results = model(dataset_path)

# 5. Extraction des données et export en CSV
all_detections = []
summary_counts = {}

for r in results:
    image_name = os.path.basename(r.path)
    detections = r.boxes.data.cpu().numpy()  # Obtenir les bounding boxes
    
    if len(detections) == 0:
        summary_counts[image_name] = {"Spores": 0, "Debris": 0, "Mycelium": 0}
    
    for det in detections:
        x_min, y_min, x_max, y_max, confidence, class_id = det
        class_name = model.names[int(class_id)]  # Nom de l'objet détecté
        
        # Calculer les coordonnées du centre et les dimensions
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        width, height = x_max - x_min, y_max - y_min
        
        # Ajouter aux résultats détaillés
        all_detections.append([image_name, class_name, center_x, center_y, width, height, confidence])
        
        # Compter les objets par image
        if image_name not in summary_counts:
            summary_counts[image_name] = {"Spores": 0, "Debris": 0, "Mycelium": 0}
        summary_counts[image_name][class_name] += 1

# Sauvegarde des résultats détaillés
columns = ["image", "class", "center_x", "center_y", "width", "height", "confidence"]
df_detections = pd.DataFrame(all_detections, columns=columns)
df_detections.to_csv("../outputs/model/results/detections_details.csv", index=False)

# Sauvegarde des résultats par image
summary_list = [[img, counts["Spores"], counts["Debris"], counts["Mycelium"]] for img, counts in summary_counts.items()]
df_summary = pd.DataFrame(summary_list, columns=["image", "spore_count", "debris_count", "mycelium_count"])
df_summary.to_csv("../outputs/model/results/detections_summary.csv", index=False)

print("Export terminé : fichiers 'detections_details.csv' et 'detections_summary.csv' générés.")
