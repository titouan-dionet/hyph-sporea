# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:54:27 2025

@author: Titouan Dionet
"""

#%% Packages

from ultralytics import YOLO
import pandas as pd
import os
import pyprojroot

#%% Functions

from Python.yolo_model_download   import yolo_model_dwl
from Python.yolo_model_train      import yolo_model_train
from Python.yolo_model_validation import yolo_model_val

#%% Racine du projet 
PROJECT_ROOT = pyprojroot.here()

#%% Pipeline

# Download and load model
model = yolo_model_dwl(path = PROJECT_ROOT / "data/blank_models", model = 'yolo11m.pt')

# Train the model
data_yaml = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/data.yaml"
train_results = yolo_model_train(model = model, data_yaml_path = data_yaml, epochs = 50, imgsz = 800, batch = 16, output_path = PROJECT_ROOT / "outputs/model_CLAQ", name = "train")

# train_results = yolo_model_train(model = model, data_yaml_path = data_yaml, epochs = 1, imgsz = 250, batch = 5, output_path = "../outputs/model_CLAQ", name = "train")

# Validate the model
val_results = yolo_model_val(model = model, output_path = PROJECT_ROOT / "outputs/model_CLAQ") 

# Predictions
dataset_path = PROJECT_ROOT / "data/raw_data/hypho_images/T5_CLAQ_MS"
results = model(dataset_path, save = True, project = PROJECT_ROOT / "outputs/model_CLAQ/predict", name = "T5_CLAQ_MS")


all_detections = []
summary_counts = {}

for r in results:
    image_name = os.path.basename(r.path)
    detections = r.boxes.data.cpu().numpy()  # Obtenir les bounding boxes
    
    if len(detections) == 0:
        summary_counts[image_name] = {"CLAQ": 0}
    
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
            summary_counts[image_name] = {"CLAQ": 0}
        summary_counts[image_name][class_name] += 1
        
        
results_path = PROJECT_ROOT / "outputs/model_CLAQ/results"

columns = ["image", "class", "center_x", "center_y", "width", "height", "confidence"]
df_detections = pd.DataFrame(all_detections, columns=columns)

file_name = "CLAQ_detections_details.csv"

df_detections.to_csv(os.path.join(results_path, file_name), index=False)

#%% Sauvegarde des résultats par image

summary_list = [[img, counts["CLAQ"]] for img, counts in summary_counts.items()]
df_summary = pd.DataFrame(summary_list, columns=["image", "CLAQ_count"])

file_name = "CLAQ_detections_summary.csv"

df_summary.to_csv(os.path.join(results_path, file_name), index=False)

print("Export terminé : fichiers 'detections_details.csv' et 'detections_summary.csv' générés.")


