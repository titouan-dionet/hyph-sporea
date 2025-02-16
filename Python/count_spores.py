# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:50:09 2025

@author: Titouan Dionet
"""

#%% Packages

from ultralytics import YOLO
# import torch
import pandas as pd
import os
from datetime import datetime

# import tkinter as tk
# from tkinter import ttk
from tkinter import filedialog

#%% Initialisation de tkinter
# # Cette initialisation est nécessaire pour le bon déroulement du programme

# init = tk.Tk()
# init.title("Initialisation")
# init.resizable(0,0)
# init.wm_attributes("-topmost", 1)

# text_label = tk.Label(init, text="Initialisation of the program.\nPlease, click OK to continue.", padx=10, pady=10)
# text_label.pack()

# close_button = ttk.Button(init, text="OK", command=init.destroy)
# close_button.pack()

# init.mainloop()

#%% Sélection des chemins d'accès

dataset_path = filedialog.askdirectory(title = "Choose the dataset path.")

# model_path = filedialog.askopenfilename(
#     title = "Sélectionner un fichier",
#     filetypes = [("Modèles YOLO", "*.pt"), ("Tous les fichiers", "*.*")]
# )

#%% Modèle

model_path = "../outputs/model/runs/train/hypho_model3/weights/best.pt"
model = YOLO(model_path)

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_name = os.path.basename(dataset_path)
output_dir = f"{date_str}_{dataset_name}"

results = model(dataset_path, save = True, project = "../outputs/model/predict", name = output_dir)

#%% Extraction des données et export en CSV
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

#%% Sauvegarde des résultats détaillés

results_path = "../outputs/model/results"

columns = ["image", "class", "center_x", "center_y", "width", "height", "confidence"]
df_detections = pd.DataFrame(all_detections, columns=columns)

file_name = f"{date_str}_{dataset_name}_detections_details.csv"

df_detections.to_csv(os.path.join(results_path, file_name), index=False)

#%% Sauvegarde des résultats par image

summary_list = [[img, counts["Spores"], counts["Debris"], counts["Mycelium"]] for img, counts in summary_counts.items()]
df_summary = pd.DataFrame(summary_list, columns=["image", "spore_count", "debris_count", "mycelium_count"])

file_name = f"{date_str}_{dataset_name}_detections_summary.csv"

df_summary.to_csv(os.path.join(results_path, file_name), index=False)

print("Export terminé : fichiers 'detections_details.csv' et 'detections_summary.csv' générés.")
print(f"Les images prédites sont enregistrées dans : {output_dir}")


