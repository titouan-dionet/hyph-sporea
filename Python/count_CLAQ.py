# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:09:34 2025

@author: Titouan Dionet
"""

#%% Packages

from ultralytics import YOLO
import torch
import pandas as pd
import os
from datetime import datetime

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pyprojroot

from pathlib import Path
import gc  # Importer le module garbage collector
from itertools import islice

#%% Racine du projet 
PROJECT_ROOT = pyprojroot.here()

#%% Initialisation de tkinter
# Cette initialisation est nécessaire pour le bon déroulement du programme

init = tk.Tk()
init.title("Initialisation")
init.resizable(0,0)
init.wm_attributes("-topmost", 1)

text_label = tk.Label(init, text="Initialisation of the program.\nPlease, click OK to continue.", padx=10, pady=10)
text_label.pack()

close_button = ttk.Button(init, text="OK", command=init.destroy)
close_button.pack()

init.mainloop()

#%% Sélection des chemins d'accès

# Récupérer toutes les images dans le dossier et ses sous-dossiers
def get_all_images(dataset_path):
    dataset_path = Path(dataset_path)
    image_paths = list(dataset_path.rglob("*.jpeg")) + list(dataset_path.rglob("*.jpg"))  # Cherche toutes les images
    return image_paths


dataset_path = filedialog.askdirectory(title = "Choose the dataset path.")

# Récupération des images dans les sous-dossiers
image_files = get_all_images(dataset_path)

# Vérification
if not image_files:
    print("Aucune image trouvée dans les sous-dossiers !")
else:
    print(f"{len(image_files)} images trouvées.")


model_path = filedialog.askopenfilename(
    title = "Sélectionner un fichier",
    filetypes = [("Modèles YOLO", "*.pt"), ("Tous les fichiers", "*.*")]
)

#%% Modèle

# model_path = "../outputs/model/runs/train/hypho_model3/weights/best.pt"
model = YOLO(model_path)

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_name = os.path.basename(dataset_path)
output_dir = f"{date_str}_{dataset_name}"

#%%

all_detections = []
summary_counts = {}

for img_path in image_files:
    image_name = os.path.basename(img_path)
    sample_name = "_".join(image_name.split("_")[:3])
    sample_dir = PROJECT_ROOT / "outputs/model_CLAQ/predict" / f"{date_str}_{sample_name}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    results = model(str(img_path), save=True, project=str(sample_dir), name="", exist_ok=True)
    
    for r in results:
        image_name = os.path.basename(r.path)
        detections = r.boxes.data.cpu().numpy()

        if len(detections) == 0:
            summary_counts[image_name] = {"CLAQ": 0}

        for det in detections:
            x_min, y_min, x_max, y_max, confidence, class_id = det
            class_name = model.names[int(class_id)]

            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            width, height = x_max - x_min, y_max - y_min

            all_detections.append([image_name, class_name, center_x, center_y, width, height, confidence])

            if image_name not in summary_counts:
                summary_counts[image_name] = {"CLAQ": 0}
            summary_counts[image_name][class_name] += 1

    # Vide la mémoire GPU après chaque image !
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"{image_name} done")

#%% Sauvegarde des résultats détaillés

results_path = PROJECT_ROOT / "outputs/model_CLAQ/results"
os.makedirs(results_path, exist_ok = True)

os.makedirs(os.path.join(results_path, "bounding_box_coord"), exist_ok = True)
os.makedirs(os.path.join(results_path, "object_count"), exist_ok = True)

columns = ["image", "class", "center_x", "center_y", "width", "height", "confidence"]
df_detections = pd.DataFrame(all_detections, columns=columns)

file_name = f"{date_str}_{dataset_name}_detections_details.csv"

df_detections.to_csv(os.path.join(results_path, "bounding_box_coord", file_name), index=False)

#%% Sauvegarde des résultats par image

summary_list = [[img, counts["CLAQ"]] for img, counts in summary_counts.items()]
df_summary = pd.DataFrame(summary_list, columns=["image", "CLAQ_count"])

file_name = f"{date_str}_{dataset_name}_detections_summary.csv"

df_summary.to_csv(os.path.join(results_path, "object_count", file_name), index=False)

print("Export terminé : fichiers 'detections_details.csv' et 'detections_summary.csv' générés.")
print(f"Les images prédites sont enregistrées dans : {output_dir}")


