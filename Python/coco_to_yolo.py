# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:33:44 2025

@author: Titouan Dionet
"""

#%% Packages

import json
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import yaml
import random

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

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

#%% Chemins d'accès

# Images brutes
# C:\Users\p05421\Titouan\10. Programmation\Python\hyph-sporea\data\raw_data\hypho_images\T5_CLAQ_MS

# Fichier JSON d'annotation
# "C:\Users\p05421\Titouan\10. Programmation\Python\hyph-sporea\data\proc_data\23160-claq.json"

#%% Read Json file

# file_name = '22861-trsp-21-ms-1.json'
# file_path = os.path.join(input_path, file_name)

json_path = filedialog.askopenfilename(
    title = "Sélectionner le fichier JSON correspondant aux labels d'annotations.",
    filetypes = [("JSON files", "*.json"), ("Tous les fichiers", "*.*")]
)

f = open(json_path)
data = json.load(f)
f.close()

#%% List files in folder

images_path = filedialog.askdirectory(title = "Sélectionnez le dossier d'images correspondant")
images_list = [img for img in os.listdir(images_path) if img.endswith(".jpeg")]

print(images_list)  # Liste des fichiers .jpeg

#%% Helper functions

def get_img_ann(image_id):
    img_ann = []
    isFound = False
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            img_ann.append(ann)
            isFound = True
    if isFound:
        return img_ann
    else:
        return None
    
def get_img(filename):
    for img in data['images']:
        if img['file_name'] == filename:
            return img
    return None # Return None if image is not in data

#%% Processing labels for training

# Mapping des catégories
category_mapping = {}
next_category_id = 0

count = 0

# Sélection du dossier de sortie
output_path = filedialog.askdirectory(title = "Sélectionnez le dossier d'entrainement du modèle.")

# Création des dossier 'train'
train_path = os.path.join(output_path, "train")
os.makedirs(train_path, exist_ok=True)

train_images_path = os.path.join(train_path, "images")
train_labels_path = os.path.join(train_path, "labels")

os.makedirs(train_images_path, exist_ok = True)
os.makedirs(train_labels_path, exist_ok = True)

# val_path = os.path.join(output_path, "val")
# os.makedirs(val_path, exist_ok=True)


for filename in images_list:
    # Extracting image 
    img = get_img(filename)
    print("File selected : ", filename)
    
    if img == None : continue

    img_id = img['id']
    img_w = img['width']
    img_h = img['height']
    
    # Get Annotations for this image
    img_ann = get_img_ann(img_id)
    
    if img_ann:
        
        # Copie de l'image dans le dossier 'train/images'
        source_path = os.path.join(images_path, filename)
        destination_path = os.path.join(train_images_path, filename)
        try:
            shutil.copy(source_path, destination_path)
            print("File ", filename, " copied successfully.")
        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        
        # Opening file for current image
        basename_file = os.path.splitext(filename)[0] # Suppression de l'extension
        file_object = open(f"{train_labels_path}/{basename_file}.txt", "a")
      
        for ann in img_ann:
            
            original_category = ann['category_id']
            
            # Vérifier si la catégorie est déjà mappée, sinon lui attribuer un nouvel ID
            if original_category not in category_mapping:
                category_mapping[original_category] = next_category_id
                next_category_id += 1  # Incrémente l'ID pour la prochaine catégorie inconnue

            # Récupérer l'ID YOLO correspondant
            current_category = category_mapping[original_category]
            
            current_bbox = ann['bbox']
            x = current_bbox[0]
            y = current_bbox[1]
            w = current_bbox[2]
            h = current_bbox[3]
            
            # Finding midpoints
            x_centre = (x + (x+w))/2
            y_centre = (y + (y+h))/2
            
            # Normalization
            x_centre = x_centre / img_w
            y_centre = y_centre / img_h
            w = w / img_w
            h = h / img_h
            
            # Limiting upto fix number of decimal places
            x_centre = format(x_centre, '.6f')
            y_centre = format(y_centre, '.6f')
            w = format(w, '.6f')
            h = format(h, '.6f')
                
            # Writing current object 
            file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")
            
    file_object.close()
    
    count += 1  # This should be outside the if img_ann block.

print("Mapping des catégories :", category_mapping)

#%% Creating a validation dataset

# Création des dossier 'val'
val_path = os.path.join(output_path, "val")
os.makedirs(val_path, exist_ok=True)

val_images_path = os.path.join(val_path, "images")
val_labels_path = os.path.join(val_path, "labels")

os.makedirs(val_images_path, exist_ok = True)
os.makedirs(val_labels_path, exist_ok = True)

# Liste des images dans train/images
images_train = [f for f in os.listdir(train_images_path) if f.endswith((".jpeg", ".jpg"))]

# Sélectionner aléatoirement 10 images (ou toutes si moins de 10)
nb_images = min(10, len(images_train))
images_selected = random.sample(images_train, nb_images)

# Copier les fichiers sélectionnés vers val/
for img in images_selected:
    # Définition des chemins
    img_train_path = os.path.join(train_images_path, img)
    img_val_path = os.path.join(val_images_path, img)

    # Copier l'image
    shutil.copy(img_train_path, img_val_path)

    # Gérer le fichier label correspondant
    img_label = os.path.splitext(img)[0] + ".txt"
    label_train_path = os.path.join(train_labels_path, img_label)
    label_val_path = os.path.join(val_labels_path, img_label)

    # Vérifier si le fichier label existe avant de copier
    if os.path.exists(label_train_path):
        shutil.copy(label_train_path, label_val_path)

print(f"{nb_images} images et leurs labels ont été copiés dans 'val'.")

#%% Création du fichier yaml

yaml_path = f"{output_path}/data.yaml"

# Récupérer les noms des catégories en gardant l'ordre d'apparition
category_ids = {cat["id"]: cat["name"] for cat in data["categories"]}
class_names = [category_ids[key] for key in sorted(category_ids.keys())]

# Création du dictionnaire pour le fichier YAML
data_yaml = {
    "train": train_images_path.replace("\\", "/"),
    "val": val_images_path.replace("\\", "/"),     
    "nc": len(class_names),     # Nombre de classes
    "names": class_names        # Liste des noms des classes
}

# Écriture du fichier YAML
with open(yaml_path, "w", encoding="utf-8") as yaml_file:
    yaml.dump(data_yaml, yaml_file, sort_keys=False, default_flow_style=None, allow_unicode=True)

print(f"Fichier data.yaml généré avec succès : {yaml_path}")