# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:12:04 2025

@author: p05421
"""

#%% Packages

import json
import cv2
import os
import matplotlib.pyplot as plt
import shutil
import yaml

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

#%% Paths

# Sélection du dossier d'entrée
input_path = filedialog.askdirectory(title = "Sélectionnez le dossier d'entrée")

# Sélection du dossier de sortie
output_path = filedialog.askdirectory(title = "Sélectionnez le dossier de sortie")

os.makedirs(output_path, exist_ok = True)

#%% Read Json file

file_name = '22861-trsp-21-ms-1.json'
file_path = os.path.join(input_path, file_name)

f = open(file_path)
data = json.load(f)
f.close()

#%% List files in folder

images_path = filedialog.askdirectory(title = "Sélectionnez le dossier d'images")
images_list = [img for img in os.listdir(images_path) if img.endswith(".jpeg")]

print(images_list)  # Liste des fichiers .jpeg



#%% Processing images

def load_images_from_folder(folder, output_path):
    file_names = []
    count = 0
    
    destination_path = os.path.join(output_path, "images")
    os.makedirs(destination_path, exist_ok=True)
    
    for filename in os.listdir(folder):
          source = os.path.join(folder, filename)
          destination = f"{destination_path}/img{count}.jpg"
    
          try:
              shutil.copy(source, destination)
              print("File ", filename, " copied successfully.")
          # If source and destination are same
          except shutil.SameFileError:
              print("Source and destination represents the same file.")
    
          file_names.append(filename)
          count += 1
    
    return file_names

folder_name = "images_folder"
folder_path = os.path.join(input_path, folder_name)

file_names_list = load_images_from_folder(folder_path, output_path)

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


#%%Processing labels

# Mapping des catégories
category_mapping = {}
next_category_id = 0

count = 0

destination_path = os.path.join(output_path, "labels")
os.makedirs(destination_path, exist_ok=True)

for filename in file_names_list:
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
        # Opening file for current image
        file_object = open(f"{destination_path}/img{count}.txt", "a")
      
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

#%% Création du fichier yaml

yaml_path = f"{output_path}/data.yaml"

# Récupérer les noms des catégories en gardant l'ordre d'apparition
category_ids = {cat["id"]: cat["name"] for cat in data["categories"]}
class_names = [category_ids[key] for key in sorted(category_ids.keys())]

# Création du dictionnaire pour le fichier YAML
data_yaml = {
    "train": os.path.join(output_path, "images"),
    "val": os.path.join(output_path, "images"),         # Modifie si tu as un dossier de validation séparé
    "nc": len(class_names),     # Nombre de classes
    "names": class_names        # Liste des noms des classes
}

# Écriture du fichier YAML
with open(yaml_path, "w", encoding="utf-8") as yaml_file:
    yaml.dump(data_yaml, yaml_file, default_flow_style=False, allow_unicode=True)

print(f"Fichier data.yaml généré avec succès : {yaml_path}")