# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:22:55 2025

@author: p05421
"""

import cv2
import os
import glob
import random

# Chemins des dossiers
image_folder = "C:/Users/p05421/Documents/Hypho_Biigle/train/images"
label_folder = "C:/Users/p05421/Documents/Hypho_Biigle/train/labels"
output_folder = "C:/Users/p05421/Documents/Hypho_Biigle/annotated_images"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Générer des couleurs aléatoires pour chaque catégorie
category_colors = {}

# Parcourir les images
for image_path in glob.glob(os.path.join(image_folder, "*.jpg")):  
    filename = os.path.basename(image_path).split(".")[0]  # Récupère le nom sans extension
    label_path = os.path.join(label_folder, filename + ".txt")

    # Vérifier si le fichier de labels existe
    if not os.path.exists(label_path):
        print(f"Pas de labels pour {filename}, image ignorée.")
        continue

    # Charger l'image
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Lire le fichier des annotations
    with open(label_path, "r") as file:
        lines = file.readlines()

    # Dessiner chaque bounding box
    for line in lines:
        values = line.strip().split()
        category_id = int(values[0])  # ID de la catégorie
        x_center, y_center, w, h = map(float, values[1:])

        # Convertir les coordonnées normalisées en pixels
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        # Générer une couleur unique pour chaque catégorie
        if category_id not in category_colors:
            category_colors[category_id] = [random.randint(0, 255) for _ in range(3)]

        color = category_colors[category_id]

        # Dessiner le rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(category_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sauvegarder l'image annotée
    output_path = os.path.join(output_folder, filename + "_annotated.jpg")
    cv2.imwrite(output_path, img)

    print(f"Image annotée sauvegardée : {output_path}")
