# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:18:17 2025

@author: Titouan Dionet
"""

#%% Packages

import os
from PIL import Image

#%% Conversion function

def convert_tif_to_jpeg(input_dir, output_dir):
    # Parcourt les sous-dossiers
    for root, _, files in os.walk(input_dir):
        # Conserve la structure des sous-dossiers
        relative_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(save_dir, os.path.splitext(file)[0] + ".jpeg")

                # Convertit l'image
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    img.save(output_path, "JPEG", quality=95)
                print(f"Converted: {input_path} -> {output_path}")

