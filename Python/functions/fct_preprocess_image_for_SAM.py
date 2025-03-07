# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:52:07 2025

@author: Titouan Dionet
"""

import cv2
import numpy as np
from skimage.filters import threshold_isodata

def preprocess_image(image_path, save = False, output_path = None):
    # Chargement de l'image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Conversion en 32 bits
    gray_32 = np.float32(gray)
    
    # Seuillage avec l'algorithme IsoData
    thresh_value = threshold_isodata(gray_32)
    binary_mask = gray_32 < thresh_value
    
    # Création d'un masque binaire
    mask = np.uint8(binary_mask * 255)
    
    # Appliquer le masque à l'image d'origine
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Remplacement du fond noir par du blanc
    white_background = np.ones_like(image) * 255
    final_image = np.where(result == 0, white_background, result)
    
    if save == True:
        # Sauvegarde de l'image traitée
        cv2.imwrite(output_path, final_image)
        print(f"Image prétraitée sauvegardée sous : {output_path}")
    
    return final_image