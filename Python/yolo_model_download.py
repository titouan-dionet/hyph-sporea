# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:06:59 2025

@author: Titouan Dionet
"""

#%% Packages

from ultralytics import YOLO
import os

#%% Download model function

def yolo_model_dwl(path, model = 'yolo11n.pt'):
    
    # Create folder if not already exists
    if not os.path.exists(path):
        os.makedirs(path)
        print("New folder(s) created.")
    
    # Model path
    path = os.path.join(path, model)
    print("Model loaded.")
    
    return YOLO(path)

