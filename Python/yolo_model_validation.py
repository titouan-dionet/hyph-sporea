# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:09:02 2025

@author: Titouan Dionet
"""

#%% Packages

import os

#%% Train model function

def yolo_model_val(model, output_path):
    
    # Create folder(s) if not already exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("New folder(s) created.")
    
    # Model validation
    path = os.path.join(output_path, "runs/val")
    val_results = model.val(project = path)
    print("Validation done.")
    
    return val_results
