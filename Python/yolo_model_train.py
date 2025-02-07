# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:58:00 2025

@author: Titouan Dionet
"""

#%% Packages

import os

#%% Train model function

def yolo_model_train(model, data_yaml_path, epochs, imgsz, batch, output_path, name = "model"):
    
    # Create folder(s) if not already exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("New folder(s) created.")
    
    # Model train
    path = os.path.join(output_path, "runs/train")
    train_results = model.train(data = data_yaml_path, epochs = epochs, imgsz = imgsz, batch = batch, name = name, project = path)
    print("Training done.")
    
    return train_results
