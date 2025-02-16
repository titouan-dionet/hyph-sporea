# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:54:27 2025

@author: Titouan Dionet
"""

#%% Packages

from ultralytics import YOLO
import pandas as pd
import os

#%% Functions

from Python.yolo_model_download   import yolo_model_dwl
from Python.yolo_model_train      import yolo_model_train
from Python.yolo_model_validation import yolo_model_val

#%% Pipeline

# Download and load model
model = yolo_model_dwl(path = "./outputs/model", model = 'yolo11m.pt')

# Train the model
data_yaml = "./data/hypho_data/data.yaml"
train_results = yolo_model_train(model = model, data_yaml_path = data_yaml, epochs = 50, imgsz = 800, batch = 16, output_path = "./outputs/model", name = "hypho_model")

# Validate the model
val_results = yolo_model_val(model = model, output_path = "./outputs/model")

