# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:57:00 2025

@author: Titouan Dionet
"""

#%% Packages

from ultralytics import SAM
import pyprojroot
from Python.functions.fct_model_download import model_dwl
from Python.yolo_model_train      import yolo_model_train
from Python.yolo_model_validation import yolo_model_val
from Python.functions.fct_preprocess_image_for_SAM import preprocess_image
import cv2

#%% Racine du projet 
PROJECT_ROOT = pyprojroot.here()

#%% Pipeline

# Download and load a model
model = model_dwl(path = PROJECT_ROOT / "data/blank_models", model = 'sam2.1_l.pt')

# Display model information (optional)
model.info()

#%% Results

img_path = "C:/Users/p05421/Documents/CLAQ/T5_CLAQ_C6_2/T5_CLAQ_C6_2_000079.jpeg"

image = preprocess_image(img_path)
print("Image preprocessed")

cv2.imshow("Detections SAM", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

results = model.predict(image, project = PROJECT_ROOT / "outputs/SAM_test", save = True)



#%% Results

# Train the model
data_yaml = PROJECT_ROOT / "data/proc_data/hypho_train_data_model/model_CLAQ/data.yaml"
train_results = yolo_model_train(model = model, data_yaml_path = data_yaml, epochs = 5, imgsz = 800, batch = 4, output_path = PROJECT_ROOT / "outputs/SAM_test", name = "train")


img_path = "C:/Users/p05421/Documents/CLAQ/T5_CLAQ_C6_2/T5_CLAQ_C6_2_000079.jpeg"

# Run 
results = model.predict(img_path, project = PROJECT_ROOT / "outputs/SAM_test", save = True)


#%%

# Run inference with single point
results = model(points=[900, 370], labels=[1], save = True)

# Run inference with multiple points
results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

# Run inference with multiple points prompt per object
results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# Run inference with negative points prompt
results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])