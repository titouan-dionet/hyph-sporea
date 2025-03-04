# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:11:54 2025

@author: Titouan Dionet
"""

#%% Packages

from ultralytics import YOLO
from ultralytics import SAM
import os

#%% Download model function

def model_dwl(path, model):
    """
    Downloads and loads a model based on the provided model name.
    It checks whether the model is either a YOLO model or a SAM model and loads it accordingly.
    If the model type is not supported, an error is raised.

    Parameters
    ----------
    path : str
        The directory where the model file will be saved or is located.
    model : str
        The name of the model file. It can be either a YOLO or SAM model. The function will
        attempt to load the model based on the name.

    Raises
    ------
    ValueError
        If the model name does not contain 'yolo' or 'sam', a ValueError is raised, indicating that
        the model type is unsupported.

    Returns
    -------
    None
        The function does not return any value. It simply prints out messages regarding the model loading.
    """
    
    # Create folder if not already exists
    if not os.path.exists(path):
        os.makedirs(path)
        print("New folder(s) created.")
    
    # Model path
    model_path = os.path.join(path, model)
    
    # Check model type and execute corresponding model loading
    if 'yolo' in model.lower():
        YOLO(model_path)
        print(f"YOLO model {model} loaded.")
    elif 'sam' in model.lower():
        SAM(model_path)
        print(f"SAM model {model} loaded.")
    else:
        raise ValueError("Model type not supported. Please provide a YOLO or SAM model.")
    
    return print(f"Model {model} downloaded and ready.")
