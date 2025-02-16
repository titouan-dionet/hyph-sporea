# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:45:20 2025

@author: Titouan Dionet
"""

#%% Packages

import os
from PIL import Image

# import tkinter as tk
# from tkinter import ttk
from tkinter import filedialog

# from Python.get_user_input_function import get_user_input
from Python.tiff_to_jpeg_function import convert_tif_to_jpeg

#%% Initialisation de tkinter
# Cette initialisation est nécessaire pour le bon déroulement du programme

# init = tk.Tk()
# init.title("Initialisation")
# init.resizable(0,0)
# init.wm_attributes("-topmost", 1)

# text_label = tk.Label(init, text="Initialisation of the program.\nPlease, click OK to continue.", padx=10, pady=10)
# text_label.pack()

# close_button = ttk.Button(init, text="OK", command=init.destroy)
# close_button.pack()

# init.mainloop()

#%% Choix des dossier d'entrée et de sortie

# input_dir, output_dir = get_user_input()

# Sélection du dossier d'entrée
input_dir = filedialog.askdirectory(title = "Sélectionnez le dossier d'entrée")

# Sélection du dossier de sortie
output_dir = filedialog.askdirectory(title = "Sélectionnez le dossier de sortie")

# Affichage des valeurs dans la console
print("Dossier d'entrée :", input_dir)
print("Dossier de sortie :", output_dir)

#%% Conversion des images TIFF en JPEG

convert_tif_to_jpeg(input_dir, output_dir)