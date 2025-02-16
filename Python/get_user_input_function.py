# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:41:48 2025

@author: Titouan Dionet
"""

#%% Packages

import tkinter as tk
from tkinter import filedialog

#%% Fonction : demander les dossiers entrée/sortie à l'utilisateur

def get_user_input():
    root = tk.Tk() # Création de la boîte de dialogue

    # Sélection du dossier d'entrée
    input_folder = filedialog.askdirectory(title="Sélectionnez le dossier d'entrée")

    # Sélection du dossier de sortie
    output_folder = filedialog.askdirectory(title="Sélectionnez le dossier de sortie")

    root.destroy() # Destruction de la boîte de dialogue
    return input_folder, output_folder

