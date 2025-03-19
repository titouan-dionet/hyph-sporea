#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface graphique pour l'assemblage d'images du projet HYPH-SPOREA.

Cette interface permet de :
- Sélectionner un dossier d'images
- Sélectionner un fichier de grille
- Choisir les options d'assemblage
- Lancer l'assemblage ou la détection d'overlap
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import json
import time
import datetime
from io import StringIO
from contextlib import redirect_stdout

# Importer les fonctions d'assemblage et de détection
try:
    from image_stitcher import stitch_images, estimate_overlap_from_sample
except ImportError:
    # Si le module n'est pas dans le PYTHONPATH, essayer d'importer depuis le répertoire courant
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from image_stitcher import stitch_images, estimate_overlap_from_sample


class StitcherGUI:
    """Interface graphique pour l'assemblage d'images"""
    
    def __init__(self, root):
        """Initialise l'interface graphique"""
        self.root = root
        self.root.title("Assemblage d'images HYPH-SPOREA")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)
        
        # Variables pour les entrées utilisateur
        self.input_dir = tk.StringVar()
        self.grid_file = tk.StringVar()
        self.output_dir = tk.StringVar()  # Nouveau: dossier de sortie séparé
        self.output_filename = tk.StringVar()  # Nouveau: nom de fichier de sortie
        self.sample_name = tk.StringVar()
        self.h_overlap = tk.IntVar(value=105)
        self.v_overlap = tk.IntVar(value=93)
        self.pixel_size = tk.DoubleVar(value=0.264)
        self.use_tiff = tk.BooleanVar(value=True)
        self.auto_overlap = tk.BooleanVar(value=False)
        self.num_samples = tk.IntVar(value=20)
        
        # Créer l'interface
        self.create_widgets()
        
        # État de traitement
        self.processing = False
        self.start_time = None
        self.total_files = 0
        self.processed_files = 0
    
    def create_widgets(self):
        """Crée les widgets de l'interface graphique"""
        # Frame principale avec padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style pour les titres de section
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 11, "bold"))
        
        # Section 1: Sélection de fichiers
        file_frame = ttk.LabelFrame(main_frame, text="Sélection des fichiers", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        # Dossier d'images
        ttk.Label(file_frame, text="Dossier d'images:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, sticky=tk.W+tk.E, pady=2)
        ttk.Button(file_frame, text="Parcourir", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=2)
        
        # Fichier de grille
        ttk.Label(file_frame, text="Fichier de grille:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.grid_file, width=50).grid(row=1, column=1, sticky=tk.W+tk.E, pady=2)
        ttk.Button(file_frame, text="Parcourir", command=self.browse_grid_file).grid(row=1, column=2, padx=5, pady=2)
        
        # Dossier de sortie
        ttk.Label(file_frame, text="Dossier de sortie:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, sticky=tk.W+tk.E, pady=2)
        ttk.Button(file_frame, text="Parcourir", command=self.browse_output_dir).grid(row=2, column=2, padx=5, pady=2)
        
        # Nom du fichier de sortie
        ttk.Label(file_frame, text="Fichier de sortie:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(file_frame, textvariable=self.output_filename, width=50).grid(row=3, column=1, sticky=tk.W+tk.E, pady=2)
        
        # Configurer les poids de colonnes
        file_frame.columnconfigure(1, weight=1)
        
        # Section 2: Options d'assemblage
        options_frame = ttk.LabelFrame(main_frame, text="Options d'assemblage", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        
        # Nom de l'échantillon
        ttk.Label(options_frame, text="Nom de l'échantillon:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(options_frame, textvariable=self.sample_name, width=20).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(options_frame, text="(Laisser vide pour auto-détection)").grid(row=0, column=2, sticky=tk.W, pady=2)
        
        # Option de format d'image
        ttk.Label(options_frame, text="Format d'image:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(options_frame, text="TIFF (tif/tiff)", variable=self.use_tiff, value=True).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Radiobutton(options_frame, text="JPEG (jpeg/jpg)", variable=self.use_tiff, value=False).grid(row=1, column=2, sticky=tk.W, pady=2)
        
        # Sous-section: Overlap
        overlap_frame = ttk.Frame(options_frame)
        overlap_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E, pady=5)
        
        # Option d'auto-détection
        ttk.Checkbutton(
            overlap_frame,
            text="Détection automatique du chevauchement (overlap)",
            variable=self.auto_overlap,
            command=self.toggle_overlap_fields
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # Nombre d'échantillons pour auto-détection
        ttk.Label(overlap_frame, text="Nombre d'échantillons:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            overlap_frame, 
            from_=5, 
            to=50, 
            increment=5, 
            textvariable=self.num_samples, 
            width=5
        ).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Overlap manuel
        ttk.Label(overlap_frame, text="Chevauchement horizontal (pixels):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.h_overlap_spinbox = ttk.Spinbox(
            overlap_frame, 
            from_=0, 
            to=300, 
            increment=1, 
            textvariable=self.h_overlap, 
            width=5
        )
        self.h_overlap_spinbox.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(overlap_frame, text="Chevauchement vertical (pixels):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.v_overlap_spinbox = ttk.Spinbox(
            overlap_frame, 
            from_=0, 
            to=300, 
            increment=1, 
            textvariable=self.v_overlap, 
            width=5
        )
        self.v_overlap_spinbox.grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Taille du pixel
        ttk.Label(options_frame, text="Taille du pixel (mm):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(
            options_frame, 
            from_=0.001, 
            to=1.0, 
            increment=0.001, 
            textvariable=self.pixel_size, 
            width=8,
            format="%.3f"
        ).grid(row=3, column=1, sticky=tk.W, pady=2)
        ttk.Label(options_frame, text="(0.0104 pouces = 0.264 mm)").grid(row=3, column=2, sticky=tk.W, pady=2)
        
        # Section 3: Options de visualisation
        grid_frame = ttk.LabelFrame(main_frame, text="Options de visualisation", padding="10")
        grid_frame.pack(fill=tk.X, pady=5)
        
        # Option d'affichage de la grille
        ttk.Checkbutton(
            grid_frame,
            text="Afficher une grille entre les images",
            variable=self.show_grid
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Couleur de la grille
        ttk.Label(grid_frame, text="Couleur de la grille:").grid(row=1, column=0, sticky=tk.W, pady=2)
        grid_color_frame = ttk.Frame(grid_frame)
        grid_color_frame.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Options de couleur pour la grille (noir, blanc, rouge, vert, bleu)
        ttk.Radiobutton(grid_color_frame, text="Noir", variable=self.grid_color, value="black").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(grid_color_frame, text="Blanc", variable=self.grid_color, value="white").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(grid_color_frame, text="Rouge", variable=self.grid_color, value="red").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(grid_color_frame, text="Vert", variable=self.grid_color, value="green").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(grid_color_frame, text="Bleu", variable=self.grid_color, value="blue").pack(side=tk.LEFT, padx=5)
        
        # Transparence de la grille
        ttk.Label(grid_frame, text="Transparence de la grille (%):").grid(row=2, column=0, sticky=tk.W, pady=2)
        grid_alpha_frame = ttk.Frame(grid_frame)
        grid_alpha_frame.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        self.grid_alpha_slider = ttk.Scale(
            grid_alpha_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.grid_alpha,
            length=200
        )
        self.grid_alpha_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(grid_alpha_frame, textvariable=self.grid_alpha_display).pack(side=tk.LEFT, padx=5)
        
        # Option d'affichage des numéros d'image
        ttk.Checkbutton(
            grid_frame,
            text="Afficher les numéros d'image",
            variable=self.show_numbers
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Configurer état initial des champs d'overlap
        self.toggle_overlap_fields()
        
        # Section 4: Actions
        actions_frame = ttk.Frame(main_frame, padding="10")
        actions_frame.pack(fill=tk.X, pady=10)
        
        # Boutons d'action
        ttk.Button(
            actions_frame, 
            text="Détecter l'overlap seulement", 
            command=self.run_overlap_detection,
            width=25
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            actions_frame, 
            text="Assembler les images", 
            command=self.run_stitching,
            width=25
        ).pack(side=tk.LEFT, padx=5)
        
        # Bouton Quitter
        ttk.Button(
            actions_frame, 
            text="Fin", 
            command=self.quit_app,
            width=10
        ).pack(side=tk.RIGHT, padx=5)
        
        # Bouton pour effacer le journal
        ttk.Button(
            actions_frame, 
            text="Effacer journal", 
            command=self.clear_log,
            width=15
        ).pack(side=tk.RIGHT, padx=5)
        
        # Section 5: Journal des opérations
        log_frame = ttk.LabelFrame(main_frame, text="Journal des opérations", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Zone de texte pour le journal
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Barre de défilement
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Désactiver la modification directe du journal
        self.log_text.config(state=tk.DISABLED)
        
        # Barre de statut
        self.status_var = tk.StringVar(value="Prêt")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Barre de progression
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Configurer les callbacks pour les mises à jour de l'interface
        self.grid_alpha.trace_add("write", self.update_grid_alpha_display)
    
    def browse_input_dir(self):
        """Ouvre un dialogue pour sélectionner le dossier d'images"""
        directory = filedialog.askdirectory(title="Sélectionner le dossier d'images")
        if directory:
            self.input_dir.set(directory)
            # Mise à jour automatique du chemin de sortie
            self.update_output_paths()
            
            # Tenter de trouver un fichier de grille associé
            self.try_find_grid_file(directory)
    
    def browse_grid_file(self):
        """Ouvre un dialogue pour sélectionner le fichier de grille"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner le fichier de grille",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if file_path:
            self.grid_file.set(file_path)
    
    def browse_output_dir(self):
        """Ouvre un dialogue pour sélectionner le dossier de sortie"""
        directory = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if directory:
            self.output_dir.set(directory)
            # Mettre à jour le nom de fichier en fonction du dossier
            self.update_output_filename()
    
    def browse_output_path(self):
        """Ouvre un dialogue pour sélectionner le fichier de sortie (obsolète)"""
        file_path = filedialog.asksaveasfilename(
            title="Enregistrer l'image assemblée sous",
            defaultextension=".jpeg",
            filetypes=[("Images JPEG", "*.jpeg"), ("Images PNG", "*.png"), ("Tous les fichiers", "*.*")]
        )
        if file_path:
            # Extraire le dossier et le nom de fichier
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            
            self.output_dir.set(directory)
            self.output_filename.set(filename)
    
    def update_output_paths(self):
        """Met à jour automatiquement le dossier et le nom de fichier de sortie"""
        if not self.input_dir.get():
            return
            
        # Définir le dossier de sortie par défaut (parent du dossier d'entrée)
        output_dir = os.path.dirname(self.input_dir.get())
        self.output_dir.set(output_dir)
        
        # Mettre à jour le nom du fichier
        self.update_output_filename()
    
    def update_output_filename(self):
        """Met à jour le nom du fichier de sortie en fonction du nom de l'échantillon"""
        # Déterminer le nom de l'échantillon
        sample_name = self.sample_name.get()
        
        if not sample_name:
            # Essayer de détecter automatiquement le nom de l'échantillon
            file_extensions = [".jpeg", ".jpg"] if not self.use_tiff.get() else [".tif", ".tiff"]
            input_dir = self.input_dir.get()
            
            if not input_dir or not os.path.isdir(input_dir):
                return
                
            files = []
            for ext in file_extensions:
                files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
            
            if files:
                first_file = files[0]
                sample_name = "_".join(first_file.split("_")[:-1])
            else:
                # Utiliser le nom du répertoire
                sample_name = os.path.basename(input_dir)
        
        # Définir le nom du fichier de sortie
        output_filename = f"{sample_name}_assemblee.jpeg"
        self.output_filename.set(output_filename)
    
    def get_output_path(self):
        """Retourne le chemin complet du fichier de sortie"""
        return os.path.join(self.output_dir.get(), self.output_filename.get())
    
    def try_find_grid_file(self, directory):
        """Tente de trouver un fichier de grille associé au dossier d'images"""
        # Chercher dans le répertoire parent
        parent_dir = os.path.dirname(directory)
        sample_name = os.path.basename(directory)
        
        # Chercher un fichier nommé [sample_name]_grid.txt
        grid_file = os.path.join(parent_dir, f"{sample_name}_grid.txt")
        
        if os.path.exists(grid_file):
            self.grid_file.set(grid_file)
            self.log("Fichier de grille trouvé automatiquement: " + grid_file)
        else:
            # Chercher un fichier nommé grille_images.txt
            grid_file = os.path.join(parent_dir, "grille_images.txt")
            if os.path.exists(grid_file):
                self.grid_file.set(grid_file)
                self.log("Fichier de grille trouvé automatiquement: " + grid_file)
    
    def toggle_overlap_fields(self):
        """Active ou désactive les champs d'overlap manuel selon l'état de l'auto-détection"""
        if self.auto_overlap.get():
            state = "disabled"
        else:
            state = "normal"
            
        self.h_overlap_spinbox.configure(state=state)
        self.v_overlap_spinbox.configure(state=state)
    
    def validate_inputs(self, check_grid=True):
        """Valide les entrées utilisateur avant l'exécution"""
        if not self.input_dir.get() or not os.path.isdir(self.input_dir.get()):
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier d'images valide.")
            return False
        
        if check_grid and (not self.grid_file.get() or not os.path.isfile(self.grid_file.get())):
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier de grille valide.")
            return False
        
        if not self.output_dir.get():
            # Utiliser le répertoire parent du dossier d'images par défaut
            self.output_dir.set(os.path.dirname(self.input_dir.get()))
        
        if not self.output_filename.get():
            # Mettre à jour le nom du fichier de sortie
            self.update_output_filename()
        
        return True
    
    def estimate_remaining_time(self):
        """Estime le temps restant en fonction du nombre d'images traitées"""
        if not self.start_time or self.processed_files == 0 or self.total_files == 0:
            return "Inconnue"
            
        elapsed = time.time() - self.start_time
        rate = self.processed_files / elapsed  # Images par seconde
        remaining_files = self.total_files - self.processed_files
        
        if rate > 0:
            remaining_seconds = remaining_files / rate
            
            # Formater le temps restant
            if remaining_seconds < 60:
                return f"{int(remaining_seconds)} secondes"
            elif remaining_seconds < 3600:
                return f"{int(remaining_seconds / 60)} minutes"
            else:
                hours = int(remaining_seconds / 3600)
                minutes = int((remaining_seconds % 3600) / 60)
                return f"{hours}h {minutes}min"
        else:
            return "Inconnue"
    
    def clear_log(self):
        """Efface le journal des opérations après l'avoir sauvegardé"""
        # Si le journal n'est pas vide, proposer de l'enregistrer
        if self.log_text.get("1.0", tk.END).strip():
            self.save_log("Journal effacé par l'utilisateur")
            
        # Effacer le journal
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def save_log(self, reason=""):
        """Sauvegarde le journal des opérations dans un fichier"""
        # Ne rien faire si le journal est vide
        log_content = self.log_text.get("1.0", tk.END).strip()
        if not log_content:
            return
            
        # Créer un nom de fichier avec la date et l'heure
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        # Déterminer le nom de l'échantillon
        sample_name = self.sample_name.get()
        if not sample_name and self.input_dir.get():
            sample_name = os.path.basename(self.input_dir.get())
        
        # Créer le répertoire de logs s'il n'existe pas
        output_dir = self.output_dir.get()
        if not output_dir:
            output_dir = os.path.dirname(self.input_dir.get() or ".")
            
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Nom du fichier log
        log_filename = f"{sample_name or 'assemblage'}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # Ajouter un en-tête avec la date et l'heure
        header = f"=== Journal des opérations - {now.strftime('%d/%m/%Y %H:%M:%S')} ===\n"
        if reason:
            header += f"Raison: {reason}\n"
        header += "=" * 50 + "\n\n"
        
        # Écrire le contenu du journal
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(header + log_content)
            
        return log_path
    
    def quit_app(self):
        """Ferme l'application après avoir sauvegardé le journal"""
        # Sauvegarder le journal si nécessaire
        if self.log_text.get("1.0", tk.END).strip():
            self.save_log("Fermeture de l'application")
            
        # Fermer l'application
        self.root.destroy()
    
    def log(self, message):
        """Ajoute un message au journal des opérations"""
        self.log_text.config(state=tk.NORMAL)
        # Ajouter l'horodatage au message
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update()
    
    def format_time(self, seconds):
        """Formate un temps en secondes en une chaîne lisible"""
        if seconds < 60:
            return f"{seconds:.1f} secondes"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)} min {int(secs)} sec"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{int(hours)}h {int(minutes)}min {int(secs)}s"
    
    def set_status(self, message):
        """Met à jour la barre de statut"""
        self.status_var.set(message)
        self.root.update()
    
    def set_progress(self, value):
        """Met à jour la barre de progression"""
        self.progress_var.set(value)
        self.root.update()
    
    def run_overlap_detection(self):
        """Lance la détection de chevauchement"""
        if self.processing:
            return
            
        if not self.validate_inputs(check_grid=False):
            return
        
        self.processing = True
        self.start_time = time.time()
        self.set_status("Détection du chevauchement...")
        self.set_progress(0)
        
        # Fonction à exécuter dans un thread séparé
        def detection_thread():
            try:
                # Importer io et redirect_stdout au début de la fonction
                import io
                from contextlib import redirect_stdout
                
                self.log("Démarrage de la détection de chevauchement...")
                self.log(f"Dossier d'images: {self.input_dir.get()}")
                self.log(f"Nombre d'échantillons: {self.num_samples.get()}")
                self.log(f"Format d'image: {'TIFF' if self.use_tiff.get() else 'JPEG'}")
                
                # Exécuter la détection de chevauchement
                file_extensions = [".tif", ".tiff"] if self.use_tiff.get() else [".jpeg", ".jpg"]
                input_dir = self.input_dir.get()
                
                # Trouver le motif du nom de fichier
                files = []
                for ext in file_extensions:
                    files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
                
                if not files:
                    self.log(f"Erreur: Aucun fichier {', '.join(file_extensions)} trouvé dans le dossier.")
                    self.set_status("Erreur: Aucun fichier trouvé")
                    self.processing = False
                    return
                
                # Trouver le nom de l'échantillon
                sample_name = self.sample_name.get()
                if not sample_name:
                    first_file = files[0]
                    sample_name = "_".join(first_file.split("_")[:-1])
                
                pattern = f"{sample_name}_*"
                self.log(f"Motif de recherche: {pattern}")
                
                # Redirection de stdout pour capturer les messages
                f = io.StringIO()
                with redirect_stdout(f):
                    h_overlap, v_overlap = estimate_overlap_from_sample(
                        input_dir,
                        pattern,
                        num_samples=self.num_samples.get(),
                        max_overlap=200
                    )
                
                # Récupérer la sortie et l'afficher dans le journal
                output = f.getvalue()
                self.log(output)
                
                # Mettre à jour les champs d'overlap
                self.h_overlap.set(h_overlap)
                self.v_overlap.set(v_overlap)
                
                # Sauvegarder les résultats
                results = {
                    "sample_name": sample_name,
                    "horizontal_overlap": h_overlap,
                    "vertical_overlap": v_overlap,
                    "num_samples": self.num_samples.get(),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "input_directory": input_dir,
                }
                
                # Créer un fichier de résultats
                output_dir = self.output_dir.get() or os.path.dirname(input_dir)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{sample_name}_overlap.json")
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Calculer le temps d'exécution
                execution_time = time.time() - self.start_time
                execution_time_str = self.format_time(execution_time)
                
                self.log(f"Résultats de détection sauvegardés dans: {output_file}")
                self.log("\nRésultats de la détection:")
                self.log(f"- Chevauchement horizontal: {h_overlap} pixels")
                self.log(f"- Chevauchement vertical: {v_overlap} pixels")
                self.log(f"- Temps d'exécution: {execution_time_str}")
                
                # Sauvegarder le journal
                log_path = self.save_log("Détection d'overlap terminée")
                if log_path:
                    self.log(f"Journal sauvegardé dans: {log_path}")
                
                self.set_status(f"Détection terminée en {execution_time_str}")
                self.set_progress(100)
                
                # Afficher un message de succès
                messagebox.showinfo("Détection terminée", 
                                  f"Chevauchement horizontal: {h_overlap} pixels\n"
                                  f"Chevauchement vertical: {v_overlap} pixels\n\n"
                                  f"Temps d'exécution: {execution_time_str}\n"
                                  f"Résultats sauvegardés dans:\n{output_file}")
            
            except Exception as e:
                self.log(f"Erreur lors de la détection: {str(e)}")
                self.set_status(f"Erreur: {str(e)}")
                messagebox.showerror("Erreur", f"Une erreur est survenue lors de la détection:\n{str(e)}")
            
            finally:
                self.processing = False
        
        # Lancer le thread
        thread = threading.Thread(target=detection_thread)
        thread.daemon = True
        thread.start()
    
    def run_stitching(self):
        """Lance l'assemblage des images"""
        if self.processing:
            return
            
        if not self.validate_inputs():
            return
        
        self.processing = True
        self.start_time = time.time()
        self.set_status("Assemblage des images...")
        self.set_progress(0)
        
        # Fonction à exécuter dans un thread séparé
        def stitching_thread():
            try:
                # Importer io et redirect_stdout ici au début de la fonction
                import io
                from contextlib import redirect_stdout
                
                output_path = self.get_output_path()
                
                self.log("Démarrage de l'assemblage des images...")
                self.log(f"Dossier d'images: {self.input_dir.get()}")
                self.log(f"Fichier de grille: {self.grid_file.get()}")
                self.log(f"Fichier de sortie: {output_path}")
                
                # Si auto-détection demandée, lancer d'abord la détection
                if self.auto_overlap.get():
                    self.log("Détection automatique du chevauchement...")
                    
                    # Trouver le motif du nom de fichier
                    file_extensions = [".tif", ".tiff"] if self.use_tiff.get() else [".jpeg", ".jpg"]
                    input_dir = self.input_dir.get()
                    
                    files = []
                    for ext in file_extensions:
                        files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
                    
                    sample_name = self.sample_name.get()
                    if not sample_name and files:
                        first_file = files[0]
                        sample_name = "_".join(first_file.split("_")[:-1])
                    
                    pattern = f"{sample_name}_*"
                    
                    # Exécuter la détection
                    detection_start = time.time()
                    f = io.StringIO()
                    with redirect_stdout(f):
                        h_overlap, v_overlap = estimate_overlap_from_sample(
                            input_dir,
                            pattern,
                            num_samples=self.num_samples.get(),
                            max_overlap=200
                        )
                    
                    # Récupérer la sortie et l'afficher dans le journal
                    output = f.getvalue()
                    self.log(output)
                    
                    # Mettre à jour les valeurs
                    self.h_overlap.set(h_overlap)
                    self.v_overlap.set(v_overlap)
                    
                    detection_time = time.time() - detection_start
                    self.log(f"Chevauchement détecté: {h_overlap}x{v_overlap} pixels (en {self.format_time(detection_time)})")
                
                # Exécuter l'assemblage
                self.log("\nAssemblage des images...")
                self.log(f"Chevauchement horizontal: {self.h_overlap.get()} pixels")
                self.log(f"Chevauchement vertical: {self.v_overlap.get()} pixels")
                self.log(f"Taille du pixel: {self.pixel_size.get()} mm")
                
                # S'assurer que le répertoire de sortie existe
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Redirection de stdout
                f = io.StringIO()
                with redirect_stdout(f):
                    result = stitch_images(
                        self.input_dir.get(),
                        self.grid_file.get(),
                        output_path,
                        h_overlap=self.h_overlap.get(),
                        v_overlap=self.v_overlap.get(),
                        sample_name=self.sample_name.get() or None,
                        pixel_size_mm=self.pixel_size.get(),
                        use_tiff=self.use_tiff.get()
                    )
                
                # Récupérer la sortie et l'afficher dans le journal
                output = f.getvalue()
                self.log(output)
                
                # Ajouter des métadonnées supplémentaires au fichier JSON
                info_path = os.path.splitext(output_path)[0] + '.json'
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        
                        # Ajouter la date et l'heure
                        info["creation_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Ajouter le chemin vers le dossier source
                        info["source_directory"] = self.input_dir.get()
                        
                        # Ajouter le temps d'exécution
                        execution_time = time.time() - self.start_time
                        info["execution_time_seconds"] = execution_time
                        info["execution_time"] = self.format_time(execution_time)
                        
                        # Sauvegarder les informations mises à jour
                        with open(info_path, 'w') as f:
                            json.dump(info, f, indent=2)
                        
                        # Afficher les informations supplémentaires
                        self.log("\nInformations sur l'image assemblée:")
                        self.log(f"- Échantillon: {info.get('sample_name', 'N/A')}")
                        self.log(f"- Dimensions (pixels): {info.get('image_dimensions', 'N/A')}")
                        self.log(f"- Dimensions physiques: {info.get('physical_dimensions', 'N/A')}")
                        self.log(f"- Images traitées: {info.get('processed_count', 0)}/{info.get('original_count', 0)}")
                        self.log(f"- Temps d'exécution: {info.get('execution_time', 'N/A')}")
                    except Exception as e:
                        self.log(f"Erreur lors de la mise à jour des métadonnées: {str(e)}")
                
                # Sauvegarder le journal
                log_path = self.save_log("Assemblage terminé")
                if log_path:
                    self.log(f"Journal sauvegardé dans: {log_path}")
                
                # Calculer le temps d'exécution total
                execution_time = time.time() - self.start_time
                execution_time_str = self.format_time(execution_time)
                
                self.set_status(f"Assemblage terminé en {execution_time_str}")
                self.set_progress(100)
                
                # Afficher un message de succès
                messagebox.showinfo("Assemblage terminé", 
                                   f"Temps d'exécution: {execution_time_str}\n\n"
                                   f"Image assemblée sauvegardée dans:\n{output_path}")
            
            except Exception as e:
                self.log(f"Erreur lors de l'assemblage: {str(e)}")
                self.set_status(f"Erreur: {str(e)}")
                messagebox.showerror("Erreur", f"Une erreur est survenue lors de l'assemblage:\n{str(e)}")
            
            finally:
                self.processing = False
                
                # Sauvegarder le journal en cas d'erreur
                if self.log_text.get("1.0", tk.END).strip() and "Erreur" in self.log_text.get("1.0", tk.END):
                    self.save_log("Erreur lors de l'assemblage")
        
        # Lancer le thread
        thread = threading.Thread(target=stitching_thread)
        thread.daemon = True
        thread.start()


def main():
    """Fonction principale"""
    root = tk.Tk()
    app = StitcherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()