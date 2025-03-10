# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:12:43 2025

@author: Titouan Dionet
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path

# from Python.core import HyphSporeaProcessor
from Python.core.utils import get_sample_info_from_path

class HyphSporeaGUI:
    """
    Interface graphique principale pour le projet HYPH-SPOREA.
    
    Cette interface permet à l'utilisateur de:
    - Sélectionner la dernière version du modèle ou une version spécifique
    - Fournir un dossier d'images à traiter
    - Configurer les options de traitement
    - Lancer le traitement
    - Visualiser les résultats
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("HYPH-SPOREA - Analyse de spores d'hyphomycètes")
        self.root.geometry("900x600")
        
        from Python.core import HyphSporeaProcessor
        
        # Définir le projet
        self.project_dir = Path.cwd()
        self.processor = HyphSporeaProcessor(self.project_dir)
        
        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(self.project_dir / "outputs"))
        self.model_version = tk.StringVar(value="latest")
        self.save_images = tk.BooleanVar(value=True)
        self.convert_images = tk.BooleanVar(value=True)
        
        # Configuration de l'interface
        self.setup_ui()
    
    def setup_ui(self):
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, text="HYPH-SPOREA", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Dossiers
        folder_frame = ttk.LabelFrame(main_frame, text="Dossiers", padding="10")
        folder_frame.pack(fill=tk.X, pady=5)
        
        # Dossier d'entrée
        input_frame = ttk.Frame(folder_frame)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Dossier d'entrée:").pack(side=tk.LEFT)
        ttk.Entry(input_frame, textvariable=self.input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_frame, text="Parcourir", command=self.browse_input).pack(side=tk.LEFT)
        
        # Dossier de sortie
        output_frame = ttk.Frame(folder_frame)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Dossier de sortie:").pack(side=tk.LEFT)
        ttk.Entry(output_frame, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(output_frame, text="Parcourir", command=self.browse_output).pack(side=tk.LEFT)
        
        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        
        # Version du modèle
        model_frame = ttk.Frame(options_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Version du modèle:").pack(side=tk.LEFT)
        ttk.Entry(model_frame, textvariable=self.model_version).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(model_frame, text="Parcourir", command=self.browse_model).pack(side=tk.LEFT)
        
        # Options supplémentaires
        option_checks = ttk.Frame(options_frame)
        option_checks.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(option_checks, text="Enregistrer les images traitées", variable=self.save_images).pack(anchor=tk.W)
        ttk.Checkbutton(option_checks, text="Convertir automatiquement les formats d'image", variable=self.convert_images).pack(anchor=tk.W)
        
        # Boutons d'action
        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill=tk.X, pady=10)
        ttk.Button(actions_frame, text="Lancer le traitement", command=self.run_pipeline).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Ouvrir l'outil d'annotation", command=self.open_annotation_tool).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions_frame, text="Voir les résultats", command=self.view_results).pack(side=tk.LEFT, padx=5)
        
        # Zone de log
        log_frame = ttk.LabelFrame(main_frame, text="Journal d'exécution", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Barre de progression
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=5)
        
        # Barre d'état
        self.status_var = tk.StringVar(value="Prêt")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
    
    def browse_input(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier d'entrée"""
        folder = filedialog.askdirectory(title="Sélectionner le dossier d'entrée")
        if folder:
            self.input_dir.set(folder)
    
    def browse_output(self):
        """Ouvre une boîte de dialogue pour sélectionner le dossier de sortie"""
        folder = filedialog.askdirectory(title="Sélectionner le dossier de sortie")
        if folder:
            self.output_dir.set(folder)
    
    def browse_model(self):
        """Ouvre une boîte de dialogue pour sélectionner un modèle spécifique"""
        file = filedialog.askopenfilename(
            title="Sélectionner un modèle",
            filetypes=[("Modèles PyTorch", "*.pt"), ("Modèles Keras", "*.h5"), ("Tous les fichiers", "*.*")]
        )
        if file:
            self.model_version.set(file)
    
    def log(self, message):
        """Ajoute un message dans la zone de log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def run_pipeline(self):
        """Lance le pipeline de traitement"""
        if not self.input_dir.get():
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier d'entrée.")
            return
        
        # Vérifier si les dossiers existent
        input_dir = Path(self.input_dir.get())
        output_dir = Path(self.output_dir.get())
        
        if not input_dir.exists():
            messagebox.showerror("Erreur", f"Le dossier d'entrée {input_dir} n'existe pas.")
            return
        
        # Créer le dossier de sortie s'il n'existe pas
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Mettre à jour l'interface
        self.status_var.set("Traitement en cours...")
        self.progress.start()
        
        try:
            # Exécuter le pipeline
            self.log(f"Début du traitement avec {self.processor.__class__.__name__}...")
            self.log(f"Dossier d'entrée: {input_dir}")
            self.log(f"Dossier de sortie: {output_dir}")
            
            # Configurer le processeur
            self.processor.project_root = self.project_dir
            self.processor.output_dir = output_dir
            
            # Déterminer le modèle à utiliser
            model_version = self.model_version.get()
            model_path = None
            
            if model_version == "latest":
                self.log("Utilisation de la dernière version du modèle...")
                model_path = get_latest_model_version(self.processor.models_dir, 'yolo')
            else:
                model_path = model_version
            
            if not model_path:
                self.log("Aucun modèle trouvé. Téléchargement du modèle YOLO par défaut...")
                model_path = str(self.project_root / "data" / "blank_models" / "yolo11m.pt")
            
            # Traitement
            if self.convert_images.get():
                self.log("Conversion des images...")
                jpeg_dir = self.processor.convert_tiff_to_jpeg(str(input_dir))
                input_dir = jpeg_dir
            
            self.log("Prétraitement des images...")
            preprocessed_dir = self.processor.preprocess_dataset(str(input_dir))
            
            self.log(f"Traitement des échantillons avec le modèle {os.path.basename(model_path)}...")
            for sample_dir in Path(input_dir).glob("*"):
                if sample_dir.is_dir():
                    sample_name = sample_dir.name
                    self.log(f"Traitement de l'échantillon {sample_name}...")
                    
                    sample_info = get_sample_info_from_path(sample_dir)
                    sample_output = output_dir / f"processed_{sample_name}"
                    
                    self.processor.process_sample(
                        str(sample_dir),
                        model_path,
                        str(sample_output),
                        sample_info
                    )
            
            self.log("Comparaison des échantillons...")
            comparison_dir = self.processor.compare_samples()
            
            self.log(f"Traitement terminé. Résultats dans {comparison_dir}")
            messagebox.showinfo("Terminé", "Le traitement a été effectué avec succès.")
            
        except Exception as e:
            self.log(f"ERREUR: {str(e)}")
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")
            
        finally:
            # Restaurer l'interface
            self.progress.stop()
            self.status_var.set("Prêt")
    
    def open_annotation_tool(self):
        """Ouvre l'outil d'annotation"""
        from Python.gui.annotation_tool import run_annotation_tool
        self.log("Ouverture de l'outil d'annotation...")
        run_annotation_tool()
    
    def view_results(self):
        """Ouvre l'explorateur de fichiers sur le dossier de résultats"""
        output_dir = self.output_dir.get()
        if not output_dir:
            messagebox.showerror("Erreur", "Aucun dossier de sortie spécifié.")
            return
        
        if not os.path.exists(output_dir):
            messagebox.showerror("Erreur", f"Le dossier {output_dir} n'existe pas.")
            return
        
        # Ouvrir l'explorateur de fichiers
        os.startfile(output_dir)  # Pour Windows


def run_gui():
    """Lance l'interface graphique principale"""
    root = tk.Tk()
    app = HyphSporeaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()