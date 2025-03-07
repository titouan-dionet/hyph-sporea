"""
Module d'interface graphique pour l'annotation des spores.

Ce module contient une interface graphique Tkinter permettant d'annoter
manuellement les spores dans les images et de générer des annotations
pour l'entraînement du modèle YOLO.
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
from pathlib import Path
import yaml
from skimage import measure

from ..image_processing.preprocessing import enhanced_preprocess_image


class SporeAnnotationTool:
    """
    Interface graphique pour l'annotation des spores dans les images.
    
    Cette classe fournit une interface utilisateur permettant de:
    - Charger un dossier d'images
    - Prétraiter automatiquement les images
    - Annoter manuellement les spores avec leur classe
    - Sauvegarder les annotations au format JSON et YOLO
    
    Attributes:
        root (tk.Tk): Fenêtre principale Tkinter
        current_folder (str): Chemin du dossier d'images courant
        current_image_path (str): Chemin de l'image courante
        current_image (PIL.Image): Image PIL courante pour affichage
        current_image_cv (numpy.ndarray): Image OpenCV courante
        current_mask (numpy.ndarray): Masque binaire courant
        image_files (list): Liste des fichiers d'images dans le dossier
        current_index (int): Index de l'image courante
        zoom_factor (float): Facteur de zoom pour l'affichage
        spore_classes (list): Liste des classes de spores disponibles
        current_class (tk.StringVar): Classe sélectionnée
        annotations (dict): Dictionnaire contenant toutes les annotations
    """
    
    def __init__(self, root):
        """
        Initialise l'outil d'annotation.
        
        Args:
            root (tk.Tk): Fenêtre principale Tkinter
        """
        self.root = root
        self.root.title("Outil d'annotation de spores")
        self.root.geometry("1200x800")
        
        # Variables
        self.current_folder = ""
        self.current_image_path = ""
        self.current_image = None
        self.current_image_cv = None
        self.current_mask = None
        self.image_files = []
        self.current_index = 0
        self.zoom_factor = 1.0
        self.spore_classes = [
            'ALAC',  # Alatospora acuminata
            'LUCU',  # Lunulospora curvula
            'HEST',  # Heliscella stellata
            'HELU',  # Heliscus lugdunensis
            'CLAQ',  # Clavatospora aquatica
            'ARTE',  # Articulospora tetracladia
            'TEMA',  # Tetracladium marchalanium
            'TRSP',  # Tricladium splendens
            'TUAC',  # Tumularia aquatica
            'Debris' # Débris divers
        ]
        self.current_class = tk.StringVar(value=self.spore_classes[0])
        self.annotations = {}
        self.regions = []
        
        # Configuration de l'interface
        self.setup_ui()
    
    def setup_ui(self):
        """
        Configure l'interface utilisateur.
        
        Crée tous les composants de l'interface:
        - Panel de contrôle à gauche
        - Zone d'affichage de l'image à droite
        - Boutons de navigation et d'actions
        """
        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame gauche pour les contrôles
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Bouton pour charger un dossier
        ttk.Button(control_frame, text="Charger un dossier", command=self.load_folder).pack(fill=tk.X, pady=5)
        
        # Liste des classes de spores
        ttk.Label(control_frame, text="Classe:").pack(anchor=tk.W, pady=(10, 0))
        for spore_class in self.spore_classes:
            ttk.Radiobutton(
                control_frame, 
                text=spore_class, 
                variable=self.current_class, 
                value=spore_class
            ).pack(anchor=tk.W)
        
        # Boutons de navigation
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        ttk.Button(nav_frame, text="←", command=self.prev_image).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text="→", command=self.next_image).pack(side=tk.LEFT, expand=True)
        
        # Informations sur l'image
        self.info_label = ttk.Label(control_frame, text="")
        self.info_label.pack(fill=tk.X, pady=5)
        
        # Nombre d'annotations
        self.annotations_label = ttk.Label(control_frame, text="Annotations: 0")
        self.annotations_label.pack(fill=tk.X, pady=5)
        
        # Boutons d'enregistrement
        ttk.Button(control_frame, text="Enregistrer", command=self.save_annotations).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Convertir en YOLO", command=self.convert_to_yolo_format).pack(fill=tk.X, pady=5)
        
        # Frame droite pour l'affichage de l'image
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas pour l'image
        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Liaison des événements
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_wheel)  # Pour Windows
        self.canvas.bind("<Button-4>", self.on_wheel)    # Pour Linux (scroll up)
        self.canvas.bind("<Button-5>", self.on_wheel)    # Pour Linux (scroll down)
    
    def load_folder(self):
        """
        Charge un dossier d'images.
        
        Ouvre une boîte de dialogue pour sélectionner un dossier, puis charge
        toutes les images JPEG, JPG, PNG, TIFF et TIF présentes dans ce dossier.
        """
        folder = filedialog.askdirectory(title="Sélectionner un dossier d'images")
        if folder:
            self.current_folder = folder
            self.image_files = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.jpeg', '.jpg', '.png', '.tiff', '.tif'))
            ]
            self.image_files.sort()
            self.current_index = 0
            
            if not self.image_files:
                messagebox.showinfo("Information", "Aucune image trouvée dans ce dossier.")
                return
                
            self.load_current_image()
            
            # Charger les annotations existantes si elles existent
            annotation_file = os.path.join(folder, "annotations.json")
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r') as f:
                        self.annotations = json.load(f)
                    messagebox.showinfo("Information", f"Annotations chargées depuis {annotation_file}")
                except Exception as e:
                    messagebox.showerror("Erreur", f"Impossible de charger les annotations: {str(e)}")
    
    def load_current_image(self):
        """
        Charge et affiche l'image courante.
        
        Charge l'image courante, applique le prétraitement, détecte les objets,
        et affiche l'image avec les annotations existantes.
        """
        if 0 <= self.current_index < len(self.image_files):
            self.current_image_path = self.image_files[self.current_index]
            
            try:
                # Charger l'image
                self.current_image_cv = cv2.imread(self.current_image_path)
                
                if self.current_image_cv is None:
                    messagebox.showerror("Erreur", f"Impossible de lire l'image: {self.current_image_path}")
                    return
                
                # Prétraitement pour détecter les objets
                final_image, self.current_mask = enhanced_preprocess_image(
                    self.current_image_path, 
                    intensity='very_light'  # Utiliser le prétraitement léger
                )
                
                # Extraction des composantes connexes
                labeled_mask = measure.label(self.current_mask)
                self.regions = measure.regionprops(labeled_mask)
                
                # Créer une image pour l'affichage
                img_with_contours = self.current_image_cv.copy()
                
                # Dessiner les contours des objets détectés en BLEU (BGR: B=255, G=0, R=0)
                contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_with_contours, contours,  -1, (255, 0, 0), 1)
                
                # Afficher les annotations existantes
                img_name = os.path.basename(self.current_image_path)
                if img_name in self.annotations:
                    for ann in self.annotations[img_name]:
                        x, y = ann["position"]
                        label = ann["class"]
                        color = self.get_color_for_class(label)
                        cv2.circle(img_with_contours, (x, y), 10, color, -1)
                        cv2.putText(img_with_contours, label[:2], (x-10, y+5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Mettre à jour le compteur d'annotations
                self.update_annotations_count(img_name)
                
                # Convertir pour affichage
                img_rgb = cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(img_rgb)
                self.display_image()
                
                # Mettre à jour les informations
                img_name = os.path.basename(self.current_image_path)
                self.info_label.config(text=f"Image: {self.current_index+1}/{len(self.image_files)}\n{img_name}")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du chargement de l'image: {str(e)}")
    
    def display_image(self):
        """
        Affiche l'image courante dans le canvas.
        
        Redimensionne l'image selon le facteur de zoom et l'affiche dans le canvas.
        """
        if self.current_image:
            # Redimensionner l'image selon le facteur de zoom
            width, height = self.current_image.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            img_resized = self.current_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convertir pour tkinter
            self.tk_image = ImageTk.PhotoImage(img_resized)
            
            # Afficher sur le canvas
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
    
    def on_click(self, event):
        """
        Gère les clics sur l'image.
        
        Détecte si un clic est sur un objet, ajoute ou met à jour l'annotation correspondante.
        
        Args:
            event: Événement de clic de souris
        """
        if self.current_image_cv is not None:
            # Convertir les coordonnées du clic selon le zoom
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Ajuster selon le facteur de zoom
            img_x = int(canvas_x / self.zoom_factor)
            img_y = int(canvas_y / self.zoom_factor)
            
            # Vérifier si le clic est sur un objet
            if 0 <= img_x < self.current_image_cv.shape[1] and 0 <= img_y < self.current_image_cv.shape[0]:
                # Trouver le plus proche centroïde
                closest_region = None
                min_distance = float('inf')
                
                for region in self.regions:
                    y, x = region.centroid
                    distance = np.sqrt((x - img_x)**2 + (y - img_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_region = region
                
                # Si un objet est trouvé à proximité
                if closest_region and min_distance < 50:  # Seuil de distance à ajuster
                    y, x = closest_region.centroid
                    
                    # Ajouter ou mettre à jour l'annotation
                    img_name = os.path.basename(self.current_image_path)
                    if img_name not in self.annotations:
                        self.annotations[img_name] = []
                    
                    # Vérifier si cet objet est déjà annoté
                    already_annotated = False
                    for i, ann in enumerate(self.annotations[img_name]):
                        ann_x, ann_y = ann["position"]
                        if np.sqrt((ann_x - x)**2 + (ann_y - y)**2) < 10:
                            # Mettre à jour la classe
                            self.annotations[img_name][i]["class"] = self.current_class.get()
                            already_annotated = True
                            break
                    
                    if not already_annotated:
                        # Nouvelle annotation
                        self.annotations[img_name].append({
                            "position": [int(x), int(y)],
                            "class": self.current_class.get(),
                            "bbox": [
                                int(closest_region.bbox[1]),
                                int(closest_region.bbox[0]),
                                int(closest_region.bbox[3]),
                                int(closest_region.bbox[2])
                            ]
                        })
                    
                    # Mettre à jour le compteur d'annotations
                    self.update_annotations_count(img_name)
                    
                    # Recharger l'image pour afficher l'annotation
                    self.load_current_image()
    
    def update_annotations_count(self, img_name):
        """
        Met à jour le compteur d'annotations pour l'image courante.
        
        Args:
            img_name (str): Nom de l'image courante
        """
        count = len(self.annotations.get(img_name, []))
        self.annotations_label.config(text=f"Annotations: {count}")
    
    def get_color_for_class(self, class_name):
        """
        Renvoie une couleur spécifique pour une classe donnée.
        
        Args:
            class_name (str): Nom de la classe
        
        Returns:
            tuple: Couleur au format BGR (Blue, Green, Red)
        """
        # Attribuer une couleur différente à chaque classe
        colors = {
            "ALAC": (255, 0, 0),    # Rouge
            "ANFI": (0, 255, 0),    # Vert
            "CLAQ": (0, 0, 255),    # Bleu
            "TFLU": (255, 255, 0),  # Jaune
            "TELE": (255, 0, 255),  # Magenta
            "Debris": (128, 128, 128)  # Gris
        }
        return colors.get(class_name, (0, 0, 0))
    
    def on_wheel(self, event):
        """
        Gère le zoom avec la molette de la souris.
        
        Args:
            event: Événement de la molette de souris
        """
        # Zoom avec la molette de la souris
        if event.num == 4 or event.delta > 0:  # Zoom in
            self.zoom_factor *= 1.1
        elif event.num == 5 or event.delta < 0:  # Zoom out
            self.zoom_factor *= 0.9
        
        self.display_image()
    
    def next_image(self):
        """
        Passe à l'image suivante.
        """
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def prev_image(self):
        """
        Passe à l'image précédente.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def save_annotations(self):
        """
        Sauvegarde les annotations au format JSON.
        """
        if self.current_folder and self.annotations:
            # Sauvegarder au format JSON
            annotation_file = os.path.join(self.current_folder, "annotations.json")
            try:
                with open(annotation_file, 'w') as f:
                    json.dump(self.annotations, f, indent=2)
                messagebox.showinfo("Information", f"Annotations sauvegardées dans {annotation_file}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder les annotations: {str(e)}")
        else:
            messagebox.showwarning("Attention", "Aucune annotation à sauvegarder ou dossier non sélectionné.")
    
    def convert_to_yolo_format(self):
        """
        Convertit les annotations au format YOLO.
        
        Crée un dossier "yolo_annotations" contenant:
        - Un fichier de classes
        - Un fichier .txt pour chaque image annotée
        - Une copie des images annotées
        - Un fichier data.yaml pour l'entraînement YOLO
        """
        if not self.current_folder or not self.annotations:
            messagebox.showwarning("Attention", "Aucune annotation à convertir ou dossier non sélectionné.")
            return
        
        try:
            # Créer un dossier pour les annotations YOLO
            yolo_dir = os.path.join(self.current_folder, "yolo_annotations")
            os.makedirs(yolo_dir, exist_ok=True)
            
            # Créer le fichier de classes
            with open(os.path.join(yolo_dir, "classes.txt"), 'w') as f:
                for cls in self.spore_classes:
                    f.write(f"{cls}\n")
            
            # Images annotées
            annotated_images = []
            
            # Parcourir toutes les annotations
            for img_name, annotations in self.annotations.items():
                if not annotations:  # Ignorer les images sans annotations
                    continue
                    
                # Charger l'image pour obtenir ses dimensions
                img_path = os.path.join(self.current_folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Créer un fichier d'annotation YOLO pour cette image
                txt_name = os.path.splitext(img_name)[0] + '.txt'
                txt_path = os.path.join(yolo_dir, txt_name)
                
                with open(txt_path, 'w') as f:
                    for ann in annotations:
                        class_name = ann["class"]
                        class_id = self.spore_classes.index(class_name)
                        
                        # Convertir les coordonnées bbox en format YOLO
                        x_min, y_min, x_max, y_max = ann["bbox"]
                        
                        # Calculer les valeurs normalisées pour YOLO
                        x_center = ((x_min + x_max) / 2) / img_width
                        y_center = ((y_min + y_max) / 2) / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        # Écrire la ligne d'annotation
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # Copier l'image dans le dossier YOLO
                dest_img_path = os.path.join(yolo_dir, img_name)
                cv2.imwrite(dest_img_path, img)
                
                # Ajouter à la liste des images annotées
                annotated_images.append(img_name)
            
            # Créer le fichier data.yaml pour YOLOv8
            data_yaml = {
                'path': os.path.abspath(yolo_dir),
                'train': 'train',  # Sera configuré par l'utilisateur
                'val': 'val',      # Sera configuré par l'utilisateur
                'nc': len(self.spore_classes),
                'names': self.spore_classes
            }
            
            with open(os.path.join(yolo_dir, 'data.yaml'), 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            
            messagebox.showinfo("Information", 
                              f"Conversion en format YOLO terminée.\n"
                              f"{len(annotated_images)} images annotées.\n"
                              f"Fichiers sauvegardés dans {yolo_dir}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la conversion en format YOLO: {str(e)}")


def run_annotation_tool():
    """
    Lance l'outil d'annotation comme application standalone.
    
    Returns:
        None
    
    Example:
        >>> run_annotation_tool()
    """
    root = tk.Tk()
    app = SporeAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    run_annotation_tool()
