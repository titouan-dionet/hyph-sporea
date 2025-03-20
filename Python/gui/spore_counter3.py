import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import cv2
import datetime
import math

class SporeCounterApp:
    """
    Application pour compter et classifier des spores sur des images de filtres circulaires.
    
    Fonctionnalités:
    - Charger une image de filtre
    - Diviser automatiquement les grandes images en régions
    - Naviguer entre les différentes parties de l'image
    - Zoomer et se déplacer dans l'image
    - Sélectionner une souche de champignon
    - Ajouter des points pour marquer les spores
    - Supprimer des points avec un clic droit
    - Afficher le comptage en temps réel
    - Exporter les données en CSV
    """
    
    def __init__(self, root):
        """Initialise l'application"""
        self.root = root
        self.root.title("Compteur de Spores")
        
        # Ajuster la taille de la fenêtre à l'écran
        self.adjust_window_to_screen()
        
        # Définir les souches de champignons
        self.fungus_strains = [
            'ALAC',  # Alatospora acuminata
            'LUCU',  # Lunulospora curvula
            'HEST',  # Heliscella stellata
            'HELU',  # Heliscus lugdunensis
            'CLAQ',  # Clavatospora aquatica
            'ARTE',  # Articulospora tetracladia
            'TEMA',  # Tetracladium marchalanium
            'TRSP',  # Tricladium splendens
            'TUAC',  # Tumularia aquatica
            'Autre'  # Autres types
        ]
        
        # Définir les couleurs pour chaque souche
        self.strain_colors = {
            'ALAC': (255, 0, 0),     # Rouge
            'LUCU': (0, 255, 0),     # Vert
            'HEST': (0, 0, 255),     # Bleu
            'HELU': (255, 255, 0),   # Jaune
            'CLAQ': (255, 0, 255),   # Magenta
            'ARTE': (0, 255, 255),   # Cyan
            'TEMA': (128, 0, 0),     # Marron
            'TRSP': (0, 128, 0),     # Vert foncé
            'TUAC': (0, 0, 128),     # Bleu foncé
            'Autre': (128, 128, 128) # Gris
        }
        
        # Variables
        self.current_image_path = ""
        self.current_image = None
        self.current_image_cv = None
        self.original_image = None  # Image originale complète
        self.current_strain = tk.StringVar(value=self.fungus_strains[0])
        self.spore_points = {}  # {strain: [(x, y), ...]}
        self.total_count = 0
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        
        # Variables pour la gestion des grandes images
        self.is_large_image = False
        self.grid_size = 4  # 4x4 grid (16 parts)
        self.current_grid_x = 0
        self.current_grid_y = 0
        self.grid_rows = 4
        self.grid_cols = 4
        self.tile_width = 0
        self.tile_height = 0
        self.grid_position = tk.StringVar(value="Partie: 1/16")
        
        # Configurer l'interface
        self.setup_ui()
        
    def adjust_window_to_screen(self):
        """Ajuste la taille de la fenêtre à l'écran actuel"""
        # Obtenir les dimensions de l'écran
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Définir une taille qui laisse de la marge pour la barre des tâches
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.85)
        
        # Positionner la fenêtre au centre de l'écran
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        
        # Définir la géométrie de la fenêtre
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
        # Adapter les polices selon la résolution
        base_size = min(screen_width, screen_height) // 100
        self.font_small = ("Arial", max(8, base_size))
        self.font_normal = ("Arial", max(10, base_size + 1))
        self.font_large = ("Arial", max(12, base_size + 2), "bold")
        
        # Option pour permettre le redimensionnement
        self.root.resizable(True, True)
        
        # Définir des tailles minimales
        self.root.minsize(800, 600)
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # PanedWindow principal
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame gauche pour les contrôles avec défilement
        control_outer_frame = ttk.Frame(main_paned)
        control_canvas = tk.Canvas(control_outer_frame, borderwidth=0)
        control_scrollbar = ttk.Scrollbar(control_outer_frame, orient="vertical", command=control_canvas.yview)
        self.control_frame = ttk.Frame(control_canvas)
        
        # Configurer le défilement
        self.control_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        # Placement des widgets pour le panneau de contrôle
        control_scrollbar.pack(side="right", fill="y")
        control_canvas.pack(side="left", fill="both", expand=True)
        
        # Frame droite pour l'affichage de l'image
        display_frame = ttk.Frame(main_paned)
        
        # Ajouter les frames au PanedWindow
        main_paned.add(control_outer_frame, weight=1)
        main_paned.add(display_frame, weight=4)
        
        # Bouton pour charger une image
        self.load_button = ttk.Button(self.control_frame, text="Charger une image", command=self.load_image)
        self.load_button.pack(fill=tk.X, pady=5)
        
        # Navigation entre les parties de l'image (créée mais pas affichée)
        self.grid_nav_frame = ttk.LabelFrame(self.control_frame, text="Navigation d'image")
        
        # Indicateur de position actuelle
        ttk.Label(self.grid_nav_frame, textvariable=self.grid_position).pack(anchor=tk.CENTER, pady=5)
    
        # Créer une grille de boutons de navigation
        grid_buttons_frame = ttk.Frame(self.grid_nav_frame)
        grid_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Boutons directionnels
        ttk.Button(grid_buttons_frame, text="↑", width=3, 
                  command=lambda: self.navigate_grid(0, -1)).grid(row=0, column=1)
        ttk.Button(grid_buttons_frame, text="←", width=3, 
                  command=lambda: self.navigate_grid(-1, 0)).grid(row=1, column=0)
        ttk.Button(grid_buttons_frame, text="→", width=3, 
                  command=lambda: self.navigate_grid(1, 0)).grid(row=1, column=2)
        ttk.Button(grid_buttons_frame, text="↓", width=3, 
                  command=lambda: self.navigate_grid(0, 1)).grid(row=2, column=1)
        
        # Sélection directe de la partie
        grid_select_frame = ttk.Frame(self.grid_nav_frame)
        grid_select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(grid_select_frame, text="Aller à:").pack(side=tk.LEFT, padx=5)
        
        # Créer une grille 4x4 de boutons numériques
        grid_num_frame = ttk.Frame(self.grid_nav_frame)
        grid_num_frame.pack(fill=tk.X, pady=5)
        
        for row in range(4):
            for col in range(4):
                num = row * 4 + col + 1
                ttk.Button(grid_num_frame, text=str(num), width=3,
                          command=lambda r=row, c=col: self.select_grid_tile(r, c)).grid(
                              row=row, column=col, padx=2, pady=2)
        
        # Sélection de la souche
        strain_frame = ttk.LabelFrame(self.control_frame, text="Souche de champignon")
        strain_frame.pack(fill=tk.X, pady=5, padx=5)
        
        for strain in self.fungus_strains:
            rb = ttk.Radiobutton(
                strain_frame, 
                text=strain, 
                variable=self.current_strain, 
                value=strain,
                style="TRadiobutton"
            )
            rb.pack(anchor=tk.W, padx=5)
        
        # Compteur de spores par souche
        count_frame = ttk.LabelFrame(self.control_frame, text="Comptage des spores")
        count_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.counts_by_strain = {}
        for strain in self.fungus_strains:
            self.counts_by_strain[strain] = tk.StringVar(value=f"{strain}: 0")
            ttk.Label(count_frame, textvariable=self.counts_by_strain[strain], font=self.font_small).pack(anchor=tk.W, padx=5)
        
        # Compteur total
        self.total_count_var = tk.StringVar(value="Total: 0")
        ttk.Label(self.control_frame, textvariable=self.total_count_var, font=self.font_large).pack(pady=10)
        
        # Instructions
        instruction_frame = ttk.LabelFrame(self.control_frame, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=5, padx=5)
        
        instructions = [
            "- Clic gauche: ajouter un point",
            "- Clic droit: supprimer un point",
            "- Molette: zoom avant/arrière",
            "- Clic molette + déplacer: naviguer",
            "- Double-clic: réinitialiser zoom",
            "- Flèches: naviguer entre les parties"
        ]
        
        for instr in instructions:
            ttk.Label(instruction_frame, text=instr, wraplength=230, font=self.font_small).pack(anchor=tk.W, padx=5, pady=2)
        
        # Boutons d'action
        ttk.Button(self.control_frame, text="Exporter CSV", command=self.export_csv).pack(fill=tk.X, pady=5)
        ttk.Button(self.control_frame, text="Réinitialiser points", command=self.reset_points).pack(fill=tk.X, pady=5)
        
        # Canvas pour l'image avec barres de défilement
        self.canvas_frame = ttk.Frame(display_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas et scrollbars
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="crosshair")
        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Placement du canvas et des scrollbars
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Liaison des événements
        self.canvas.bind("<ButtonPress-1>", self.add_point)
        self.canvas.bind("<ButtonPress-3>", self.remove_point)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Pour Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)    # Pour Linux (molette vers le haut)
        self.canvas.bind("<Button-5>", self.on_mousewheel)    # Pour Linux (molette vers le bas)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)   # Molette enfoncée
        self.canvas.bind("<B2-Motion>", self.pan_image)       # Déplacement avec molette enfoncée
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)  # Relâchement de la molette
        self.canvas.bind("<Double-Button-1>", self.reset_zoom)  # Double-clic pour réinitialiser le zoom
        
        # Raccourcis clavier pour naviguer
        self.root.bind("<Left>", lambda e: self.navigate_grid(-1, 0))
        self.root.bind("<Right>", lambda e: self.navigate_grid(1, 0))
        self.root.bind("<Up>", lambda e: self.navigate_grid(0, -1))
        self.root.bind("<Down>", lambda e: self.navigate_grid(0, 1))
        
        # Statut
        self.status_var = tk.StringVar(value="Prêt. Chargez une image pour commencer.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def toggle_fullscreen(self):
        """Basculer entre le mode plein écran et normal"""
        is_fullscreen = not self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', is_fullscreen)
        
        # Option : créer un raccourci clavier pour quitter le plein écran
        if is_fullscreen:
            self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
    
    def load_image(self):
        """Charge une image et l'affiche dans le canvas avec gestion des grandes images"""
        # Ouvrir une boîte de dialogue pour sélectionner un fichier image
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Désactiver la limite de décompression de PIL
            # IMPORTANT: N'utilisez ceci que pour des images de confiance
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # en MB
            self.status_var.set(f"Chargement de l'image ({file_size:.2f} Mo)...")
            self.root.update()
            
            # Chargement d'image avec PIL
            pil_img = Image.open(file_path)
            width, height = pil_img.size
            
            # Conserver l'image originale pour la référence lors de l'export
            self.original_image = pil_img
            
            # Déterminer si c'est une grande image
            pixel_count = width * height
            max_pixels = 40000000  # 40 millions de pixels (ajuster selon votre machine)
            
            # Calcul de la grille
            self.current_grid_x = 0
            self.current_grid_y = 0
            
            if pixel_count > max_pixels:
                # C'est une grande image, on va la diviser en parties
                self.is_large_image = True
                
                # Calculer les dimensions des tuiles
                self.tile_width = width // self.grid_cols
                self.tile_height = height // self.grid_rows
                
                # Afficher le panneau de navigation
                self.grid_nav_frame.pack(fill=tk.X, pady=5, padx=5, after=self.load_button)
                
                # Mettre à jour l'étiquette de position
                self.update_grid_position()
                
                # Extraire la première tuile (en haut à gauche)
                self.extract_and_display_tile()
                
                # Créer/mettre à jour la miniature
                self.update_minimap()
                
                # Informer l'utilisateur
                messagebox.showinfo(
                    "Grande image détectée", 
                    f"L'image est très grande ({width}x{height}, {pixel_count} pixels).\n"
                    f"Elle a été divisée en {self.grid_rows}x{self.grid_cols} parties pour faciliter le traitement.\n"
                    f"Utilisez les boutons ou les flèches du clavier pour naviguer entre les parties."
                )
            else:
                # Pour les images plus petites, pas besoin de les diviser
                self.is_large_image = False
                
                # Cacher le panneau de navigation s'il était visible
                self.grid_nav_frame.pack_forget()
                
                # Utiliser directement l'image PIL sans passer par OpenCV pour préserver les couleurs
                self.current_image = pil_img
                
                # Convertir pour OpenCV si nécessaire
                img_np = np.array(pil_img)
                
                # Si l'image est en niveaux de gris, la convertir en BGR pour OpenCV
                if len(img_np.shape) == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                # Si l'image est en RGB, la convertir en BGR pour OpenCV
                elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # Si l'image a un canal alpha, le supprimer
                elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                
                self.current_image_cv = img_np
                
                # Afficher l'image
                self.display_image()
            
            # Mettre à jour le chemin de l'image
            self.current_image_path = file_path
            
            # Réinitialiser les points et le zoom
            self.reset_points()
            self.zoom_factor = 1.0
            
            # Mettre à jour le statut
            file_name = os.path.basename(file_path)
            self.status_var.set(f"Image chargée: {file_name} ({width}x{height}, {file_size:.2f} Mo)")
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # Affiche la trace de l'erreur dans la console
            messagebox.showerror("Erreur", f"Erreur lors du chargement de l'image: {str(e)}")
    
    def update_minimap(self):
        """Affiche une miniature de l'image complète avec la région active mise en évidence"""
        if not self.is_large_image or self.original_image is None:
            return
            
        # Créer un frame pour la miniature si nécessaire
        if not hasattr(self, 'minimap_frame'):
            self.minimap_frame = ttk.LabelFrame(self.grid_nav_frame, text="Position")
            self.minimap_frame.pack(fill=tk.X, pady=5, padx=5)
            self.minimap_canvas = tk.Canvas(self.minimap_frame, width=200, height=200, bg="black")
            self.minimap_canvas.pack(padx=5, pady=5)
        
        # Redimensionner l'image originale pour la miniature
        thumbnail_size = 200
        aspect_ratio = self.original_image.width / self.original_image.height
        
        if aspect_ratio > 1:
            thumb_width = thumbnail_size
            thumb_height = int(thumbnail_size / aspect_ratio)
        else:
            thumb_width = int(thumbnail_size * aspect_ratio)
            thumb_height = thumbnail_size
        
        thumbnail = self.original_image.copy()
        thumbnail.thumbnail((thumb_width, thumb_height))
        
        # Convertir en PhotoImage pour Tkinter
        self.thumbnail_image = ImageTk.PhotoImage(thumbnail)
        
        # Afficher la miniature
        self.minimap_canvas.delete("all")
        self.minimap_canvas.create_image(
            (thumbnail_size - thumb_width) // 2, 
            (thumbnail_size - thumb_height) // 2, 
            anchor=tk.NW, 
            image=self.thumbnail_image
        )
        
        # Calculer et dessiner le rectangle pour la région active
        x_ratio = thumb_width / self.original_image.width
        y_ratio = thumb_height / self.original_image.height
        
        x1 = self.current_grid_x * self.tile_width * x_ratio + (thumbnail_size - thumb_width) // 2
        y1 = self.current_grid_y * self.tile_height * y_ratio + (thumbnail_size - thumb_height) // 2
        x2 = min(x1 + self.tile_width * x_ratio, (thumbnail_size - thumb_width) // 2 + thumb_width)
        y2 = min(y1 + self.tile_height * y_ratio, (thumbnail_size - thumb_height) // 2 + thumb_height)
        
        self.minimap_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    
    def extract_and_display_tile(self):
        """Extrait et affiche la tuile actuelle de la grande image"""
        if not self.is_large_image or self.original_image is None:
            return
            
        # Calculer les coordonnées de la tuile
        x_start = self.current_grid_x * self.tile_width
        y_start = self.current_grid_y * self.tile_height
        x_end = min(x_start + self.tile_width, self.original_image.width)
        y_end = min(y_start + self.tile_height, self.original_image.height)
        
        # Extraire la tuile directement de l'image PIL
        tile = self.original_image.crop((x_start, y_start, x_end, y_end))
        
        # Utiliser directement l'image PIL pour l'affichage
        self.current_image = tile
        
        # Convertir pour OpenCV si nécessaire
        img_np = np.array(tile)
        
        # Si l'image est en niveaux de gris, la convertir en BGR pour OpenCV
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        # Si l'image est en RGB, la convertir en BGR pour OpenCV
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Si l'image a un canal alpha, le supprimer
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        
        self.current_image_cv = img_np
        
        # Afficher la tuile
        self.display_image()
        
        # Mettre à jour l'étiquette de position et la miniature
        self.update_grid_position()
        self.update_minimap()
    
    def navigate_grid(self, dx, dy):
        """Navigue dans la grille de tuiles dans la direction spécifiée"""
        if not self.is_large_image:
            return
            
        # Calculer les nouvelles coordonnées de grille
        new_x = max(0, min(self.grid_cols - 1, self.current_grid_x + dx))
        new_y = max(0, min(self.grid_rows - 1, self.current_grid_y + dy))
        
        # Vérifier si les coordonnées ont changé
        if new_x != self.current_grid_x or new_y != self.current_grid_y:
            self.current_grid_x = new_x
            self.current_grid_y = new_y
            
            # Extraire et afficher la nouvelle tuile
            self.extract_and_display_tile()
    
    def select_grid_tile(self, row, col):
        """Sélectionne directement une tuile spécifique dans la grille"""
        if not self.is_large_image:
            return
            
        # Vérifier que les coordonnées sont valides
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            self.current_grid_y = row
            self.current_grid_x = col
            
            # Extraire et afficher la nouvelle tuile
            self.extract_and_display_tile()
    
    def update_grid_position(self):
        """Met à jour l'affichage de la position actuelle dans la grille"""
        if not self.is_large_image:
            return
            
        # Calculer la position actuelle (1-indexed pour l'utilisateur)
        position = self.current_grid_y * self.grid_cols + self.current_grid_x + 1
        self.grid_position.set(f"Partie: {position}/{self.grid_rows * self.grid_cols}")
        
        # Mettre à jour le statut avec les coordonnées de la tuile
        x_start = self.current_grid_x * self.tile_width
        y_start = self.current_grid_y * self.tile_height
        x_end = min(x_start + self.tile_width, self.original_image.width)
        y_end = min(y_start + self.tile_height, self.original_image.height)
        
        self.status_var.set(
            f"Partie {position}/{self.grid_rows * self.grid_cols} - "
            f"Région: ({x_start},{y_start}) à ({x_end},{y_end})"
        )
    
    def display_image(self):
        """Affiche l'image dans le canvas avec les points marqués"""
        if self.current_image is None:
            return
        
        # Créer une copie de l'image pour l'affichage
        display_img = self.current_image.copy()
        draw_img = np.array(display_img)
        
        # Dessiner les points pertinents pour la région actuelle
        if self.is_large_image:
            # Calculer les coordonnées de la tuile actuelle
            x_start = self.current_grid_x * self.tile_width
            y_start = self.current_grid_y * self.tile_height
            x_end = min(x_start + self.tile_width, self.original_image.width)
            y_end = min(y_start + self.tile_height, self.original_image.height)
            
            # Dessiner uniquement les points situés dans la région courante
            for strain, points in self.spore_points.items():
                color = self.strain_colors.get(strain, (255, 255, 255))  # Blanc par défaut
                
                # Filtrer les points qui sont dans cette région
                region_points = []
                
                for i, (abs_x, abs_y) in enumerate(points):
                    # Vérifier si le point est dans la région actuelle
                    if x_start <= abs_x < x_end and y_start <= abs_y < y_end:
                        # Convertir les coordonnées absolues en coordonnées relatives à la tuile
                        rel_x = abs_x - x_start
                        rel_y = abs_y - y_start
                        region_points.append((i, rel_x, rel_y))
                
                # Dessiner les points de la région
                for i, rel_x, rel_y in region_points:
                    # Dessiner un cercle pour chaque point
                    cv2.circle(draw_img, (rel_x, rel_y), 10, color, -1)
                    # Ajouter un identifiant numérique
                    cv2.putText(draw_img, str(i+1), (rel_x-5, rel_y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            # Pour les petites images, dessiner tous les points
            for strain, points in self.spore_points.items():
                color = self.strain_colors.get(strain, (255, 255, 255))  # Blanc par défaut
                
                for i, (x, y) in enumerate(points):
                    # Dessiner un cercle pour chaque point
                    cv2.circle(draw_img, (x, y), 10, color, -1)
                    # Ajouter un identifiant numérique
                    cv2.putText(draw_img, str(i+1), (x-5, y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convertir l'image dessinée en PIL
        display_img = Image.fromarray(draw_img)
        
        # Redimensionner l'image selon le zoom
        width, height = display_img.size
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        display_img = display_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convertir pour tkinter
        self.tk_image = ImageTk.PhotoImage(display_img)
        
        # Configurer le canvas
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Afficher l'image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
    
    def add_point(self, event):
        """Ajoute un point sur l'image au clic gauche"""
        if self.current_image is None:
            messagebox.showinfo("Information", "Veuillez d'abord charger une image.")
            return
        
        # Obtenir les coordonnées du clic ajustées au zoom et au défilement
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convertir les coordonnées canvas en coordonnées image
        img_x = int(canvas_x / self.zoom_factor)
        img_y = int(canvas_y / self.zoom_factor)
        
        # Vérifier que le clic est dans les limites de l'image
        width, height = self.current_image.size
        if 0 <= img_x < width and 0 <= img_y < height:
            # Pour les grandes images, convertir les coordonnées relatives en absolues
            if self.is_large_image:
                # Calculer les coordonnées absolues
                abs_x = self.current_grid_x * self.tile_width + img_x
                abs_y = self.current_grid_y * self.tile_height + img_y
            else:
                abs_x = img_x
                abs_y = img_y
            
            # Obtenir la souche sélectionnée
            strain = self.current_strain.get()
            
            # Initialiser la liste de points pour cette souche si nécessaire
            if strain not in self.spore_points:
                self.spore_points[strain] = []
            
            # Ajouter le point (coordonnées absolues)
            self.spore_points[strain].append((abs_x, abs_y))
            
            # Mettre à jour l'affichage et les compteurs
            self.update_counts()
            self.display_image()
    
    def remove_point(self, event):
        """Supprime le point le plus proche du clic droit"""
        if self.current_image is None or not any(self.spore_points.values()):
            return
        
        # Obtenir les coordonnées du clic ajustées au zoom et au défilement
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convertir les coordonnées canvas en coordonnées image
        img_x = int(canvas_x / self.zoom_factor)
        img_y = int(canvas_y / self.zoom_factor)
        
        # Pour les grandes images, convertir les coordonnées relatives en absolues
        if self.is_large_image:
            # Calculer les coordonnées absolues
            abs_x = self.current_grid_x * self.tile_width + img_x
            abs_y = self.current_grid_y * self.tile_height + img_y
            
            # Calculer les limites de la tuile actuelle
            x_start = self.current_grid_x * self.tile_width
            y_start = self.current_grid_y * self.tile_height
            
            # Trouver le point le plus proche dans la région actuelle
            closest_strain = None
            closest_index = -1
            min_distance = float('inf')
            
            for strain, points in self.spore_points.items():
                for i, (point_x, point_y) in enumerate(points):
                    # Vérifier si le point est dans la région actuelle
                    if (x_start <= point_x < x_start + self.tile_width and 
                        y_start <= point_y < y_start + self.tile_height):
                        
                        # Calculer la distance
                        distance = np.sqrt((point_x - abs_x)**2 + (point_y - abs_y)**2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_strain = strain
                            closest_index = i
        else:
            # Pour les petites images, comportement normal
            closest_strain = None
            closest_index = -1
            min_distance = float('inf')
            
            for strain, points in self.spore_points.items():
                for i, (x, y) in enumerate(points):
                    distance = np.sqrt((x - img_x)**2 + (y - img_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_strain = strain
                        closest_index = i
        
        # Supprimer le point s'il est suffisamment proche (seuil dynamique)
        threshold = 30 / self.zoom_factor if self.zoom_factor > 1 else 30
        if min_distance < threshold and closest_strain and closest_index >= 0:
            self.spore_points[closest_strain].pop(closest_index)
            
            # Si plus aucun point pour cette souche, supprimer l'entrée
            if not self.spore_points[closest_strain]:
                del self.spore_points[closest_strain]
            
            # Mettre à jour l'affichage et les compteurs
            self.update_counts()
            self.display_image()
    
    def update_counts(self):
        """Met à jour les compteurs de spores"""
        # Réinitialiser le total
        self.total_count = 0
        
        # Mettre à jour le comptage pour chaque souche
        for strain in self.fungus_strains:
            count = len(self.spore_points.get(strain, []))
            self.counts_by_strain[strain].set(f"{strain}: {count}")
            self.total_count += count
        
        # Mettre à jour le comptage total
        self.total_count_var.set(f"Total: {self.total_count}")
    
    def on_mousewheel(self, event):
        """Gère le zoom avec la molette de la souris"""
        if self.current_image is None:
            return
        
        # Récupérer la position du curseur
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Facteur de zoom
        if event.num == 4 or event.delta > 0:  # Zoom in (Linux / Windows)
            self.zoom_factor *= 1.1
        elif event.num == 5 or event.delta < 0:  # Zoom out (Linux / Windows)
            self.zoom_factor = max(0.1, self.zoom_factor / 1.1)
        
        # Mettre à jour l'affichage
        self.display_image()
        
        # Ajuster la vue pour que la position du curseur reste visible
        self.canvas.xview_moveto(max(0, (x * 1.1 - event.x)) / (self.current_image.width * self.zoom_factor))
        self.canvas.yview_moveto(max(0, (y * 1.1 - event.y)) / (self.current_image.height * self.zoom_factor))
    
    def start_pan(self, event):
        """Commence le déplacement à la pression de la molette"""
        self.canvas.config(cursor="fleur")  # Changer le curseur
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.is_panning = True
    
    def pan_image(self, event):
        """Déplace l'image pendant le déplacement avec la molette enfoncée"""
        if not self.is_panning:
            return
            
        # Calculer le déplacement
        dx = (self.pan_start_x - event.x) / self.zoom_factor
        dy = (self.pan_start_y - event.y) / self.zoom_factor
        
        # Déplacer la vue du canvas
        self.canvas.xview_scroll(int(dx), "units")
        self.canvas.yview_scroll(int(dy), "units")
        
        # Mettre à jour le point de départ
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def stop_pan(self, event):
        """Arrête le déplacement au relâchement de la molette"""
        self.canvas.config(cursor="crosshair")  # Restaurer le curseur
        self.is_panning = False
    
    def reset_zoom(self, event=None):
        """Réinitialise le zoom à 100%"""
        if self.current_image is None:
            return
            
        self.zoom_factor = 1.0
        self.display_image()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
    
    def reset_points(self):
        """Supprime tous les points marqués"""
        if not any(self.spore_points.values()):
            return
            
        if messagebox.askyesno("Réinitialiser", "Voulez-vous supprimer tous les points marqués?"):
            self.spore_points = {}
            self.update_counts()
            self.display_image()
    
    def export_csv(self):
        """Exporte les points marqués en fichier CSV"""
        if not any(self.spore_points.values()):
            messagebox.showinfo("Information", "Aucun point à exporter.")
            return
        
        # Proposer un nom de fichier par défaut basé sur la date et l'heure
        default_name = f"spores_count_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        file_path = filedialog.asksaveasfilename(
            title="Enregistrer le CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Préparer les données
            data = []
            for strain, points in self.spore_points.items():
                for i, (x, y) in enumerate(points):
                    data.append({
                        'ID': i+1,
                        'Souche': strain,
                        'X': x,
                        'Y': y,
                        'Region': self.get_region_for_point(x, y) if self.is_large_image else "Image complète"
                    })
            
            # Créer un résumé
            summary = []
            for strain in self.fungus_strains:
                count = len(self.spore_points.get(strain, []))
                if count > 0:
                    summary.append({
                        'Souche': strain,
                        'Nombre': count,
                        'Pourcentage': f"{count/self.total_count*100:.1f}%"
                    })
            
            # Créer un DataFrame et exporter
            df = pd.DataFrame(data)
            df_summary = pd.DataFrame(summary)
            
            # Ajouter une ligne pour le total
            df_summary = pd.concat([
                df_summary,
                pd.DataFrame([{
                    'Souche': 'TOTAL',
                    'Nombre': self.total_count,
                    'Pourcentage': '100.0%'
                }])
            ])
            
            # Exporter le CSV
            with open(file_path, 'w', newline='') as f:
                f.write("# RÉSUMÉ\n")
                df_summary.to_csv(f, index=False)
                f.write("\n# DÉTAILS DES POINTS\n")
                df.to_csv(f, index=False)
            
            # Ajouter des métadonnées sur l'image
            with open(file_path, 'a', newline='') as f:
                f.write(f"\n# INFORMATIONS SUR L'IMAGE\n")
                f.write(f"Fichier,{os.path.basename(self.current_image_path)}\n")
                f.write(f"Chemin,{self.current_image_path}\n")
                
                # Ajouter des informations sur les dimensions
                if self.original_image:
                    # Si c'est une grande image
                    if self.is_large_image:
                        f.write(f"Dimensions originales,{self.original_image.width}x{self.original_image.height}\n")
                        f.write(f"Division en régions,{self.grid_rows}x{self.grid_cols}\n")
                        f.write(f"Taille des régions,{self.tile_width}x{self.tile_height}\n")
                    else:
                        f.write(f"Dimensions,{self.original_image.width}x{self.original_image.height}\n")
                
                f.write(f"Date d'exportation,{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            messagebox.showinfo("Succès", f"Données exportées avec succès vers:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'exportation: {str(e)}")
    
    def get_region_for_point(self, x, y):
        """Détermine la région (tuile) à laquelle appartient un point"""
        if not self.is_large_image:
            return "Image complète"
            
        # Calculer la position dans la grille
        grid_x = min(int(x / self.tile_width), self.grid_cols - 1)
        grid_y = min(int(y / self.tile_height), self.grid_rows - 1)
        
        # Calculer le numéro de région (1-indexed)
        region_num = grid_y * self.grid_cols + grid_x + 1
        
        return f"Région {region_num}/{self.grid_rows * self.grid_cols}"


def main():
    """
    Fonction principale qui initialise l'application Tkinter.
    
    Cette fonction crée la fenêtre racine Tkinter,
    initialise l'application SporeCounterApp et
    lance la boucle principale.
    """
    root = tk.Tk()
    app = SporeCounterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()