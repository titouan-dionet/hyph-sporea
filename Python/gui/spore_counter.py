import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import cv2
import datetime

class SporeCounterApp:
    """
    Application pour compter et classifier des spores sur des images de filtres circulaires.
    
    Fonctionnalités:
    - Charger une image de filtre
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
        self.root.geometry("1400x900")
        
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
        self.current_strain = tk.StringVar(value=self.fungus_strains[0])
        self.spore_points = {}  # {strain: [(x, y), ...]}
        self.total_count = 0
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        
        # Configurer l'interface
        self.setup_ui()
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame gauche pour les contrôles
        control_frame = ttk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Bouton pour charger une image
        ttk.Button(control_frame, text="Charger une image", command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Sélection de la souche
        strain_frame = ttk.LabelFrame(control_frame, text="Souche de champignon")
        strain_frame.pack(fill=tk.X, pady=5, padx=5)
        
        for strain in self.fungus_strains:
            rb = ttk.Radiobutton(
                strain_frame, 
                text=strain, 
                variable=self.current_strain, 
                value=strain
            )
            rb.pack(anchor=tk.W, padx=5)
        
        # Compteur de spores par souche
        count_frame = ttk.LabelFrame(control_frame, text="Comptage des spores")
        count_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.counts_by_strain = {}
        for strain in self.fungus_strains:
            self.counts_by_strain[strain] = tk.StringVar(value=f"{strain}: 0")
            ttk.Label(count_frame, textvariable=self.counts_by_strain[strain]).pack(anchor=tk.W, padx=5)
        
        # Compteur total
        self.total_count_var = tk.StringVar(value="Total: 0")
        ttk.Label(control_frame, textvariable=self.total_count_var, font=("Arial", 12, "bold")).pack(pady=10)
        
        # Instructions
        instruction_frame = ttk.LabelFrame(control_frame, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=5, padx=5)
        
        instructions = [
            "- Clic gauche: ajouter un point",
            "- Clic droit: supprimer un point",
            "- Molette: zoom avant/arrière",
            "- Clic molette + déplacer: naviguer",
            "- Double-clic: réinitialiser zoom"
        ]
        
        for instr in instructions:
            ttk.Label(instruction_frame, text=instr, wraplength=230).pack(anchor=tk.W, padx=5, pady=2)
        
        # Boutons d'action
        ttk.Button(control_frame, text="Exporter CSV", command=self.export_csv).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Réinitialiser points", command=self.reset_points).pack(fill=tk.X, pady=5)
        
        # Frame droite pour l'affichage de l'image
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
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
        
        # Statut
        self.status_var = tk.StringVar(value="Prêt. Chargez une image pour commencer.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        """Charge une image et l'affiche dans le canvas"""
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
            # Charger l'image avec OpenCV
            self.current_image_cv = cv2.imread(file_path)
            
            if self.current_image_cv is None:
                messagebox.showerror("Erreur", "Impossible de lire l'image. Format non supporté.")
                return
            
            # Convertir de BGR à RGB pour l'affichage
            img_rgb = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(img_rgb)
            
            # Mettre à jour le chemin de l'image
            self.current_image_path = file_path
            
            # Réinitialiser les points et le zoom
            self.reset_points()
            self.zoom_factor = 1.0
            
            # Afficher l'image
            self.display_image()
            
            # Mettre à jour le statut
            file_name = os.path.basename(file_path)
            height, width = self.current_image_cv.shape[:2]
            self.status_var.set(f"Image chargée: {file_name} ({width}x{height})")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement de l'image: {str(e)}")
    
    def display_image(self):
        """Affiche l'image dans le canvas avec les points marqués"""
        if self.current_image is None:
            return
        
        # Créer une copie de l'image pour l'affichage
        display_img = self.current_image.copy()
        draw_img = np.array(display_img)
        
        # Dessiner les points
        for strain, points in self.spore_points.items():
            color = self.strain_colors.get(strain, (255, 255, 255))  # Blanc par défaut
            # Convertir BGR à RGB pour PIL
            rgb_color = (color[2], color[1], color[0])
            
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
            # Obtenir la souche sélectionnée
            strain = self.current_strain.get()
            
            # Initialiser la liste de points pour cette souche si nécessaire
            if strain not in self.spore_points:
                self.spore_points[strain] = []
            
            # Ajouter le point
            self.spore_points[strain].append((img_x, img_y))
            
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
        
        # Trouver le point le plus proche
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
        
        # Supprimer le point s'il est suffisamment proche (seuil de 30 pixels)
        if min_distance < 30 and closest_strain and closest_index >= 0:
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
                        'Y': y
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
                if self.current_image:
                    f.write(f"Dimensions,{self.current_image.width}x{self.current_image.height}\n")
                f.write(f"Date d'exportation,{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            messagebox.showinfo("Succès", f"Données exportées avec succès vers:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'exportation: {str(e)}")


def run_spore_counter():
    """
    Lance l'outil de comptage de spores comme application standalone.
    
    Returns:
        None
    
    Example:
        >>> run_spore_counter()
    """
    root = tk.Tk()
    app = SporeCounterApp(root)
    root.mainloop()


def main():
    root = tk.Tk()
    app = SporeCounterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
