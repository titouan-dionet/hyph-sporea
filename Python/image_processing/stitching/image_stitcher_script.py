#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour l'assemblage d'images de filtres circulaires.
Permet d'assembler les images en une seule image complète en utilisant
une grille et en tenant compte du chevauchement entre les images.

Usage:
    python image_stitcher_script.py --input_dir path/to/images --grid_file path/to/grid.txt [options]
"""

import argparse
import os
from pathlib import Path
import json

# Importer le module d'assemblage d'images
try:
    from image_stitcher import stitch_images, estimate_overlap_from_sample
except ImportError:
    # Si le module n'est pas dans le PYTHONPATH, essayer d'importer depuis le répertoire courant
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from image_stitcher import stitch_images, estimate_overlap_from_sample


def parse_args():
    """Analyse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Assemblage d'images de filtres circulaires")
    
    # Arguments obligatoires
    parser.add_argument("--input_dir", type=str, required=True, 
                      help="Répertoire contenant les images")
    parser.add_argument("--grid_file", type=str, required=True,
                      help="Fichier de grille pour la disposition des images")
    
    # Arguments optionnels
    parser.add_argument("--output_path", type=str, 
                      help="Chemin de sortie pour l'image assemblée")
    parser.add_argument("--h_overlap", type=int, default=105,
                      help="Chevauchement horizontal entre images adjacentes (en pixels)")
    parser.add_argument("--v_overlap", type=int, default=93,
                      help="Chevauchement vertical entre images adjacentes (en pixels)")
    parser.add_argument("--sample_name", type=str,
                      help="Nom de l'échantillon (extrait du nom de fichier si non spécifié)")
    parser.add_argument("--pixel_size", type=float, default=0.264,
                      help="Taille du pixel en mm (0.0104 pouces * 25.4 = 0.264 mm)")
    parser.add_argument("--use_jpeg", action="store_true",
                      help="Utiliser des fichiers JPEG au lieu de TIFF")
    parser.add_argument("--auto_overlap", action="store_true",
                      help="Détecter automatiquement les valeurs de chevauchement")
    parser.add_argument("--num_samples", type=int, default=20,
                      help="Nombre de paires d'images à échantillonner pour la détection d'overlap")
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_args()
    
    # Valider les arguments
    if not os.path.isdir(args.input_dir):
        print(f"Erreur: Le répertoire {args.input_dir} n'existe pas.")
        return 1
    
    if not os.path.isfile(args.grid_file):
        print(f"Erreur: Le fichier de grille {args.grid_file} n'existe pas.")
        return 1
    
    # Définir le chemin de sortie s'il n'est pas spécifié
    if not args.output_path:
        # Déterminer le nom de l'échantillon
        input_dir = Path(args.input_dir)
        sample_name = args.sample_name
        
        if not sample_name:
            # Essayer de détecter automatiquement le nom de l'échantillon
            file_ext = ".jpeg" if args.use_jpeg else ".tiff"
            files = [f for f in os.listdir(args.input_dir) if f.endswith(file_ext)]
            
            if files:
                first_file = files[0]
                sample_name = "_".join(first_file.split("_")[:-1])
            else:
                # Utiliser le nom du répertoire comme nom d'échantillon par défaut
                sample_name = input_dir.name
        
        args.output_path = os.path.join(input_dir.parent, f"{sample_name}_assemblee.jpeg")
    
    # Détection automatique du chevauchement si demandé
    if args.auto_overlap:
        try:
            print("Détection automatique du chevauchement entre les images...")
            pattern = f"{args.sample_name}_*" if args.sample_name else None
            h_overlap, v_overlap = estimate_overlap_from_sample(
                args.input_dir, 
                pattern if pattern else f"*.{'jpeg' if args.use_jpeg else 'tiff'}", 
                num_samples=args.num_samples
            )
            
            args.h_overlap = h_overlap
            args.v_overlap = v_overlap
            
            print(f"Chevauchement horizontal détecté: {h_overlap} pixels")
            print(f"Chevauchement vertical détecté: {v_overlap} pixels")
        except Exception as e:
            print(f"Erreur lors de la détection automatique du chevauchement: {str(e)}")
            print("Utilisation des valeurs par défaut.")
    
    # Exécuter l'assemblage d'images
    try:
        result = stitch_images(
            args.input_dir,
            args.grid_file,
            args.output_path,
            h_overlap=args.h_overlap,
            v_overlap=args.v_overlap,
            sample_name=args.sample_name,
            pixel_size_mm=args.pixel_size,
            use_tiff=not args.use_jpeg
        )
        
        print(f"\nAssemblage réussi!")
        print(f"Image assemblée sauvegardée dans: {args.output_path}")
        
        # Afficher les informations supplémentaires
        info_path = Path(args.output_path).with_suffix('.json')
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                print("\nInformations sur l'image assemblée:")
                print(f"- Échantillon: {info.get('sample_name', 'N/A')}")
                print(f"- Dimensions (pixels): {info.get('image_dimensions', 'N/A')}")
                print(f"- Dimensions physiques: {info.get('physical_dimensions', 'N/A')}")
                print(f"- Images traitées: {info.get('processed_count', 0)}/{info.get('original_count', 0)}")
            except Exception:
                pass
        
        return 0
    
    except Exception as e:
        print(f"Erreur lors de l'assemblage des images: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)