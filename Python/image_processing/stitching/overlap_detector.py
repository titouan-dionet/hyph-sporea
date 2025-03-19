#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour détecter le chevauchement entre les images.
Permet de déterminer avec précision la valeur de l'overlap horizontal et vertical.

Usage:
    python overlap_detector.py --input_dir path/to/images [options]
"""

import argparse
import os
from pathlib import Path
import json

# Importer le module d'assemblage d'images
try:
    from image_stitcher import run_overlap_detection
except ImportError:
    # Si le module n'est pas dans le PYTHONPATH, essayer d'importer depuis le répertoire courant
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from image_stitcher import run_overlap_detection


def parse_args():
    """Analyse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Détection de chevauchement entre images")
    
    # Arguments obligatoires
    parser.add_argument("--input_dir", type=str, required=True, 
                      help="Répertoire contenant les images")
    
    # Arguments optionnels
    parser.add_argument("--output_file", type=str, 
                      help="Fichier de sortie pour les résultats (JSON)")
    parser.add_argument("--num_samples", type=int, default=20,
                      help="Nombre de paires d'images à échantillonner pour la détection")
    parser.add_argument("--max_overlap", type=int, default=200,
                      help="Chevauchement maximal à vérifier (en pixels)")
    parser.add_argument("--use_jpeg", action="store_true",
                      help="Utiliser des fichiers JPEG au lieu de TIFF")
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_args()
    
    # Valider les arguments
    if not os.path.isdir(args.input_dir):
        print(f"Erreur: Le répertoire {args.input_dir} n'existe pas.")
        return 1
    
    # Définir le fichier de sortie s'il n'est pas spécifié
    if not args.output_file:
        input_dir = Path(args.input_dir)
        # Utiliser le nom du répertoire parent pour nommer le fichier de sortie
        sample_name = input_dir.name
        args.output_file = input_dir / f"{sample_name}_overlap.json"
    
    # Exécuter la détection de chevauchement
    try:
        h_overlap, v_overlap = run_overlap_detection(
            args.input_dir,
            args.output_file,
            num_samples=args.num_samples,
            max_overlap=args.max_overlap,
            use_tiff=not args.use_jpeg
        )
        
        print("\nRésultats:")
        print(f"- Chevauchement horizontal estimé: {h_overlap} pixels")
        print(f"- Chevauchement vertical estimé: {v_overlap} pixels")
        print(f"- Résultats sauvegardés dans: {args.output_file}")
        
        return 0
    
    except Exception as e:
        print(f"Erreur lors de la détection de chevauchement: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)