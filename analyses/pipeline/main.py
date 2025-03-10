#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPH-SPOREA: Script principal pour l'analyse de spores d'hyphomycètes
"""

import argparse
import os
import pandas as pd
import torch
from pathlib import Path
import sys

from Python.core import HyphSporeaProcessor


def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="HYPH-SPOREA: Analyse de spores d'hyphomycètes")
    
    # Argument obligatoire: chemin du projet
    parser.add_argument("--project_dir", type=str, required=True, 
                        help="Chemin du répertoire du projet")
    
    # Sous-commandes
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Sous-commande: convertir
    convert_parser = subparsers.add_parser("convert", help="Convertir les images TIFF en JPEG")
    convert_parser.add_argument("--input_dir", type=str, required=True, 
                               help="Répertoire contenant les images TIFF")
    convert_parser.add_argument("--output_dir", type=str, 
                               help="Répertoire de sortie pour les images JPEG")
    
    # Sous-commande: prétraiter
    preprocess_parser = subparsers.add_parser("preprocess", help="Prétraiter les images")
    preprocess_parser.add_argument("--input_dir", type=str, required=True, 
                                  help="Répertoire contenant les images JPEG")
    preprocess_parser.add_argument("--output_dir", type=str, 
                                  help="Répertoire de sortie pour les images prétraitées")
    
    # Sous-commande: annoter
    annotate_parser = subparsers.add_parser("annotate", help="Lancer l'outil d'annotation")
    
    # Sous-commande: entraîner
    train_parser = subparsers.add_parser("train", help="Entraîner un modèle")
    train_parser.add_argument("--model_type", type=str, choices=["yolo", "unet"], required=True,
                             help="Type de modèle à entraîner")
    train_parser.add_argument("--data_yaml", type=str, 
                             help="Chemin du fichier data.yaml (pour YOLO)")
    train_parser.add_argument("--image_dir", type=str, 
                             help="Répertoire contenant les images (pour U-Net)")
    train_parser.add_argument("--mask_dir", type=str, 
                             help="Répertoire contenant les masques (pour U-Net)")
    train_parser.add_argument("--epochs", type=int, default=100, 
                             help="Nombre d'époques d'entraînement")
    
    # Sous-commande: traiter
    process_parser = subparsers.add_parser("process", help="Traiter des échantillons")
    process_parser.add_argument("--sample_dirs", type=str, nargs="+", required=True,
                               help="Répertoires des échantillons à traiter")
    process_parser.add_argument("--model_path", type=str, required=True,
                               help="Chemin du modèle à utiliser")
    process_parser.add_argument("--output_dir", type=str,
                               help="Répertoire de sortie pour les résultats")
    
    # Sous-commande: comparer
    compare_parser = subparsers.add_parser("compare", help="Comparer des échantillons")
    compare_parser.add_argument("--result_dirs", type=str, nargs="+", required=True,
                               help="Répertoires contenant les résultats de traitement")
    compare_parser.add_argument("--output_dir", type=str,
                               help="Répertoire de sortie pour la comparaison")
    
    # Sous-commande: pipeline
    pipeline_parser = subparsers.add_parser("pipeline", help="Exécuter le pipeline complet")
    pipeline_parser.add_argument("--tiff_dir", type=str, required=True,
                                help="Répertoire contenant les images TIFF")
    pipeline_parser.add_argument("--model_type", type=str, choices=["yolo", "unet"], required=True,
                                help="Type de modèle à entraîner/utiliser")
    pipeline_parser.add_argument("--sample_dirs", type=str, nargs="+", required=True,
                                help="Répertoires des échantillons à traiter")
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_args()
    
    # Initialiser le processeur
    processor = HyphSporeaProcessor(args.project_dir)
    
    # Exécuter la commande spécifiée
    if args.command == "convert":
        output_dir = args.output_dir if args.output_dir else None
        processor.convert_tiff_to_jpeg(args.input_dir, output_dir)
    
    elif args.command == "preprocess":
        output_dir = args.output_dir if args.output_dir else None
        processor.preprocess_dataset(args.input_dir, output_dir)
    
    elif args.command == "annotate":
        processor.create_annotation_tool()
    
    elif args.command == "train":
        if args.model_type == "yolo":
            if not args.data_yaml:
                print("Erreur: --data_yaml est requis pour l'entraînement YOLO")
                return
            processor.train_yolo_model(args.data_yaml, epochs=args.epochs)
        
        elif args.model_type == "unet":
            if not args.image_dir or not args.mask_dir:
                print("Erreur: --image_dir et --mask_dir sont requis pour l'entraînement U-Net")
                return
            processor.train_unet_model(args.image_dir, args.mask_dir, epochs=args.epochs)
    
    elif args.command == "process":
        for sample_dir in args.sample_dirs:
            output_dir = args.output_dir if args.output_dir else None
            processor.process_sample(sample_dir, args.model_path, output_dir)
    
    elif args.command == "compare":
        # Charger les résultats des répertoires spécifiés
        for result_dir in args.result_dirs:
            sample_name = os.path.basename(result_dir).replace('processed_', '')
            stats_file = os.path.join(result_dir, 'analysis', 'sample_stats.csv')
            
            if os.path.exists(stats_file):
                stats = pd.read_csv(stats_file).iloc[0].to_dict()
                processor.results[sample_name] = stats
        
        output_dir = args.output_dir if args.output_dir else None
        processor.compare_samples(output_dir)
    
    elif args.command == "pipeline":
        # Exécuter le pipeline complet
        print("Exécution du pipeline complet...")
        
        # 1. Convertir les images TIFF en JPEG
        jpeg_dir = processor.convert_tiff_to_jpeg(args.tiff_dir)
        
        # 2. Prétraiter les images
        preprocessed_dir = processor.preprocess_dataset(jpeg_dir)
        
        # 3. Lancer l'outil d'annotation
        print("\nVeuillez annoter quelques images pour l'entraînement...")
        processor.create_annotation_tool()
        
        # 4. Entraîner le modèle
        if args.model_type == "yolo":
            # Rechercher le fichier data.yaml généré par l'outil d'annotation
            data_yaml = os.path.join(preprocessed_dir, "yolo_annotations", "data.yaml")
            if not os.path.exists(data_yaml):
                print("Erreur: Fichier data.yaml non trouvé. Assurez-vous d'avoir annoté des images.")
                return
            
            model_path = processor.train_yolo_model(data_yaml)
        
        elif args.model_type == "unet":
            # Rechercher les répertoires d'images et de masques
            image_dir = preprocessed_dir
            mask_dir = os.path.join(preprocessed_dir, "masks")
            if not os.path.exists(mask_dir):
                print("Erreur: Répertoire de masques non trouvé. Assurez-vous d'avoir généré des masques.")
                return
            
            model_path = processor.train_unet_model(image_dir, mask_dir)
        
        # 5. Traiter les échantillons
        for sample_dir in args.sample_dirs:
            processor.process_sample(sample_dir, model_path)
        
        # 6. Comparer les résultats
        comparison_dir = processor.compare_samples()
        
        print(f"\nPipeline complet terminé. Résultats disponibles dans {comparison_dir}")
    
    else:
        print("Erreur: Commande non reconnue. Utilisez --help pour voir les options disponibles.")


if __name__ == "__main__":
    main()
