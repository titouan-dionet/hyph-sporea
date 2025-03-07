"""
Exemple de workflow pour le projet HYPH-SPOREA.

Ce script montre comment utiliser le module de workflow pour créer
un pipeline de traitement d'images et d'analyse de spores.
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au PATH pour importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from workflow import Workflow, target
from pipeline_config import load_config
from Python.core import HyphSporeaProcessor
from Python.image_processing import batch_preprocess_directory, create_binary_masks
from Python.models import train_yolo_model, batch_predict_with_yolo
from Python.analysis import analyze_detections, save_analysis_results, compare_samples


def main():
    """Exemple d'utilisation du workflow"""
    # Charger la configuration
    config = load_config()
    
    # Créer le workflow
    workflow = Workflow("hyph_sporea_pipeline")
    
    # Définir le processeur HYPH-SPOREA
    processor = HyphSporeaProcessor(config['PROJECT_ROOT'])
    
    # ===== Étape 1: Conversion des images TIFF en JPEG =====
    @target(
        workflow=workflow,
        name="convert_tiff_to_jpeg",
        file_inputs=[config['raw_data_dir'] / "example_tiffs"],
        file_outputs=[config['proc_data_dir'] / "jpeg_images"]
    )
    def convert_images():
        """Convertit les images TIFF en JPEG"""
        return processor.convert_tiff_to_jpeg(
            config['raw_data_dir'] / "example_tiffs",
            config['proc_data_dir'] / "jpeg_images"
        )
    
    # ===== Étape 2: Prétraitement des images =====
    @target(
        workflow=workflow,
        name="preprocess_images",
        inputs=["convert_tiff_to_jpeg"],
        file_outputs=[config['proc_data_dir'] / "preprocessed_images"]
    )
    def preprocess_images(jpeg_dir):
        """Prétraite les images JPEG"""
        return batch_preprocess_directory(
            jpeg_dir,
            config['proc_data_dir'] / "preprocessed_images"
        )
    
    # ===== Étape 3: Création des masques pour U-Net =====
    @target(
        workflow=workflow,
        name="create_masks",
        inputs=["preprocess_images"],
        file_outputs=[config['proc_data_dir'] / "masks"]
    )
    def create_masks(preprocessed_dir):
        """Crée des masques binaires pour l'entraînement U-Net"""
        preprocessed_dir = config['proc_data_dir'] / "preprocessed_images"
        return create_binary_masks(
            preprocessed_dir,
            config['proc_data_dir'] / "masks"
        )
    
    # ===== Étape 3b: Première utilisation de l'interface =====
    @target(
        workflow=workflow,
        name="annotate_images",
        inputs=["preprocess_images"],
        file_outputs=[config['proc_data_dir'] / "yolo_annotations" / "data.yaml"]
    )
    def annotate_images(_):
        """Lance l'interface graphique d'annotation et attend que l'utilisateur finisse"""
        from Python.gui.annotation_tool import run_annotation_tool
        
        # Vérifier si le fichier data.yaml existe déjà
        yaml_path = config['proc_data_dir'] / "yolo_annotations" / "data.yaml"
        if yaml_path.exists():
            print(f"Fichier d'annotations {yaml_path} trouvé, étape d'annotation ignorée.")
            return yaml_path
        
        print("\n" + "="*50)
        print("ÉTAPE INTERACTIVE: Veuillez annoter quelques images")
        print("1. Une interface graphique va s'ouvrir")
        print("2. Chargez les images prétraitées depuis:", config['proc_data_dir'] / "preprocessed_images")
        print("3. Annotez au moins quelques images")
        print("4. Enregistrez et convertissez en format YOLO avant de fermer")
        print("="*50 + "\n")
        
        # Lancer l'interface d'annotation
        input("Appuyez sur Entrée pour lancer l'interface d'annotation...")
        run_annotation_tool()
        
        # Vérifier si l'annotation a été complétée
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Le fichier {yaml_path} n'a pas été créé. "
                "Assurez-vous de sauvegarder et convertir en format YOLO avant de fermer l'interface."
            )
        
        print("Annotation terminée avec succès!")
        return yaml_path
    
    # ===== Étape 4: Entraînement du modèle YOLO =====
    @target(
        workflow=workflow,
        name="train_model",
        file_inputs=["annotate_images"],
        file_outputs=[config['models_dir'] / "yolo_spores_model.pt"]
    )
    def train_model(data_yaml_path):
        """Entraîne le modèle YOLO"""
        return train_yolo_model(
            data_yaml_path,
            epochs=5,
            output_dir=config['models_dir']
        )
    
    # ===== Étape 5: Traitement de l'échantillon 1 =====
    @target(
        workflow=workflow,
        name="process_sample_1",
        inputs=["train_model"],
        file_inputs=[config['proc_data_dir'] / "jpeg_images" / "sample_1"],
        file_outputs=[config['output_dir'] / "processed_sample_1"]
    )
    def process_sample_1(model_path):
        """Traite l'échantillon 1"""
        # Détection avec YOLO
        detections = batch_predict_with_yolo(
            model_path,
            config['proc_data_dir'] / "jpeg_images" / "sample_1",
            config['output_dir'] / "processed_sample_1"
        )
        
        # Analyse des résultats
        sample_info = {
            'sample': 'sample_1',
            'strain': 'T1_ALAC',
            'condition': 'C6',
            'replicate': '1'
        }
        
        stats = analyze_detections(detections, sample_info)
        
        # Sauvegarder les résultats
        save_analysis_results(
            stats,
            config['output_dir'] / "processed_sample_1" / "analysis"
        )
        
        return {
            'detections': detections,
            'stats': stats
        }
    
    # ===== Étape 6: Traitement de l'échantillon 2 =====
    @target(
        workflow=workflow,
        name="process_sample_2",
        inputs=["train_model"],
        file_inputs=[config['proc_data_dir'] / "jpeg_images" / "sample_2"],
        file_outputs=[config['output_dir'] / "processed_sample_2"]
    )
    def process_sample_2(model_path):
        """Traite l'échantillon 2"""
        # Détection avec YOLO
        detections = batch_predict_with_yolo(
            model_path,
            config['proc_data_dir'] / "jpeg_images" / "sample_2",
            config['output_dir'] / "processed_sample_2"
        )
        
        # Analyse des résultats
        sample_info = {
            'sample': 'sample_2',
            'strain': 'T1_ALAC',
            'condition': 'C7',
            'replicate': '1'
        }
        
        stats = analyze_detections(detections, sample_info)
        
        # Sauvegarder les résultats
        save_analysis_results(
            stats,
            config['output_dir'] / "processed_sample_2" / "analysis"
        )
        
        return {
            'detections': detections,
            'stats': stats
        }
    
    # ===== Étape 7: Comparaison des échantillons =====
    @target(
        workflow=workflow,
        name="compare_samples",
        inputs=["process_sample_1", "process_sample_2"],
        file_outputs=[config['output_dir'] / "comparison"]
    )
    def compare_results(sample1_results, sample2_results):
        """Compare les résultats des échantillons"""
        # Fusionner les résultats
        all_results = {
            'sample_1': sample1_results['stats'],
            'sample_2': sample2_results['stats']
        }
        
        # Comparer les échantillons
        compare_df, _ = compare_samples(
            all_results,
            config['output_dir'] / "comparison"
        )
        
        return compare_df
    
    # Visualiser le workflow
    workflow.visualize(config['output_dir'] / "visualizations" / "workflow_graph.png")
    
    # Exécuter le workflow
    print("\nExécution du workflow complet...")
    results = workflow.execute(parallel=False)
    
    print("\nRésultats du workflow:")
    for target_name, result in results.items():
        if target_name == "compare_samples":
            print(f"\nComparaison des échantillons:")
            print(result[['sample', 'total_spores_detected']])
    
    print("\nWorkflow terminé.")


if __name__ == "__main__":
    main()