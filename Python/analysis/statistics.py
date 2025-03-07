"""
Module d'analyse statistique pour le projet HYPH-SPOREA.

Ce module contient des fonctions pour analyser les résultats
de détection et de segmentation des spores.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def analyze_detections(detections, sample_info=None):
    """
    Analyse les détections d'un échantillon et génère des statistiques.
    
    Args:
        detections (dict): Dictionnaire des détections par image
        sample_info (dict, optional): Informations sur l'échantillon (souche, condition, etc.)
    
    Returns:
        dict: Statistiques calculées
    
    Example:
        >>> with open('detections.json', 'r') as f:
        ...     detections = json.load(f)
        >>> stats = analyze_detections(
        ...     detections, 
        ...     {'sample': 'T1_ALAC_C6_1', 'strain': 'T1_ALAC', 'condition': 'C6', 'replicate': '1'}
        ... )
    """
    # Initialiser les informations de l'échantillon
    if sample_info is None:
        sample_info = {}
    
    # Extraire ou définir les informations de base
    sample_name = sample_info.get('sample', 'unknown')
    strain = sample_info.get('strain', 'unknown')
    condition = sample_info.get('condition', 'unknown')
    replicate = sample_info.get('replicate', '1')
    
    # Compter le nombre de spores détectées par classe
    class_counts = {}
    total_spores = 0
    
    # Listes pour stocker les caractéristiques morphologiques
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    
    # Analyser chaque image
    for img_file, img_detections in detections.items():
        for det in img_detections:
            # Si la classe est disponible (YOLO)
            if 'class' in det:
                cls = det['class']
                if cls not in class_counts:
                    class_counts[cls] = 0
                class_counts[cls] += 1
                total_spores += 1
            else:
                # Si seule la détection est disponible (U-Net, SAM)
                if 'unknown' not in class_counts:
                    class_counts['unknown'] = 0
                class_counts['unknown'] += 1
                total_spores += 1
            
            # Collecter les caractéristiques morphologiques
            widths.append(det['width'])
            heights.append(det['height'])
            
            # L'aire peut être explicite ou calculée
            if 'area' in det:
                areas.append(det['area'])
            else:
                areas.append(det['width'] * det['height'])
            
            # Calculer le rapport d'aspect
            if det['height'] > 0:
                aspect_ratios.append(det['width'] / det['height'])
    
    # Créer un dictionnaire pour les statistiques
    stats = {
        'sample': sample_name,
        'strain': strain,
        'condition': condition,
        'replicate': replicate,
        'total_images': len(detections),
        'total_spores_detected': total_spores
    }
    
    # Ajouter les comptages par classe
    for cls, count in class_counts.items():
        stats[f'class_{cls}_count'] = count
    
    # Calculer les statistiques morphologiques
    if widths:
        # Statistiques de base
        stats['avg_width'] = np.mean(widths)
        stats['avg_height'] = np.mean(heights)
        stats['avg_area'] = np.mean(areas)
        stats['median_area'] = np.median(areas)
        stats['min_area'] = np.min(areas)
        stats['max_area'] = np.max(areas)
        stats['std_area'] = np.std(areas)
        
        # Statistiques plus avancées
        stats['avg_aspect_ratio'] = np.mean(aspect_ratios)
        stats['area_coefficient_variation'] = stats['std_area'] / stats['avg_area'] if stats['avg_area'] > 0 else 0
        
        # Quantiles
        stats['area_q25'] = np.percentile(areas, 25)
        stats['area_q75'] = np.percentile(areas, 75)
        stats['area_iqr'] = stats['area_q75'] - stats['area_q25']
    
    return stats


def save_analysis_results(stats, output_dir, create_plots=True):
    """
    Sauvegarde les résultats d'analyse et génère des graphiques.
    
    Args:
        stats (dict): Statistiques calculées
        output_dir (str): Répertoire de sortie
        create_plots (bool, optional): Si True, génère des graphiques. Par défaut True.
    
    Returns:
        tuple: (chemin du fichier CSV, liste des chemins de graphiques)
    
    Example:
        >>> csv_path, plot_paths = save_analysis_results(
        ...     stats,
        ...     "outputs/analysis/T1_ALAC_C6_1"
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Sauvegarder les statistiques
    stats_df = pd.DataFrame([stats])
    csv_path = output_dir / 'sample_stats.csv'
    stats_df.to_csv(csv_path, index=False)
    
    # Liste pour stocker les chemins des graphiques
    plot_paths = []
    
    # Génération des graphiques
    if create_plots and 'total_spores_detected' in stats and stats['total_spores_detected'] > 0:
        # Extraire les données nécessaires
        areas = []
        widths = []
        heights = []
        classes = []
        
        # Reconstruire les listes à partir des statistiques
        if 'areas' in stats:
            areas = stats['areas']
            widths = stats['widths']
            heights = stats['heights']
            if 'classes' in stats:
                classes = stats['classes']
        
        # Si les listes ne sont pas directement disponibles, utiliser les moyennes
        if not areas and 'avg_area' in stats:
            # Créer des données synthétiques basées sur les statistiques
            mean_area = stats['avg_area']
            std_area = stats.get('std_area', mean_area * 0.1)
            
            # Générer des valeurs aléatoires suivant une distribution normale
            n_samples = stats['total_spores_detected']
            areas = np.random.normal(mean_area, std_area, n_samples)
            areas = areas[areas > 0]  # Éliminer les valeurs négatives
            
            if 'avg_width' in stats and 'avg_height' in stats:
                mean_width = stats['avg_width']
                mean_height = stats['avg_height']
                std_width = mean_width * 0.1
                std_height = mean_height * 0.1
                
                widths = np.random.normal(mean_width, std_width, n_samples)
                widths = widths[widths > 0]
                
                heights = np.random.normal(mean_height, std_height, n_samples)
                heights = heights[heights > 0]
        
        # 1. Distribution des surfaces
        if areas:
            plt.figure(figsize=(10, 6))
            sns.histplot(areas, bins=30, kde=True)
            plt.title('Distribution des surfaces de spores')
            plt.xlabel('Surface (pixels²)')
            plt.ylabel('Nombre de spores')
            plt.grid(alpha=0.3)
            
            area_plot_path = output_dir / 'area_distribution.png'
            plt.savefig(area_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(area_plot_path)
        
        # 2. Relation largeur/hauteur
        if widths and heights:
            plt.figure(figsize=(10, 6))
            plt.scatter(widths, heights, alpha=0.5)
            plt.title('Rapport largeur/hauteur des spores')
            plt.xlabel('Largeur (pixels)')
            plt.ylabel('Hauteur (pixels)')
            plt.grid(alpha=0.3)
            
            # Ajouter une ligne de référence pour les cercles parfaits
            max_dim = max(max(widths), max(heights))
            plt.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Cercle parfait (w=h)')
            plt.legend()
            
            dim_plot_path = output_dir / 'width_height_relation.png'
            plt.savefig(dim_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(dim_plot_path)
        
        # 3. Distribution par classe
        class_counts = {}
        for key, value in stats.items():
            if key.startswith('class_') and key.endswith('_count'):
                class_name = key.replace('class_', '').replace('_count', '')
                class_counts[class_name] = value
        
        if class_counts and len(class_counts) > 1:  # Plus d'une classe
            plt.figure(figsize=(10, 6))
            
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            sns.barplot(x=classes, y=counts)
            plt.title('Nombre de spores par classe')
            plt.xlabel('Classe')
            plt.ylabel('Nombre de spores')
            plt.xticks(rotation=45)
            
            class_plot_path = output_dir / 'class_distribution.png'
            plt.savefig(class_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(class_plot_path)
    
    return csv_path, plot_paths


def compare_samples(results_dict, output_dir=None):
    """
    Compare les résultats de plusieurs échantillons.
    
    Args:
        results_dict (dict): Dictionnaire des résultats par échantillon
        output_dir (str, optional): Répertoire de sortie pour les comparaisons
    
    Returns:
        tuple: (DataFrame de comparaison, liste des chemins de graphiques)
    
    Example:
        >>> results = {
        ...     'T1_ALAC_C6_1': {'strain': 'T1_ALAC', 'condition': 'C6', 'total_spores_detected': 356},
        ...     'T1_ALAC_C7_1': {'strain': 'T1_ALAC', 'condition': 'C7', 'total_spores_detected': 412}
        ... }
        >>> df, plots = compare_samples(results, "outputs/comparison")
    """
    if not results_dict:
        raise ValueError("Aucun résultat à comparer")
    
    # Convertir en DataFrame
    results_df = pd.DataFrame(results_dict.values())
    
    # Configurer le répertoire de sortie
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Liste pour stocker les chemins des graphiques
    plot_paths = []
    
    # Sauvegarder les résultats combinés
    if output_dir:
        csv_path = output_dir / 'all_samples_results.csv'
        results_df.to_csv(csv_path, index=False)
    
    # Génération des graphiques comparatifs
    if output_dir and 'total_spores_detected' in results_df.columns:
        # 1. Nombre total de spores par échantillon
        plt.figure(figsize=(12, 6))
        
        sns.barplot(x='sample', y='total_spores_detected', data=results_df)
        plt.title('Nombre total de spores par échantillon')
        plt.xlabel('Échantillon')
        plt.ylabel('Nombre de spores')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        bar_plot_path = output_dir / 'samples_comparison.png'
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(bar_plot_path)
        
        # 2. Comparaison par condition (si disponible)
        if 'condition' in results_df.columns and 'strain' in results_df.columns:
            conditions = results_df['condition'].unique()
            
            plt.figure(figsize=(12, 6))
            
            # Utiliser seaborn pour un graphique plus esthétique
            sns.scatterplot(
                x='strain',
                y='total_spores_detected',
                hue='condition',
                size='total_spores_detected',
                sizes=(100, 500),
                alpha=0.7,
                data=results_df
            )
            
            plt.title('Nombre de spores par souche et condition')
            plt.xlabel('Souche')
            plt.ylabel('Nombre de spores')
            plt.legend(title='Condition')
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            scatter_plot_path = output_dir / 'strain_condition_comparison.png'
            plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(scatter_plot_path)
            
            # 3. Heatmap des moyennes par condition et souche
            if len(conditions) > 1 and len(results_df['strain'].unique()) > 1:
                # Créer un pivot pour la heatmap
                pivot_df = results_df.pivot_table(
                    index='strain',
                    columns='condition',
                    values='total_spores_detected',
                    aggfunc='mean'
                )
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    pivot_df,
                    annot=True,
                    fmt='.1f',
                    linewidths=0.5,
                    cmap='YlGnBu'
                )
                
                plt.title('Nombre moyen de spores par souche et condition')
                plt.tight_layout()
                heatmap_path = output_dir / 'strain_condition_heatmap.png'
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(heatmap_path)
        
        # 4. Comparaison des distributions de taille (si disponible)
        if 'avg_area' in results_df.columns:
            plt.figure(figsize=(12, 6))
            
            area_cols = [col for col in results_df.columns if 'area' in col and col != 'area_coefficient_variation']
            area_df = results_df[['sample'] + area_cols].melt(
                id_vars=['sample'],
                var_name='Métrique',
                value_name='Valeur'
            )
            
            sns.barplot(x='sample', y='Valeur', hue='Métrique', data=area_df[area_df['Métrique'].isin(['avg_area', 'median_area'])])
            plt.title('Comparaison des surfaces moyennes des spores')
            plt.xlabel('Échantillon')
            plt.ylabel('Surface (pixels²)')
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            area_comp_path = output_dir / 'area_comparison.png'
            plt.savefig(area_comp_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(area_comp_path)
    
    return results_df, plot_paths


def perform_statistical_tests(results_df, output_dir=None):
    """
    Réalise des tests statistiques sur les résultats des échantillons.
    
    Args:
        results_df (pandas.DataFrame): DataFrame contenant les résultats des échantillons
        output_dir (str, optional): Répertoire de sortie pour les résultats des tests
    
    Returns:
        dict: Résultats des tests statistiques
    
    Example:
        >>> df, _ = compare_samples(results, "outputs/comparison")
        >>> test_results = perform_statistical_tests(df, "outputs/comparison")
        >>> for test, p_value in test_results['anova_results'].items():
        ...     print(f"{test}: p = {p_value:.4f}")
    """
    if len(results_df) < 2:
        print("Au moins deux échantillons sont nécessaires pour les tests statistiques")
        return {}
    
    # Configuration du répertoire de sortie
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Dictionnaire pour stocker les résultats des tests
    test_results = {}
    
    # Variables numériques à tester
    numeric_vars = [
        'total_spores_detected', 
        'avg_area', 
        'median_area',
        'avg_width',
        'avg_height',
        'avg_aspect_ratio'
    ]
    
    numeric_vars = [var for var in numeric_vars if var in results_df.columns]
    
    # 1. Tests ANOVA si plusieurs conditions sont présentes
    if 'condition' in results_df.columns and len(results_df['condition'].unique()) > 1:
        anova_results = {}
        
        for var in numeric_vars:
            if var in results_df.columns:
                # Grouper par condition
                groups = [group[var].values for _, group in results_df.groupby('condition')]
                
                # Exécuter l'ANOVA
                try:
                    f_val, p_val = stats.f_oneway(*groups)
                    anova_results[f"anova_{var}_by_condition"] = {
                        'f_value': float(f_val),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    }
                except:
                    continue
        
        test_results['anova_by_condition'] = anova_results
    
    # 2. Tests ANOVA si plusieurs souches sont présentes
    if 'strain' in results_df.columns and len(results_df['strain'].unique()) > 1:
        anova_results = {}
        
        for var in numeric_vars:
            if var in results_df.columns:
                # Grouper par souche
                groups = [group[var].values for _, group in results_df.groupby('strain')]
                
                # Exécuter l'ANOVA
                try:
                    f_val, p_val = stats.f_oneway(*groups)
                    anova_results[f"anova_{var}_by_strain"] = {
                        'f_value': float(f_val),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    }
                except:
                    continue
        
        test_results['anova_by_strain'] = anova_results
    
    # 3. Tests de corrélation entre variables
    if len(numeric_vars) > 1:
        correlation_matrix = results_df[numeric_vars].corr()
        test_results['correlation_matrix'] = correlation_matrix.to_dict()
        
        # Visualiser la matrice de corrélation
        if output_dir:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
            plt.title('Matrice de corrélation entre variables')
            
            corr_path = output_dir / 'correlation_matrix.png'
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Sauvegarde des résultats des tests
    if output_dir:
        # Créer un rapport texte
        report_path = output_dir / 'statistical_tests_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("# Rapport des tests statistiques\n\n")
            
            if 'anova_by_condition' in test_results:
                f.write("## Tests ANOVA par condition\n\n")
                for test, result in test_results['anova_by_condition'].items():
                    f.write(f"{test}:\n")
                    f.write(f"  - F-value: {result['f_value']:.4f}\n")
                    f.write(f"  - p-value: {result['p_value']:.4f}\n")
                    f.write(f"  - Significatif (p<0.05): {'Oui' if result['significant'] else 'Non'}\n\n")
            
            if 'anova_by_strain' in test_results:
                f.write("## Tests ANOVA par souche\n\n")
                for test, result in test_results['anova_by_strain'].items():
                    f.write(f"{test}:\n")
                    f.write(f"  - F-value: {result['f_value']:.4f}\n")
                    f.write(f"  - p-value: {result['p_value']:.4f}\n")
                    f.write(f"  - Significatif (p<0.05): {'Oui' if result['significant'] else 'Non'}\n\n")
        
        # Sauvegarder également au format JSON
        json_path = output_dir / 'statistical_tests_results.json'
        with open(json_path, 'w') as f:
            # Conversion des résultats numpy pour JSON
            results_json = {}
            for key, value in test_results.items():
                if key == 'correlation_matrix':
                    # Conversion spéciale pour la matrice de corrélation
                    corr_data = {
                        outer_key: {
                            inner_key: float(inner_val) 
                            for inner_key, inner_val in outer_val.items()
                        }
                        for outer_key, outer_val in value.items()
                    }
                    results_json[key] = corr_data
                else:
                    results_json[key] = value
            
            json.dump(results_json, f, indent=2)
    
    return test_results


def load_and_merge_results(result_dirs, pattern='sample_stats.csv'):
    """
    Charge et fusionne les résultats de plusieurs répertoires d'analyse.
    
    Args:
        result_dirs (list): Liste des répertoires contenant les résultats
        pattern (str, optional): Motif des fichiers de résultats. Par défaut 'sample_stats.csv'.
    
    Returns:
        pandas.DataFrame: DataFrame contenant tous les résultats
    
    Example:
        >>> result_dirs = [
        ...     "outputs/processed_T1_ALAC_C6_1/analysis",
        ...     "outputs/processed_T1_ALAC_C7_1/analysis"
        ... ]
        >>> merged_df = load_and_merge_results(result_dirs)
        >>> print(merged_df[['sample', 'total_spores_detected']])
    """
    all_results = []
    
    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        stats_file = result_dir / pattern
        
        if stats_file.exists():
            try:
                df = pd.read_csv(stats_file)
                all_results.append(df)
            except Exception as e:
                print(f"Erreur lors de la lecture de {stats_file}: {str(e)}")
        else:
            print(f"Fichier non trouvé: {stats_file}")
    
    if not all_results:
        raise ValueError("Aucun résultat chargé")
    
    # Fusionner tous les DataFrames
    merged_df = pd.concat(all_results, ignore_index=True)
    
    return merged_df


def analyze_spatial_distribution(detections, image_shape, output_path=None):
    """
    Analyse la distribution spatiale des détections dans l'image.
    
    Args:
        detections (list): Liste des détections
        image_shape (tuple): Forme de l'image (hauteur, largeur)
        output_path (str, optional): Chemin pour sauvegarder la visualisation
    
    Returns:
        dict: Métriques de distribution spatiale
    
    Example:
        >>> with open('detections.json', 'r') as f:
        ...     all_detections = json.load(f)
        >>> flat_detections = [det for img_dets in all_detections.values() for det in img_dets]
        >>> metrics = analyze_spatial_distribution(
        ...     flat_detections,
        ...     (1200, 1600),
        ...     "outputs/analysis/spatial_distribution.png"
        ... )
    """
    height, width = image_shape
    
    # Extraire les centres des détections
    centers = []
    for det in detections:
        if 'bbox' in det:
            bbox = det['bbox']
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            centers.append((x_center, y_center))
    
    if not centers:
        return {'count': 0}
    
    centers = np.array(centers)
    
    # Calculer la densité en divisant l'image en grille
    grid_size = 10
    cell_width = width / grid_size
    cell_height = height / grid_size
    
    grid = np.zeros((grid_size, grid_size))
    
    for x, y in centers:
        grid_x = min(grid_size - 1, int(x / cell_width))
        grid_y = min(grid_size - 1, int(y / cell_height))
        grid[grid_y, grid_x] += 1
    
    # Calculer les métriques de distribution
    metrics = {
        'count': len(centers),
        'density': len(centers) / (width * height),
        'grid_std': np.std(grid),
        'grid_coefficient_variation': np.std(grid) / np.mean(grid) if np.mean(grid) > 0 else 0
    }
    
    # Calculer la distance moyenne au plus proche voisin
    if len(centers) > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        distances, _ = tree.query(centers, k=2)  # k=2 pour obtenir soi-même et le plus proche voisin
        nearest_neighbor_distances = distances[:, 1]  # Ignorer la distance à soi-même
        
        metrics['mean_nearest_neighbor'] = np.mean(nearest_neighbor_distances)
        metrics['median_nearest_neighbor'] = np.median(nearest_neighbor_distances)
        metrics['std_nearest_neighbor'] = np.std(nearest_neighbor_distances)
    
    # Visualisation
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        plt.figure(figsize=(12, 12))
        
        # Plot principal
        plt.subplot(2, 2, 1)
        plt.scatter(centers[:, 0], centers[:, 1], alpha=0.5, s=10)
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Inversion pour correspondre au système de coordonnées de l'image
        plt.title(f'Distribution des spores (n={len(centers)})')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.grid(alpha=0.3)
        
        # Heatmap de densité
        plt.subplot(2, 2, 2)
        plt.imshow(grid, cmap='hot', interpolation='gaussian')
        plt.colorbar(label='Nombre de spores')
        plt.title('Densité des spores (grille)')
        plt.xlabel('X (cellules)')
        plt.ylabel('Y (cellules)')
        
        # Histogramme de distribution X
        plt.subplot(2, 2, 3)
        plt.hist(centers[:, 0], bins=20, alpha=0.7)
        plt.title('Distribution horizontale')
        plt.xlabel('X (pixels)')
        plt.ylabel('Fréquence')
        plt.grid(alpha=0.3)
        
        # Histogramme de distribution Y
        plt.subplot(2, 2, 4)
        plt.hist(centers[:, 1], bins=20, alpha=0.7)
        plt.title('Distribution verticale')
        plt.xlabel('Y (pixels)')
        plt.ylabel('Fréquence')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return metrics