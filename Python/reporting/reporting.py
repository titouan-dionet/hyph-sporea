# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:15:12 2025

@author: Titouan Dionet
"""

import os
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import json


def generate_quarto_report(results_data, output_file, title="Rapport d'analyse de spores", template=None):
    """
    Génère un rapport Quarto à partir des résultats d'analyse.
    
    Args:
        results_data (dict): Données des résultats d'analyse
        output_file (str): Chemin du fichier de sortie (.html, .pdf, .docx)
        title (str, optional): Titre du rapport
        template (str, optional): Chemin du modèle Quarto personnalisé
    
    Returns:
        str: Chemin du rapport généré
    """
    # Vérifier si Quarto est installé
    try:
        subprocess.run(["quarto", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("Quarto n'est pas installé ou n'est pas accessible dans le PATH")
    
    # Sérialiser les données en JSON pour les insérer dans le fichier QMD
    results_json = json.dumps(results_data)
    
    # Créer un fichier Quarto temporaire
    with tempfile.NamedTemporaryFile(suffix='.qmd', delete=False) as tmp:
        tmp_path = tmp.name
        
        # Écrire le contenu du fichier Quarto
        content = f"""---
            title: "{title}"
            format: 
              html:
                toc: true
                toc-depth: 3
                theme: cosmo
                code-fold: true
            execute:
              echo: false
            ---
            
            ```{{python}}
            # Importer les bibliothèques nécessaires
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            import json
            
            # Charger les données depuis le JSON intégré
            results_data = json.loads('''{results_json}''')
            
            # Créer un DataFrame pour l'analyse
            results_df = pd.DataFrame(list(results_data.values()))
            
            # Synthèse des résultats
            # Afficher le tableau récapitulatif
            if len(results_df) > 0:
                display(results_df[['sample', 'strain', 'condition', 'total_spores_detected']].style.highlight_max(
                    subset=['total_spores_detected']))
            else:
                print("Aucune donnée disponible pour l'analyse")
                
            # Analyse par échantillon
            # Créer un graphique du nombre de spores par échantillon
            if len(results_df) > 0 and 'total_spores_detected' in results_df.columns:
                plt.figure(figsize=(10, 6))
                plt.bar(results_df['sample'], results_df['total_spores_detected'])
                plt.title('Nombre de spores détectées par échantillon')
                plt.xlabel('Échantillon')
                plt.ylabel('Nombre de spores')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            else:
                print("Données insuffisantes pour générer le graphique")
                
            # Graphique par condition
            if len(results_df) > 0 and 'condition' in results_df.columns and 'total_spores_detected' in results_df.columns:
                conditions = results_df['condition'].unique()
                if len(conditions) > 1:
                    plt.figure(figsize=(10, 6))
                    
                    for condition in conditions:
                        condition_data = results_df[results_df['condition'] == condition]
                        plt.bar(condition_data['sample'], condition_data['total_spores_detected'], label=condition)
                    
                    plt.title('Nombre de spores par échantillon et condition')
                    plt.xlabel('Échantillon')
                    plt.ylabel('Nombre de spores')
                    plt.xticks(rotation=45, ha='right')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                else:
                    print("Une seule condition disponible - graphique par condition non généré")
            else:
                print("Données insuffisantes pour générer le graphique par condition")
                
            # Analyse morphologique
            # Distribution des tailles
            if len(results_df) > 0 and 'avg_area' in results_df.columns:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                samples = results_df['sample'].tolist()
                
                # Créer des données pour le boxplot
                data_to_plot = []
                for sample in samples:
                    sample_data = results_df[results_df['sample'] == sample]['avg_area']
                    if not sample_data.empty:
                        data_to_plot.append(sample_data)
                
                if data_to_plot:
                    plt.boxplot(data_to_plot)
                    plt.title('Distribution des surfaces moyennes')
                    plt.xlabel('Échantillon')
                    plt.xticks(range(1, len(samples) + 1), samples, rotation=45, ha='right')
                    plt.ylabel('Surface moyenne (pixels²)')
                
                # Graphique largeur/hauteur si disponibles
                plt.subplot(1, 2, 2)
                if 'avg_width' in results_df.columns and 'avg_height' in results_df.columns:
                    for i, sample in enumerate(results_df['sample']):
                        row_idx = results_df.index[results_df['sample'] == sample][0]
                        plt.scatter(
                            results_df.loc[row_idx, 'avg_width'], 
                            results_df.loc[row_idx, 'avg_height'],
                            label=sample,
                            s=100
                        )
                    
                    plt.title('Dimensions moyennes des spores')
                    plt.xlabel('Largeur moyenne (pixels)')
                    plt.ylabel('Hauteur moyenne (pixels)')
                    plt.legend()
                
                plt.tight_layout()
                plt.show()
            else:
                print("Données morphologiques insuffisantes pour générer les graphiques")
            
            # Conclusion
            # Générer des conclusions basées sur les données disponibles
            if len(results_df) > 0 and 'total_spores_detected' in results_df.columns:
                total_spores = results_df['total_spores_detected'].sum()
                
                if total_spores > 0:
                    max_sample_idx = results_df['total_spores_detected'].idxmax()
                    max_sample = results_df.loc[max_sample_idx, 'sample']
                    max_spores = results_df['total_spores_detected'].max()
                    
                    print(f"Cette analyse a permis d'identifier les caractéristiques des spores dans {len(results_df)} échantillons différents.")
                    print(f"Le nombre total de spores détectées est de {total_spores}.")
                    print(f"L'échantillon contenant le plus de spores est \"{max_sample}\" avec {max_spores} spores.")
                    
                    if 'condition' in results_df.columns:
                        conditions = results_df['condition'].unique()
                        print(f"L'analyse a couvert {len(conditions)} conditions différentes: {', '.join(conditions)}.")
                else:
                    print("Aucune spore n'a été détectée dans les échantillons analysés.")
            else:
                print("Données insuffisantes pour générer des conclusions.")
            """
        tmp.write(content.encode('utf-8'))
        
        # Compiler le document Quarto
        output_file = Path(output_file)
        cmd = ["quarto", "render", tmp_path, "--output", str(output_file)]
        
        if template:
            cmd.extend(["--template", template])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Rapport généré avec succès: {output_file}")
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode('utf-8')
            raise RuntimeError(f"Erreur lors de la génération du rapport Quarto: {error_output}")
        finally:
            # Supprimer le fichier temporaire
            os.unlink(tmp_path)
        
        return str(output_file)

