import pandas as pd
import os
import glob
import json # Inclus si vous l'utilisez pour l'affichage, sinon facultatif

def charger_et_preparer_instances(file_pattern: str = "instance_*.csv") -> dict:
    # --- Changement clé ici ---
    # Récupérer le chemin du répertoire où le script est exécuté
    # Ceci garantit que la recherche se fait dans le bon dossier.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(script_dir, file_pattern)
    # --------------------------
    
    instances_data = {}
    
    # Utilisation de glob pour trouver tous les fichiers correspondant au nouveau chemin
    instance_files = sorted(glob.glob(search_path))

    if not instance_files:
        print(f"Aucun fichier trouvé avec le modèle : {file_pattern}")
        # Afficher le chemin de recherche pour aider au débogage
        print(f"Chemin de recherche absolu tenté : {search_path}") 
        return instances_data
    
    print(f"Fichiers d'instances détectés : {len(instance_files)}")
    
    # Reste du code de chargement (inchangé)
    for file_path in instance_files:
        try:
            file_name = os.path.basename(file_path)
            parts = file_name.split('_')
            instance_id = parts[-1].split('.')[0]
            
            print(f"Traitement du fichier : {file_name} (ID: {instance_id})...")
            
            # Utiliser file_path qui est maintenant le chemin absolu
            df = pd.read_csv(file_path)
            
            # 1. Trier par 'window_start' en plaçant la valeur NaN (du dépôt) en premier.
            df_sorted = df.sort_values(
                by='window_start', 
                ascending=True, 
                na_position='first'
            ).reset_index(drop=True)
            
            # 2. Convertir en dictionnaire
            data_dict = df_sorted.to_dict('list')
            
            instances_data[instance_id] = data_dict
            
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
            
    print("\nDonnées de toutes les instances traitées avec succès.")
    return instances_data

# ... (Le reste de votre bloc __main__ ici) ...