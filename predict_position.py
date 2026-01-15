import timesfm
import pandas as pd
import numpy as np
import matplotlib
# Utiliser un backend non interactif pour les environnements sans GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FICHIER_CSV = 'data.csv'      # Nom de votre fichier
# Nom logique de la colonne à prédire (insensible à la casse, on cherchera une correspondance)
COLONNE_A_PREDIRE = 'Position'  # Exemple: 'Clicks' ou 'Impressions'
JOURS_A_PREDIRE = 365       # Combien de jours voulez-vous prédire ?

# 1. Chargement des données
print(f"Lecture de {FICHIER_CSV}...")
try:
    df = pd.read_csv(FICHIER_CSV)

    # Normaliser les noms de colonnes (suppression des espaces, casse uniforme)
    original_columns = df.columns.tolist()
    normalized = {c: c.strip() for c in original_columns}
    df = df.rename(columns=normalized)

    # Détection de la colonne date (insensible à la casse)
    date_col = None
    for c in df.columns:
        if c.lower() == 'date':
            date_col = c
            break
    if date_col is None:
        raise ValueError("Aucune colonne 'Date' trouvée. Assurez-vous que le CSV contient une colonne 'Date'.")

    # Conversion en datetime et tri
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().all():
        raise ValueError(f"Impossible de parser les dates dans la colonne '{date_col}'.")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Détection de la colonne à prédire (insensible à la casse)
    target_col = None
    for c in df.columns:
        if c.lower() == COLONNE_A_PREDIRE.lower():
            target_col = c
            break
    if target_col is None:
        raise ValueError(
            f"Colonne à prédire '{COLONNE_A_PREDIRE}' introuvable. Colonnes disponibles: {df.columns.tolist()}"
        )

    # Nettoyage des valeurs manquantes
    # Gérer les pourcentages (ex: "0.81%" -> 0.81)
    if target_col.lower() == 'ctr':
        df[target_col] = df[target_col].astype(str).str.rstrip('%').astype(float).fillna(0)
    else:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)

    # Extraction de la série temporelle sous forme de tableau numpy
    series_historique = df[target_col].values
    print(f"Historique chargé : {len(series_historique)} jours trouvés.")

except Exception as e:
    print(f"Erreur de lecture du CSV : {e}")
    exit()

# 2. Initialisation de TimesFM
prediction = None
print("Chargement du modèle TimesFM...")
try:
    # API mise à jour: utiliser TimesFmHparams et TimesFmCheckpoint
    hparams = timesfm.TimesFmHparams(
        context_len=512,
        horizon_len=JOURS_A_PREDIRE,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
    )
    checkpoint = timesfm.TimesFmCheckpoint(
        version="jax",  # utiliser la version JAX (poids disponibles sur le repo)
        huggingface_repo_id="google/timesfm-1.0-200m",
    )
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

    # 3. Prédiction TimesFM
    print("Calcul de la prévision en cours (TimesFM)...")
    means, quantiles = tfm.forecast([series_historique], freq=[0])
    prediction = means[0]
except Exception as e:
    print(f"TimesFM indisponible, bascule sur une prévision simple: {e}")
    # Fallback simple: répéter les 7 derniers jours comme prévision des 7 prochains
    recent = df[target_col].tail(JOURS_A_PREDIRE).values
    if len(recent) < JOURS_A_PREDIRE:
        # Si l'historique est trop court, utiliser la moyenne
        mean_val = float(df[target_col].mean())
        prediction = np.array([mean_val] * JOURS_A_PREDIRE)
    else:
        prediction = recent.copy()

# 4. Création des dates futures
last_date = df[date_col].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=JOURS_A_PREDIRE)

# 5. Sauvegarde des résultats
df_result = pd.DataFrame({
    'date': future_dates,
    'prediction': prediction
})

print("\n--- PRÉDICTIONS ---")
print(df_result)

# Sauvegarde dans un fichier Excel/CSV
df_result.to_csv('resultat_prediction_Position.csv', index=False)
print("\nLes résultats ont été sauvegardés dans 'resultat_prediction_Position.csv'")

# 6. Visualisation
plt.figure(figsize=(12, 6))
plt.plot(df[date_col], df[target_col], label='Historique')
plt.plot(future_dates, prediction, label='Prédiction Eskimoz', color='red', linestyle='--', marker='o')
plt.title(f"Prévision pour : {target_col}")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)
# Sauvegarder la figure au lieu d'afficher une fenêtre GUI
plt.tight_layout()
plt.savefig('Position.png', dpi=150)
print("La figure a été sauvegardée dans 'Position.png'.")