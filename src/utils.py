# src/utils.py

import os
import pandas as pd

def ensure_directory_exists(path):
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)

def fillna_features(df, feature_cols, value=0):
    # rellena NaN en las columnas de features indicadas
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df

def print_feature_availability(df, feature_cols):
    # imprime qué features están y cuáles faltan
    available = [col for col in feature_cols if col in df.columns]
    missing = [col for col in feature_cols if col not in df.columns]
    print(f"features disponibles: {len(available)}")
    print(f"features faltantes: {len(missing)}")
    if missing:
        print(f"faltantes: {missing}")
    return available, missing

def make_dual_rows(df):
    # duplica filas para balancear el dataset y alterna los targets
    df1 = df.copy()
    df1["target"] = 1

    df2 = df.copy()
    df2["target"] = 0
    for col in ["elo_winner", "elo_loser", "surface_elo_winner", "surface_elo_loser"]:
        tmp = df2[col.replace("winner", "loser")]
        df2[col.replace("winner", "loser")] = df2[col]
        df2[col] = tmp

    # intercambiar features adicionales también
    for feature in [
        "elo_diff", "surface_elo_diff", "elo_advantage", "surface_elo_advantage",
        "rank_diff", "rank_advantage", "elo_surface_interaction", "elo_rank_mismatch", "tier_diff"
    ]:
        if feature in df2.columns:
            df2[feature] = -df2[feature]
    if "rank_ratio" in df2.columns:
        df2["rank_ratio"] = 1 / df2["rank_ratio"]
    if "h2h_balance" in df2.columns:
        df2["h2h_balance"] = -df2["h2h_balance"]
    # h2h_count se mantiene igual
    return pd.concat([df1, df2], ignore_index=True)
