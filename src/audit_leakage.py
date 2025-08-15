# src/audit_leakage.py
"""
Script para auditar fuga de información en features: correlación con el target.
"""
import pandas as pd
import numpy as np

def audit_feature_leakage(X, y):
    """
    Calcula la correlación absoluta de cada feature con el target y avisa si alguna es sospechosa (>0.95).
    """
    corrs = {}
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            cor = np.corrcoef(X[col], y)[0, 1]
            corrs[col] = cor
    corrs = pd.Series(corrs).abs().sort_values(ascending=False)
    print("\nCorrelación absoluta de cada feature con el target:")
    print(corrs)
    print("\nFeatures con correlación > 0.95 (posible fuga de información):")
    print(corrs[corrs > 0.95])
    return corrs

if __name__ == "__main__":
    # USO DIRECTO: Cambia aquí el path y la columna target según tu dataset
    features_path = '../data/processed/features_test_with_target.csv'  # Cambia por el path que quieras
    target_col = 'target'  # Cambia por el nombre de la columna target si es distinto
    df = pd.read_csv(features_path)
    if target_col not in df.columns:
        print(f"Columna target '{target_col}' no encontrada en el archivo.")
        exit(1)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    audit_feature_leakage(X, y)
