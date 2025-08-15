# src/train_model.py
import pandas as pd
import os
from model import train_model
from sklearn.metrics import accuracy_score
from data_loader import BASE_DIR
from utils import make_dual_rows, fillna_features, print_feature_availability

# este script entrena el modelo principal y muestra métricas e importancia de features

features_train_path = os.path.join(BASE_DIR, "data", "processed", "features_train.csv")
features_test_path = os.path.join(BASE_DIR, "data", "processed", "features_test.csv")

# cargar datos
print("cargando datos de entrenamiento y test...")
df_train = pd.read_csv(features_train_path)
df_test = pd.read_csv(features_test_path)

df_train = make_dual_rows(df_train)
df_test = make_dual_rows(df_test)

# features expandidas con nuevas características
feature_cols = [
    "elo_winner", "elo_loser", "elo_diff",
    "surface_elo_winner", "surface_elo_loser", "surface_elo_diff",
    "elo_advantage", "surface_elo_advantage",
    "elo_surface_interaction", "elo_consistency",
    "rank_diff", "rank_advantage", "rank_ratio", "elo_rank_mismatch",
    "elo_tier_winner", "elo_tier_loser", "tier_diff",
    "match_competitiveness", "is_upset_potential",
    "h2h_count", "h2h_balance"
]

# verificar qué features están disponibles
available_features, missing_features = print_feature_availability(df_train, feature_cols)
feature_cols = available_features

df_train = fillna_features(df_train, feature_cols)
df_test = fillna_features(df_test, feature_cols)

X_train = df_train[feature_cols]
y_train = df_train["target"]
X_test = df_test[feature_cols]
y_test = df_test["target"]

print(f"train: {len(X_train)} muestras, test: {len(X_test)} muestras")
print(f"distribución train: {y_train.value_counts().to_dict()}")
print(f"distribución test: {y_test.value_counts().to_dict()}")

model, val_acc = train_model(X_train, y_train)
print(f"accuracy validación interna (80/20): {val_acc:.4f}")

y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"accuracy en test (2023+): {test_acc:.4f}")

# mostrar importancia de features
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("importancia de features:")
print(feature_importance_df)
