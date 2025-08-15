# src/generate_features_test_with_target.py
import pandas as pd
import os
from data_loader import BASE_DIR
from utils import make_dual_rows

# Cargar el archivo de features ya generadas (sin target)
features_path = os.path.join(BASE_DIR, "data", "processed", "features_test.csv")
df = pd.read_csv(features_path)

# Aplicar make_dual_rows para añadir la columna target
print("Generando dataset balanceado con columna target...")
df_balanced = make_dual_rows(df)

# Guardar el nuevo archivo con la columna target
output_path = os.path.join(BASE_DIR, "data", "processed", "features_test_with_target.csv")
df_balanced.to_csv(output_path, index=False)
print(f"Archivo guardado en: {output_path}")

