# src/generate_features_train.py
from data_loader import load_train_data, BASE_DIR
from preprocess import clean_data
from features import add_all_features
import os
import json

# este script genera las features para el dataset de entrenamiento y las guarda en un csv

df = load_train_data()
df = clean_data(df)
df, final_global_elos, final_surface_elos, final_h2h = add_all_features(df)

output_path = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(output_path, exist_ok=True)

df.to_csv(os.path.join(output_path, "features_train.csv"), index=False)

print("features de entrenamiento guardadas en:", os.path.join(output_path, "features_train.csv"))

# Guardar ELOs finales para usarlos en el test set
with open(os.path.join(output_path, "final_global_elos.json"), "w") as f:
    json.dump(dict(final_global_elos), f)

with open(os.path.join(output_path, "final_surface_elos.json"), "w") as f:
    json.dump(dict(final_surface_elos), f)

    # Guardar historial h2h final
with open(os.path.join(output_path, "final_h2h.json"), "w") as f:
    json.dump(final_h2h, f)
print("Historial H2H final guardado para el conjunto de test.")
print("ELOs finales guardados para el conjunto de test.")
