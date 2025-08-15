# src/generate_features_test.py
from data_loader import load_test_data, BASE_DIR
from preprocess import clean_data
from features import add_all_features
import os
import json

# este script genera las features para el dataset de test y las guarda en un csv

processed_data_path = os.path.join(BASE_DIR, "data", "processed")
try:
    with open(os.path.join(processed_data_path, "final_global_elos.json"), "r") as f:
        initial_global_elos = json.load(f)
    with open(os.path.join(processed_data_path, "final_surface_elos.json"), "r") as f:
        initial_surface_elos = json.load(f)
    print("ELOs iniciales cargados desde el entrenamiento.")
except FileNotFoundError:
    initial_global_elos = None
    initial_surface_elos = None
    print("[Aviso] No se encontraron archivos de ELO iniciales. Se usará el ELO por defecto (1500).")

# Cargar historial h2h si existe
h2h_path = os.path.join(processed_data_path, "final_h2h.json")
try:
    with open(h2h_path, "r") as f:
        initial_h2h = json.load(f)
    print("Historial H2H inicial cargado.")
except FileNotFoundError:
    initial_h2h = None
    print("[Aviso] No se encontró historial H2H inicial. Se parte de cero.")

df = load_test_data()
df = clean_data(df)
# Pasar los ELOs y h2h iniciales al generador de features
df, final_global_elos, final_surface_elos, final_h2h = add_all_features(
    df,
    initial_global_elos=initial_global_elos,
    initial_surface_elos=initial_surface_elos,
    initial_h2h=initial_h2h
)

output_path = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(output_path, exist_ok=True)

# Guardar las features
df.to_csv(os.path.join(output_path, "features_test.csv"), index=False)
print("features de test guardadas en:", os.path.join(output_path, "features_test.csv"))

# Guardar los ELOs finales actualizados
# Guardar historial h2h actualizado
with open(h2h_path, "w") as f:
    json.dump(final_h2h, f)
print(f"Historial H2H actualizado guardado en: {h2h_path}")
with open(os.path.join(output_path, "final_global_elos.json"), "w") as f:
    json.dump(final_global_elos, f)
with open(os.path.join(output_path, "final_surface_elos.json"), "w") as f:
    json.dump(final_surface_elos, f)
print("ELOs finales actualizados y guardados en los archivos JSON")
