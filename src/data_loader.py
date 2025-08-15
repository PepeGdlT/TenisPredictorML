# src/data_loader.py
import os
import pandas as pd

# este módulo carga los datasets de partidos de tenis desde archivos csv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_dataset(folder_path):
    df_list = []
    # recorre todos los archivos csv en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            full_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(full_path)
                if not df.empty:
                    df_list.append(df)
                else:
                    print(f"[aviso] archivo vacío: {filename}")
            except Exception as e:
                print(f"[error] no se pudo leer {filename}: {e}")
    if not df_list:
        raise ValueError(f"no se pudieron cargar archivos csv útiles en {folder_path}")
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def load_train_data():
    # carga los datos de entrenamiento
    return load_dataset(os.path.join(BASE_DIR, "data/raw/train"))

def load_test_data():
    # carga los datos de test
    return load_dataset(os.path.join(BASE_DIR, "data/raw/test"))
