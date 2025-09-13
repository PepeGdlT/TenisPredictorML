# src/data_loader.py
import os
import pandas as pd
from .preprocess import clean_data

# base_dir para rutas de datos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_all_raw(folder):
    """carga todos los archivos csv de una carpeta y los concatena"""
    dfs = []
    for f in os.listdir(folder):
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(folder, f))
                dfs.append(df)
            except Exception as e:
                print(f"error cargando {f}: {e}")
                continue

    if not dfs:
        raise ValueError(f"no se encontraron archivos csv válidos en {folder}")

    return pd.concat(dfs, ignore_index=True)

def load_and_preprocess_data():
    """
    carga y preprocesa todos los datos de entrenamiento y test
    returns:
        tuple: (df_train_cleaned, df_test_cleaned)
    """
    print("cargando datos raw...")

    # definir rutas
    train_folder = os.path.join(BASE_DIR, 'data', 'raw', 'train')
    test_folder = os.path.join(BASE_DIR, 'data', 'raw', 'test')

    # verificar que existen las carpetas
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"no se encontró la carpeta de entrenamiento: {train_folder}")

    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"no se encontró la carpeta de test: {test_folder}")

    # cargar datos raw
    try:
        df_train_raw = load_all_raw(train_folder)
        df_test_raw = load_all_raw(test_folder)

        print(f"train raw shape: {df_train_raw.shape}")
        print(f"test raw shape: {df_test_raw.shape}")

    except Exception as e:
        print(f"error cargando datos raw: {e}")
        raise

    # limpiar datos
    print("limpiando datos...")

    try:
        df_train_clean = clean_data(df_train_raw)
        df_test_clean = clean_data(df_test_raw)

        print(f"train limpio shape: {df_train_clean.shape}")
        print(f"test limpio shape: {df_test_clean.shape}")

        # verificar que tenemos datos
        if len(df_train_clean) == 0:
            raise ValueError("los datos de entrenamiento están vacíos después de la limpieza")

        if len(df_test_clean) == 0:
            raise ValueError("los datos de test están vacíos después de la limpieza")

        return df_train_clean, df_test_clean

    except Exception as e:
        print(f"error limpiando datos: {e}")
        raise

def load_processed_data():
    """
    carga datos ya procesados si existen
    returns:
        tuple: (df_train_final, df_test_final) o None si no existen
    """
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    train_path = os.path.join(processed_dir, 'train_final.csv')
    test_path = os.path.join(processed_dir, 'test_final.csv')

    if os.path.exists(train_path) and os.path.exists(test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            print(f"datos procesados cargados: train {df_train.shape}, test {df_test.shape}")
            return df_train, df_test
        except Exception as e:
            print(f"error cargando datos procesados: {e}")
            return None

    return None