# src/data_loader.py
import os
import pandas as pd
from .preprocess import clean_data

# base_dir para rutas de datos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_all_raw(folder):
    """Carga todos los archivos CSV de una carpeta y los concatena"""
    dfs = []
    for f in os.listdir(folder):
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(folder, f))
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Error cargando {f}: {e}")
                continue

    if not dfs:
        raise ValueError(f"No se encontraron archivos CSV válidos en {folder}")

    return pd.concat(dfs, ignore_index=True)

def load_and_preprocess_data():
    """
    Carga y preprocesa todos los datos de entrenamiento y test
    Returns:
        tuple: (df_train_cleaned, df_test_cleaned)
    """
    print("📂 Cargando datos raw...")

    # Definir rutas
    train_folder = os.path.join(BASE_DIR, 'data', 'raw', 'train')
    test_folder = os.path.join(BASE_DIR, 'data', 'raw', 'test')

    # Verificar que existen las carpetas
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"No se encontró la carpeta de entrenamiento: {train_folder}")

    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"No se encontró la carpeta de test: {test_folder}")

    # Cargar datos raw
    try:
        df_train_raw = load_all_raw(train_folder)
        df_test_raw = load_all_raw(test_folder)

        print(f"📊 Train raw shape: {df_train_raw.shape}")
        print(f"📊 Test raw shape: {df_test_raw.shape}")

    except Exception as e:
        print(f"❌ Error cargando datos raw: {e}")
        raise

    # Limpiar datos
    print("🧹 Limpiando datos...")

    try:
        df_train_clean = clean_data(df_train_raw)
        df_test_clean = clean_data(df_test_raw)

        print(f"✅ Train limpio shape: {df_train_clean.shape}")
        print(f"✅ Test limpio shape: {df_test_clean.shape}")

        # Verificar que tenemos datos
        if len(df_train_clean) == 0:
            raise ValueError("Los datos de entrenamiento están vacíos después de la limpieza")

        if len(df_test_clean) == 0:
            raise ValueError("Los datos de test están vacíos después de la limpieza")

        return df_train_clean, df_test_clean

    except Exception as e:
        print(f"❌ Error limpiando datos: {e}")
        raise

def load_processed_data():
    """
    Carga datos ya procesados si existen
    Returns:
        tuple: (df_train_final, df_test_final) o None si no existen
    """
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    train_path = os.path.join(processed_dir, 'train_final.csv')
    test_path = os.path.join(processed_dir, 'test_final.csv')

    if os.path.exists(train_path) and os.path.exists(test_path):
        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            print(f"📂 Datos procesados cargados: Train {df_train.shape}, Test {df_test.shape}")
            return df_train, df_test
        except Exception as e:
            print(f"⚠️ Error cargando datos procesados: {e}")
            return None

    return None