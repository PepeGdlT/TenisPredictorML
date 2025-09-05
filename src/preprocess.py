# src/preprocess.py

import pandas as pd


def clean_data(df):
    """aplica limpieza básica al dataset."""
    # elimina columnas poco informativas (ajústalo según tus necesidades)
    columns_to_drop = ['winner_entry', 'loser_entry']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # convertir fechas
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')

    # eliminar duplicados
    df = df.drop_duplicates()

    # rellenar valores nulos simples
    df = df.fillna({'surface': 'Unknown', 'tourney_level': 'U'})

    return df
