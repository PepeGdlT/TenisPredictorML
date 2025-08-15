# src/features.py

from collections import defaultdict
import pandas as pd
from numpy import nan

# este módulo genera todas las features avanzadas para el modelo de predicción de tenis

def compute_elo_ratings(df, k=32, initial_elos=None):
    """
    calcula el elo global de cada jugador usando la fórmula estándar
    """
    elos = defaultdict(lambda: 1500)
    if initial_elos:
        elos.update(initial_elos)

    elo_winner_list = []
    elo_loser_list = []
    # recorre cada partido y actualiza el elo de los jugadores
    for _, row in df.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]

        R_w = elos[winner]
        R_l = elos[loser]

        # Probabilidad de que el ganador gane
        expected_w = 1 / (1 + 10 ** ((R_l - R_w) / 400))
        expected_l = 1 / (1 + 10 ** ((R_w - R_l) / 400))

        # S = 1 para el ganador, S = 0 para el perdedor
        S_w = 1
        S_l = 0

        # Actualización ELO
        elos[winner] = R_w + k * (S_w - expected_w)
        elos[loser] = R_l + k * (S_l - expected_l)

        elo_winner_list.append(R_w)
        elo_loser_list.append(R_l)

    df["elo_winner"] = elo_winner_list
    df["elo_loser"] = elo_loser_list
    df["elo_diff"] = df["elo_winner"] - df["elo_loser"]

    return df, elos


def compute_surface_elo(df, k=32, initial_elos=None):
    """
    calcula el elo por tipo de superficie (hard/clay/grass/carpet)
    """
    elos = defaultdict(lambda: 1500)
    if initial_elos:
        elos.update(initial_elos)

    elo_winner_list = []
    elo_loser_list = []
    for _, row in df.iterrows():
        surface = row.get("surface", "Unknown")
        key_winner = f"{row['winner_name']}_{surface}"
        key_loser = f"{row['loser_name']}_{surface}"

        elo_w = elos[key_winner]
        elo_l = elos[key_loser]

        expected_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        expected_l = 1 - expected_w

        elos[key_winner] += k * (1 - expected_w)
        elos[key_loser] += k * (0 - expected_l)

        elo_winner_list.append(elo_w)
        elo_loser_list.append(elo_l)

    df["surface_elo_winner"] = elo_winner_list
    df["surface_elo_loser"] = elo_loser_list
    df["surface_elo_diff"] = df["surface_elo_winner"] - df["surface_elo_loser"]

    return df, elos


def compute_h2h(df, initial_h2h=None):
    """
    calcula el historial head-to-head entre jugadores usando log(1+h2h_count) y balance logarítmico
    Permite continuar desde un historial previo (initial_h2h)
    """
    import numpy as np
    from collections import defaultdict
    h2h_count_list = []
    h2h_balance_list = []
    if initial_h2h is not None:
        h2h_history = defaultdict(lambda: {"count": 0, "winner_wins": 0})
        h2h_history.update({tuple(eval(k)): v for k, v in initial_h2h.items()})
    else:
        h2h_history = defaultdict(lambda: {"count": 0, "winner_wins": 0})
    for _, row in df.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        pair = tuple(sorted([winner, loser]))
        history = h2h_history[pair]
        count = history["count"]
        winner_wins = history["winner_wins"]

        # log(1 + count) para suavizar la importancia
        log_count = np.log1p(count)

        # balance logarítmico: log(1 + victorias ganador) - log(1 + victorias perdedor)
        if count == 0:
            log_balance = 0
        else:
            loser_wins = count - winner_wins
            log_balance = np.log1p(winner_wins) - np.log1p(loser_wins)

        h2h_count_list.append(log_count)
        h2h_balance_list.append(log_balance)

        # actualizar historial para próximos enfrentamientos
        h2h_history[pair]["count"] += 1
        h2h_history[pair]["winner_wins"] += 1  # el winner siempre suma

    df["h2h_count"] = h2h_count_list
    df["h2h_balance"] = h2h_balance_list

    # Convertir las claves a string para serializar
    h2h_history_serializable = {str(k): v for k, v in h2h_history.items()}
    return df, h2h_history_serializable


def add_all_features(df, initial_global_elos=None, initial_surface_elos=None, initial_h2h=None):
    """
    aplica todas las funciones de ingeniería de features
    """
    df, final_global_elos = compute_elo_ratings(df, initial_elos=initial_global_elos)
    df, final_surface_elos = compute_surface_elo(df, initial_elos=initial_surface_elos)
    df, final_h2h = compute_h2h(df, initial_h2h=initial_h2h)
    df = add_enhanced_elo_features(df)
    return df, final_global_elos, final_surface_elos, final_h2h


def add_enhanced_elo_features(df):
    """
    agrega features adicionales basadas en elo y ranking para mejorar la predicción
    """
    # Diferencias de ELO (ya calculadas en las funciones base)
    # df['elo_diff'] y df['surface_elo_diff'] ya existen

    # 🔥 FEATURES MEJORADAS PARA MAYOR ACCURACY

    # 1. ELO clasificado por rangos más granulares
    df['elo_advantage'] = df['elo_diff'].apply(
        lambda x: 2 if x > 200 else (1 if x > 50 else (-1 if x < -50 else (-2 if x < -200 else 0)))
    )
    df['surface_elo_advantage'] = df['surface_elo_diff'].apply(
        lambda x: 2 if x > 200 else (1 if x > 50 else (-1 if x < -50 else (-2 if x < -200 else 0)))
    )

    # 2. Features de interacción entre ELO global y superficie
    df['elo_surface_interaction'] = df['elo_diff'] * df['surface_elo_diff'] / 10000  # Normalizado
    df['elo_consistency'] = abs(df['elo_diff'] - df['surface_elo_diff'])  # Consistencia entre ELOs

    # 3. Features de momentum/forma reciente (aproximada por ranking)
    if 'winner_rank' in df.columns and 'loser_rank' in df.columns:
        # Convertir a numérico para evitar errores de tipo
        df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce')
        df['loser_rank'] = pd.to_numeric(df['loser_rank'], errors='coerce')
        # Diferencia de ranking más granular
        df['rank_diff'] = df['loser_rank'] - df['winner_rank']  # Positivo = winner mejor ranking
        df['rank_advantage'] = df['rank_diff'].apply(
            lambda x: 2 if x > 50 else (1 if x > 10 else (-1 if x < -10 else (-2 if x < -50 else 0))) if pd.notnull(x) else 0
        )

        # Ratio de rankings (evitar división por cero)
        df['rank_ratio'] = (df['loser_rank'] + 1) / (df['winner_rank'] + 1)

        # Feature de "sorpresa" - cuando el ranking no coincide con ELO
        df['elo_rank_mismatch'] = ((df['elo_diff'] > 0) & (df['rank_diff'] < 0)).astype(int) - \
                                 ((df['elo_diff'] < 0) & (df['rank_diff'] > 0)).astype(int)
    else:
        df['rank_diff'] = 0
        df['rank_advantage'] = 0
        df['rank_ratio'] = 1
        df['elo_rank_mismatch'] = 0

    # 4. Features categóricas de nivel de juego
    df['elo_tier_winner'] = pd.cut(df['elo_winner'],
                                  bins=[0, 1400, 1600, 1800, 2000, float('inf')],
                                  labels=[0, 1, 2, 3, 4]).astype(int)
    df['elo_tier_loser'] = pd.cut(df['elo_loser'],
                                 bins=[0, 1400, 1600, 1800, 2000, float('inf')],
                                 labels=[0, 1, 2, 3, 4]).astype(int)
    df['tier_diff'] = df['elo_tier_winner'] - df['elo_tier_loser']

    # 5. Features de competitividad del match
    df['match_competitiveness'] = 1 / (1 + abs(df['elo_diff']) / 100)  # Más competitivo = valor más alto
    df['is_upset_potential'] = ((abs(df['elo_diff']) > 150) & (abs(df['rank_diff']) < 20)).astype(int)

    return df
