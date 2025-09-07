"""
features.py
módulo robusto y refactorizado para ingeniería de features de partidos de tenis.
incluye cálculo de elo global, elo por superficie, h2h (global/superficie/reciente),
estadísticas históricas (medias), forma reciente, fatiga/ritmo, volatilidad elo,
features avanzadas y pipeline principal.
"""
import math
import warnings

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

from collections import defaultdict, deque


# Función auxiliar para validación segura de NaN
def safe_isnan(value):
    """
    Función segura para verificar si un valor es NaN
    """
    try:
        return pd.isna(value) or (isinstance(value, (int, float)) and np.isnan(value))
    except (TypeError, ValueError):
        return False

def safe_to_numeric(value, default=np.nan):
    """
    Convierte un valor a numérico de forma segura
    """
    if safe_isnan(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


PCA_GROUPS = {
    # ===============================
    # Stats históricas / rendimiento por jugador
    # ===============================
    "ace_df": [
        "p1_ace_mean", "p2_ace_mean",
        "p1_df_mean", "p2_df_mean",
        "p1_ace_surface_mean", "p2_ace_surface_mean"
    ],
    "serve": [
        "p1_svpt_mean", "p2_svpt_mean",
        "p1_1stIn_mean", "p2_1stIn_mean",
        "p1_1stWon_mean", "p2_1stWon_mean",
        "p1_2ndWon_mean", "p2_2ndWon_mean"
    ],
    "breakpoints": [
        "p1_bpSaved_mean", "p2_bpSaved_mean",
        "p1_bpFaced_mean", "p2_bpFaced_mean"
    ],

    # ===============================
    # Elo y probabilidades
    # ===============================
    "elo_core": [
        "elo_prob","surface_elo_prob",
        "elo_diff_clip","surface_elo_diff_clip",
        "elo_diff_sq","surface_elo_diff_sq",
        "elo_surface_interaction","elo_consistency",
        "elo_vol5_diff","surface_elo_vol5_diff"
    ],

    # ===============================
    # Ranking
    # ===============================
    "ranking": [
        "rank_diff","rank_ratio","log_rank_ratio","elo_rank_mismatch"
    ],

    # ===============================
    # H2H
    # ===============================
    "h2h": [
        "h2h_count","h2h_balance",
        "h2h_surface_count","h2h_surface_balance",
        "h2h_recent3_balance"
    ],

    # ===============================
    # Forma reciente, fatiga y ritmo MEJORADO
    # ===============================
    "recent_form": [
        "form_last5_diff","form_last10_diff",
        "surface_wr_all_diff","surface_wr_last5_diff",
        "streak_wins_diff"
    ],

    "fatigue": [
        "matches_recent_weighted", "days_since_last_diff_scaled", "elo_matches_interaction",
        "fatigue_immediate", "fatigue_short_term", "recovery_factor", "elo_fatigue_immediate"
    ],
    
    # Agrupamos features raw de fatiga para reducir ruido
    "fatigue_raw": [
        "matches_last3d_diff", "matches_last7d_diff", "matches_last14d_diff", "days_since_last_diff",
        "matches_last3d_diff_scaled", "matches_last7d_diff_scaled", "matches_last14d_diff_scaled",
        "days_since_surface_diff", "days_since_surface_diff_scaled", "surface_matches_30d_diff"
    ],
    
    # Features individuales por jugador (para reducir dimensionalidad)
    "fatigue_individual": [
        "p1_matches_last3d", "p2_matches_last3d",
        "p1_matches_last7d", "p2_matches_last7d", 
        "p1_matches_last14d", "p2_matches_last14d",
        "p1_days_since_last", "p2_days_since_last",
        "p1_days_since_surface", "p2_days_since_surface",
        "p1_surface_matches_last30d", "p2_surface_matches_last30d"
    ],
    
    # ===============================
    # Contexto del partido
    # ===============================
    "match_context": [
        "match_competitiveness"
    ]
}



# ===============================
# utilidades
# ===============================
def _as_date_series(df):
    """intenta detectar columna de fecha y devolverla como serie datetime (o None)."""
    for c in ['match_date', 'tourney_date', 'date', 'tourney_date_raw']:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors='coerce')
            if s.notna().any():
                return s
    return None

def _safe_div(a, b, default=np.nan):
    return a / b if (b is not None and b not in [0, np.nan] and b == b) else default

def _rolling_mean(lst, k):
    if not lst:
        return np.nan
    return np.mean(lst[-k:]) if len(lst) >= k else np.mean(lst)

def _rolling_sum(lst, k):
    if not lst:
        return 0
    return np.sum(lst[-k:]) if len(lst) >= k else np.sum(lst)

# ===============================
# elo global (con volatilidad previa)
# ===============================

def _k_adaptive(n_matches, k_min=18.0, k_max=32.0, half_life=50.0):
    """K(n) = k_min + (k_max-k_min)*exp(-n/half_life)"""
    if n_matches is None or n_matches < 0:
        n_matches = 0
    return float(k_min + (k_max - k_min) * math.exp(-float(n_matches)/float(half_life)))

def compute_elo_ratings(
    df,
    k_mode="adaptive",          # "fixed" o "adaptive"
    k=32,                       # usado si k_mode="fixed"
    k_min=18.0, k_max=32.0, half_life=50.0,  # usado si k_mode="adaptive"
    initial_elos=None,
    initial_match_counts=None   # dict opcional {player: num_partidos_previos}
):
    elos = defaultdict(lambda: 1500.0)
    if initial_elos: elos.update(initial_elos)

    match_counts = defaultdict(int)
    if initial_match_counts:
        for p, n in initial_match_counts.items():
            match_counts[p] = int(n)

    elo_delta_hist = defaultdict(list)
    elo_p1_list, elo_p2_list = [], []
    vol5_p1_list, vol5_p2_list = [], []

    for _, row in df.iterrows():
        p1, p2 = row["player_1"], row["player_2"]
        R1, R2 = elos[p1], elos[p2]

        # volatilidad previa (tu lógica existente)
        vol5_p1 = _rolling_mean([abs(d) for d in elo_delta_hist[p1]], 5)
        vol5_p2 = _rolling_mean([abs(d) for d in elo_delta_hist[p2]], 5)

        # prob. esperada
        exp1 = 1.0 / (1.0 + 10.0 ** ((R2 - R1) / 400.0))
        S1 = row.get("target", 1.0)
        S2 = 1.0 - S1

        # K por jugador (adaptativo o fijo)
        if k_mode == "adaptive":
            K1 = _k_adaptive(match_counts[p1], k_min=k_min, k_max=k_max, half_life=half_life)
            K2 = _k_adaptive(match_counts[p2], k_min=k_min, k_max=k_max, half_life=half_life)
        else:
            K1 = K2 = float(k)

        new_R1 = R1 + K1 * (S1 - exp1)
        new_R2 = R2 + K2 * (S2 - (1.0 - exp1))

        # guardar “ratings pre-partido”
        elo_p1_list.append(R1)
        elo_p2_list.append(R2)
        vol5_p1_list.append(vol5_p1)
        vol5_p2_list.append(vol5_p2)

        # actualizar
        elo_delta_hist[p1].append(new_R1 - R1)
        elo_delta_hist[p2].append(new_R2 - R2)
        elos[p1] = new_R1
        elos[p2] = new_R2

        match_counts[p1] += 1
        match_counts[p2] += 1

    df["elo_p1"] = elo_p1_list
    df["elo_p2"] = elo_p2_list
    df["elo_diff"] = df["elo_p1"] - df["elo_p2"]
    df["elo_vol5_p1"] = vol5_p1_list
    df["elo_vol5_p2"] = vol5_p2_list
    df["elo_vol5_diff"] = df["elo_vol5_p1"] - df["elo_vol5_p2"]

    return df, dict(elos), dict(match_counts)


def compute_surface_elo(
    df,
    k_mode="adaptive",
    k=32,
    k_min=18.0, k_max=32.0, half_life=50.0,
    alpha_surface=0.7,                  # peso de partidos globales en n_eff
    initial_elos=None,
    initial_match_counts_global=None,   # {player: n_global}
    initial_match_counts_surface=None   # {(player,surface): n_surface}
):
    elos = defaultdict(lambda: 1500.0)
    if initial_elos: elos.update(initial_elos)

    n_global = defaultdict(int)
    if initial_match_counts_global:
        for p, n in initial_match_counts_global.items():
            n_global[p] = int(n)

    n_surface = defaultdict(int)
    if initial_match_counts_surface:
        for k2, n in initial_match_counts_surface.items():
            n_surface[k2] = int(n)

    elo_delta_hist = defaultdict(list)
    elo_p1_list, elo_p2_list = [], []
    vol5_p1_list, vol5_p2_list = [], []

    for _, row in df.iterrows():
        surface = row.get("surface", "Unknown")
        p1, p2 = row["player_1"], row["player_2"]
        key1, key2 = f"{p1}_{surface}", f"{p2}_{surface}"

        R1, R2 = elos[key1], elos[key2]
        vol5_p1 = _rolling_mean([abs(d) for d in elo_delta_hist[key1]], 5)
        vol5_p2 = _rolling_mean([abs(d) for d in elo_delta_hist[key2]], 5)

        exp1 = 1.0 / (1.0 + 10.0 ** ((R2 - R1) / 400.0))
        S1 = row.get("target", 1.0)
        S2 = 1.0 - S1

        # n efectivos combinando global y superficie
        n1_eff = alpha_surface * n_global[p1] + (1.0 - alpha_surface) * n_surface[(p1, surface)]
        n2_eff = alpha_surface * n_global[p2] + (1.0 - alpha_surface) * n_surface[(p2, surface)]

        if k_mode == "adaptive":
            K1 = _k_adaptive(n1_eff, k_min=k_min, k_max=k_max, half_life=half_life)
            K2 = _k_adaptive(n2_eff, k_min=k_min, k_max=k_max, half_life=half_life)
        else:
            K1 = K2 = float(k)

        new_R1 = R1 + K1 * (S1 - exp1)
        new_R2 = R2 + K2 * (S2 - (1.0 - exp1))

        elo_p1_list.append(R1)
        elo_p2_list.append(R2)
        vol5_p1_list.append(vol5_p1)
        vol5_p2_list.append(vol5_p2)

        elo_delta_hist[key1].append(new_R1 - R1)
        elo_delta_hist[key2].append(new_R2 - R2)
        elos[key1] = new_R1
        elos[key2] = new_R2

        # actualizar contadores
        n_global[p1] += 1; n_global[p2] += 1
        n_surface[(p1, surface)] += 1; n_surface[(p2, surface)] += 1

    df["surface_elo_p1"] = elo_p1_list
    df["surface_elo_p2"] = elo_p2_list
    df["surface_elo_diff"] = df["surface_elo_p1"] - df["surface_elo_p2"]
    df["surface_elo_vol5_p1"] = vol5_p1_list
    df["surface_elo_vol5_p2"] = vol5_p2_list
    df["surface_elo_vol5_diff"] = df["surface_elo_vol5_p1"] - df["surface_elo_vol5_p2"]

    return df, dict(elos), dict(n_global), dict(n_surface)


# ===============================
# h2h: global, por superficie y reciente
# ===============================
def compute_h2h(df, initial_h2h=None):
    """
    h2h_count / h2h_balance (global)
    h2h_surface_count / h2h_surface_balance (por superficie)
    h2h_recent3_balance (últimos 3 encuentros global)
    """
    h2h_history = defaultdict(lambda: {"count": 0, "p1_wins": 0})
    h2h_surface_history = defaultdict(lambda: {"count": 0, "p1_wins": 0})
    h2h_recent = defaultdict(list)  # lista de outcomes (1 si p1 ganó aquel duelo), orden temporal

    if initial_h2h is not None:
        # carga opcional (segura) si ya guardabas histórico
        try:
            for k, v in initial_h2h.get("global", {}).items():
                h2h_history[tuple(eval(k))] = v
            for k, v in initial_h2h.get("surface", {}).items():
                h2h_surface_history[tuple(eval(k))] = v
            for k, v in initial_h2h.get("recent", {}).items():
                h2h_recent[tuple(eval(k))] = v
        except Exception:
            pass

    h2h_count_list, h2h_balance_list = [], []
    h2h_s_count_list, h2h_s_balance_list = [], []
    h2h_recent3_bal_list = []

    for _, row in df.iterrows():
        p1 = row["player_1"]; p2 = row["player_2"]
        pair = tuple(sorted([p1, p2]))
        surface = row.get("surface", "Unknown")

        # global
        g = h2h_history[pair]
        count = g["count"]; p1_wins = g["p1_wins"]
        p2_wins = count - p1_wins

        # superficie
        s_key = (*pair, surface)
        s = h2h_surface_history[s_key]
        scount = s["count"]; sp1_wins = s["p1_wins"]
        sp2_wins = scount - sp1_wins

        # reciente (últimos 3)
        rec_list = h2h_recent[pair]
        recent3_balance = 2*_rolling_sum(rec_list,3) - min(len(rec_list),3)

        h2h_count_list.append(count)
        h2h_balance_list.append(p1_wins - p2_wins)
        h2h_s_count_list.append(scount)
        h2h_s_balance_list.append(sp1_wins - sp2_wins)
        h2h_recent3_bal_list.append(recent3_balance)

        # actualizar históricos con el resultado actual
        outcome_p1 = row.get("target", 1)  # p1 ganó (1) / perdió (0)
        h2h_history[pair]["count"] += 1
        h2h_history[pair]["p1_wins"] += outcome_p1

        h2h_surface_history[s_key]["count"] += 1
        h2h_surface_history[s_key]["p1_wins"] += outcome_p1

        rec_list.append(outcome_p1)

    df["h2h_count"] = h2h_count_list
    df["h2h_balance"] = h2h_balance_list
    df["h2h_surface_count"] = h2h_s_count_list
    df["h2h_surface_balance"] = h2h_s_balance_list
    df["h2h_recent3_balance"] = h2h_recent3_bal_list

    h2h_history_serializable = {
        "global": {str(k): v for k, v in h2h_history.items()},
        "surface": {str(k): v for k, v in h2h_surface_history.items()},
        "recent": {str(k): v for k, v in h2h_recent.items()},
    }
    return df, h2h_history_serializable

# ===============================
# stats históricas (medias previas)
# ===============================
def compute_historic_stats_with_ratios(df, initial_stats=None, min_matches=3):
    """
    calcula estadísticas históricas para cada jugador antes de cada partido,
    normaliza por número de partidos previos y crea ratios p1/p2.

    args:
        df: dataframe con partidos y stats individuales por jugador.
        initial_stats: diccionario opcional con stats previos.
        min_matches: número mínimo de partidos para normalizar (evita outliers de pocos partidos).

    returns:
        df con columnas de medias y ratios, final_stats dict.
    """
    from collections import defaultdict
    import numpy as np

    stats = defaultdict(lambda: defaultdict(list))
    if initial_stats:
        for k, v in initial_stats.items():
            stats[k].update(v)

    stat_cols = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced']
    surface_stat_cols = ['ace']

    # prealocar columnas
    for prefix in ['p1_', 'p2_']:
        for stat in stat_cols:
            df[f'{prefix}{stat}_mean'] = np.nan
        for stat in surface_stat_cols:
            df[f'{prefix}{stat}_surface_mean'] = np.nan

    for idx, row in df.iterrows():
        for prefix, player in zip(['p1_', 'p2_'], [row['player_1'], row['player_2']]):
            for stat in stat_cols:
                prev = stats[player][stat]
                if prev and len(prev) >= min_matches:
                    df.at[idx, f'{prefix}{stat}_mean'] = np.mean(prev)
                elif prev:
                    # normaliza si hay pocos partidos (factor de ajuste)
                    adj = (float(len(prev)) / float(min_matches))
                else:
                    df.at[idx, f'{prefix}{stat}_mean'] = np.nan

            # por superficie
            surface = row.get('surface', None)
            for stat in surface_stat_cols:
                prev_surface = [v for s, v in stats[player].get(f'{stat}_surface', []) if s == surface]
                if prev_surface and len(prev_surface) >= min_matches:
                    df.at[idx, f'{prefix}{stat}_surface_mean'] = np.mean(prev_surface)
                elif prev_surface:
                    # normaliza si hay pocos partidos en esa superficie
                    df.at[idx, f'{prefix}{stat}_surface_mean'] = np.mean(prev_surface) * (float(len(prev_surface)) / float(min_matches))
                else:
                    df.at[idx, f'{prefix}{stat}_surface_mean'] = np.nan

        # actualizar stats con partido actual
        for prefix, player in zip(['p1_', 'p2_'], [row['player_1'], row['player_2']]):
            for stat in stat_cols:
                raw_value = row.get(f'{prefix}{stat}', np.nan)
                value = safe_to_numeric(raw_value)
                if not safe_isnan(value):
                    stats[player][stat].append(value)
            for stat in surface_stat_cols:
                raw_value = row.get(f'{prefix}{stat}', np.nan)
                value = safe_to_numeric(raw_value)
                surface = row.get('surface', None)
                if not safe_isnan(value) and surface:
                    stats[player].setdefault(f'{stat}_surface', []).append((surface, value))

    # crear ratios p1/p2
    for stat in stat_cols + surface_stat_cols:
        col1 = f'p1_{stat}_mean'
        col2 = f'p2_{stat}_mean' if stat in stat_cols else f'p2_{stat}_surface_mean'
        ratio_col = f'{stat}_ratio'
        df[ratio_col] = df[col1] / (df[col2] + 1e-6)  # evitar división por cero

    final_stats = {k: dict(v) for k, v in stats.items()}
    return df, final_stats

# ===============================
# forma reciente, especialización, fatiga/ritmo
# ===============================
def add_recent_form_and_fatigue(df):
    """
    requiere 'target' y recomienda disponer de fecha.
    crea:
      - form_winrate_last{5,10} por jugador y diff/ratio
      - surface_winrate_alltime y last5 por jugador y diff
      - streak_wins por jugador y diff
      - days_since_last por jugador y diff
      - matches_last7d/14d y feature combinada escalada + interacción con elo
    """
    date_ser = _as_date_series(df)  # puede ser None
    if date_ser is not None:
        df = df.copy()
        df['__date__'] = date_ser
        # garantizar orden temporal
        df.sort_values('__date__', inplace=True)
        df.reset_index(drop=True, inplace=True)

    # historias por jugador
    wins_hist = defaultdict(list)          # 1/0 outcomes
    wins_surface_hist = defaultdict(list)  # (surface, outcome)
    last_date = defaultdict(lambda: None)  # fecha del último partido
    recent_dates = defaultdict(list)       # fechas de los últimos partidos (para 7/14d)
    surface_matches = defaultdict(lambda: defaultdict(list))  # Por superficie

    # prealoc columnas
    cols = [
        'p1_form_winrate_last5','p2_form_winrate_last5',
        'p1_form_winrate_last10','p2_form_winrate_last10',
        'p1_surface_wr_all','p2_surface_wr_all',
        'p1_surface_wr_last5','p2_surface_wr_last5',
        'p1_streak_wins','p2_streak_wins',
        'p1_days_since_last','p2_days_since_last',
        'p1_matches_last7d','p2_matches_last7d',
        'p1_matches_last14d','p2_matches_last14d',
        # Nuevas features mejoradas
        'p1_matches_last3d','p2_matches_last3d',
        'p1_days_since_surface','p2_days_since_surface',
        'p1_surface_matches_last30d','p2_surface_matches_last30d'
    ]
    for c in cols:
        df[c] = np.nan

    for idx, row in df.iterrows():
        p1, p2 = row['player_1'], row['player_2']
        surface = row.get('surface', 'Unknown')
        d = row['__date__'] if '__date__' in df.columns else None

        # winrates recientes
        wr5_p1 = _rolling_mean(wins_hist[p1], 5)
        wr5_p2 = _rolling_mean(wins_hist[p2], 5)
        wr10_p1 = _rolling_mean(wins_hist[p1], 10)
        wr10_p2 = _rolling_mean(wins_hist[p2], 10)

        # superficie (all-time previa y last5)
        s_prev_p1 = [o for s, o in wins_surface_hist[p1] if s == surface]
        s_prev_p2 = [o for s, o in wins_surface_hist[p2] if s == surface]
        s_wr_all_p1 = np.mean(s_prev_p1) if s_prev_p1 else np.nan
        s_wr_all_p2 = np.mean(s_prev_p2) if s_prev_p2 else np.nan
        s_wr5_p1 = np.mean(s_prev_p1[-5:]) if len(s_prev_p1) > 0 else np.nan
        s_wr5_p2 = np.mean(s_prev_p2[-5:]) if len(s_prev_p2) > 0 else np.nan

        # streak de victorias
        def _streak(lst):
            if not lst: return 0
            cnt = 0
            for v in lst[::-1]:
                if v == 1: cnt += 1
                else: break
            return cnt
        streak_p1 = _streak(wins_hist[p1])
        streak_p2 = _streak(wins_hist[p2])

        # fatiga/ritmo si hay fechas - MEJORADO
        if d is not None:
            ld1, ld2 = last_date[p1], last_date[p2]
            ds1 = (d - ld1).days if ld1 is not None else np.nan
            ds2 = (d - ld2).days if ld2 is not None else np.nan

            # Días desde último partido en esta superficie específica
            last_surface_dates_p1 = [fecha for fecha, surf in zip(recent_dates[p1],
                                     [surf for fecha, surf in surface_matches[p1][surface]])
                                     if surf == surface]
            last_surface_dates_p2 = [fecha for fecha, surf in zip(recent_dates[p2],
                                     [surf for fecha, surf in surface_matches[p2][surface]])
                                     if surf == surface]
            
            ds_surface_p1 = (d - max(last_surface_dates_p1)).days if last_surface_dates_p1 else np.nan
            ds_surface_p2 = (d - max(last_surface_dates_p2)).days if last_surface_dates_p2 else np.nan

            # ventana de partidos con más granularidad
            def _count_recent(lst_dates, days):
                if not lst_dates: return 0
                threshold = d - pd.Timedelta(days=days)
                return sum(1 for x in lst_dates if x is not None and x > threshold)

            def _count_surface_recent(player, surf, days):
                if surf not in surface_matches[player]: return 0
                surf_dates = [fecha for fecha, s in surface_matches[player][surf] if s == surf]
                threshold = d - pd.Timedelta(days=days)
                return sum(1 for x in surf_dates if x is not None and x > threshold)

            m3_p1 = _count_recent(recent_dates[p1], 3)
            m7_p1 = _count_recent(recent_dates[p1], 7)
            m14_p1 = _count_recent(recent_dates[p1], 14)
            m3_p2 = _count_recent(recent_dates[p2], 3)
            m7_p2 = _count_recent(recent_dates[p2], 7)
            m14_p2 = _count_recent(recent_dates[p2], 14)
            
            # Partidos en superficie específica últimos 30 días
            ms30_p1 = _count_surface_recent(p1, surface, 30)
            ms30_p2 = _count_surface_recent(p2, surface, 30)
        else:
            ds1 = ds2 = ds_surface_p1 = ds_surface_p2 = np.nan
            m3_p1 = m7_p1 = m14_p1 = m3_p2 = m7_p2 = m14_p2 = np.nan
            ms30_p1 = ms30_p2 = np.nan

        # escribir valores mejorados
        df.at[idx, 'p1_form_winrate_last5']  = wr5_p1
        df.at[idx, 'p2_form_winrate_last5']  = wr5_p2
        df.at[idx, 'p1_form_winrate_last10'] = wr10_p1
        df.at[idx, 'p2_form_winrate_last10'] = wr10_p2
        df.at[idx, 'p1_surface_wr_all']      = s_wr_all_p1
        df.at[idx, 'p2_surface_wr_all']      = s_wr_all_p2
        df.at[idx, 'p1_surface_wr_last5']    = s_wr5_p1
        df.at[idx, 'p2_surface_wr_last5']    = s_wr5_p2
        df.at[idx, 'p1_streak_wins']         = streak_p1
        df.at[idx, 'p2_streak_wins']         = streak_p2
        df.at[idx, 'p1_days_since_last']     = ds1
        df.at[idx, 'p2_days_since_last']     = ds2
        df.at[idx, 'p1_matches_last3d']      = m3_p1
        df.at[idx, 'p1_matches_last7d']      = m7_p1
        df.at[idx, 'p1_matches_last14d']     = m14_p1
        df.at[idx, 'p2_matches_last3d']      = m3_p2
        df.at[idx, 'p2_matches_last7d']      = m7_p2
        df.at[idx, 'p2_matches_last14d']     = m14_p2
        df.at[idx, 'p1_days_since_surface']  = ds_surface_p1
        df.at[idx, 'p2_days_since_surface']  = ds_surface_p2
        df.at[idx, 'p1_surface_matches_last30d'] = ms30_p1
        df.at[idx, 'p2_surface_matches_last30d'] = ms30_p2

        # actualizar historiales tras el partido
        outcome_p1 = row.get("target", 1)
        outcome_p2 = 1 - outcome_p1
        wins_hist[p1].append(outcome_p1)
        wins_hist[p2].append(outcome_p2)
        wins_surface_hist[p1].append((surface, outcome_p1))
        wins_surface_hist[p2].append((surface, outcome_p2))

        if d is not None:
            last_date[p1] = d; last_date[p2] = d
            recent_dates[p1].append(d); recent_dates[p2].append(d)
            surface_matches[p1][surface].append((d, surface))
            surface_matches[p2][surface].append((d, surface))

    # diffs/ratios útiles (centrar en partido) - AMPLIADO
    df['form_last5_diff']  = df['p1_form_winrate_last5']  - df['p2_form_winrate_last5']
    df['form_last10_diff'] = df['p1_form_winrate_last10'] - df['p2_form_winrate_last10']
    df['surface_wr_all_diff']   = df['p1_surface_wr_all']   - df['p2_surface_wr_all']
    df['surface_wr_last5_diff'] = df['p1_surface_wr_last5'] - df['p2_surface_wr_last5']
    df['streak_wins_diff'] = df['p1_streak_wins'] - df['p2_streak_wins']
    df['days_since_last_diff'] = df['p1_days_since_last'] - df['p2_days_since_last']
    df['matches_last3d_diff']  = df['p1_matches_last3d']  - df['p2_matches_last3d']
    df['matches_last7d_diff']  = df['p1_matches_last7d']  - df['p2_matches_last7d']
    df['matches_last14d_diff'] = df['p1_matches_last14d'] - df['p2_matches_last14d']
    df['days_since_surface_diff'] = df['p1_days_since_surface'] - df['p2_days_since_surface']
    df['surface_matches_30d_diff'] = df['p1_surface_matches_last30d'] - df['p2_surface_matches_last30d']

    # ===============================
    # escalado y combinación ponderada MEJORADO
    # ===============================
    df['matches_last3d_diff_scaled'] = df['matches_last3d_diff'].clip(-3, 3) / 3.0
    df['matches_last7d_diff_scaled'] = df['matches_last7d_diff'].clip(-5, 5) / 5.0
    df['matches_last14d_diff_scaled'] = df['matches_last14d_diff'].clip(-10, 10) / 10.0
    df['days_since_last_diff_scaled'] = df['days_since_last_diff'].clip(-30, 30) / 30.0
    df['days_since_surface_diff_scaled'] = df['days_since_surface_diff'].clip(-60, 60) / 60.0

    # Combinación ponderada más sofisticada
    df['fatigue_immediate'] = 0.5 * df['matches_last3d_diff_scaled'] + 0.3 * df['matches_last7d_diff_scaled']
    df['fatigue_short_term'] = 0.6*df['matches_last7d_diff_scaled'] + 0.4*df['matches_last14d_diff_scaled']
    df['recovery_factor'] = 0.7 * df['days_since_last_diff_scaled'] + 0.3 * df['days_since_surface_diff_scaled']

    # Feature combinada final
    df['matches_recent_weighted'] = 0.4*df['fatigue_immediate'] + 0.4*df['fatigue_short_term'] + 0.2*df['recovery_factor']

    # interacción con elo (suaviza la influencia de la fatiga)
    if 'elo_prob' in df.columns:
        df['elo_matches_interaction'] = df['elo_prob'] * df['matches_recent_weighted']
        df['elo_fatigue_immediate'] = df['elo_prob'] * df['fatigue_immediate']
    else:
        df['elo_matches_interaction'] = df['matches_recent_weighted']
        df['elo_fatigue_immediate'] = df['fatigue_immediate']

    return df

# ===============================
# elo features enriquecidas (menos dominación)
# ===============================
def add_enhanced_elo_features(df):
    # diferencias continuas principales
    df['elo_diff'] = df['elo_p1'] - df['elo_p2']
    df['surface_elo_diff'] = df['surface_elo_p1'] - df['surface_elo_p2']

    # clipping suave para evitar outliers que “aplastan” al resto
    df['elo_diff_clip'] = df['elo_diff'].clip(-400, 400)
    df['surface_elo_diff_clip'] = df['surface_elo_diff'].clip(-400, 400)

    # probabilidades derivadas de elo (calibrables por el modelo)
    df['elo_prob'] = 1 / (1 + 10 ** (-df['elo_diff_clip'] / 400))
    df['surface_elo_prob'] = 1 / (1 + 10 ** (-df['surface_elo_diff_clip'] / 400))

    # no linealidades suaves
    df['elo_diff_sq'] = (df['elo_diff_clip'] / 100.0) ** 2
    df['surface_elo_diff_sq'] = (df['surface_elo_diff_clip'] / 100.0) ** 2

    # interacciones y consistencia
    df['elo_surface_interaction'] = (df['elo_diff_clip'] * df['surface_elo_diff_clip']) / 10000
    df['elo_consistency'] = abs(df['elo_diff_clip'] - df['surface_elo_diff_clip'])

    # volatilidad previa (calculada en compute_elo_*). si no existen, crea 0 para robustez.
    for c in ['elo_vol5_diff', 'surface_elo_vol5_diff']:
        if c not in df.columns:
            df[c] = 0.0

    # ranking
    if 'p1_rank' in df.columns and 'p2_rank' in df.columns:
        df['p1_rank'] = pd.to_numeric(df['p1_rank'], errors='coerce')
        df['p2_rank'] = pd.to_numeric(df['p2_rank'], errors='coerce')
        df['rank_diff'] = df['p2_rank'] - df['p1_rank']
        df['rank_ratio'] = (df['p2_rank'] + 1) / (df['p1_rank'] + 1)
        df['log_rank_ratio'] = np.log1p(df['p2_rank']) - np.log1p(df['p1_rank'])
        df['elo_rank_mismatch'] = ((df['elo_diff_clip'] > 0) & (df['rank_diff'] < 0)).astype(int) - \
                                  ((df['elo_diff_clip'] < 0) & (df['rank_diff'] > 0)).astype(int)
    else:
        df['rank_diff'] = 0
        df['rank_ratio'] = 1.0
        df['log_rank_ratio'] = 0.0
        df['elo_rank_mismatch'] = 0

    # competitividad del partido (suavizada)
    df['match_competitiveness'] = 1 / (1 + abs(df['elo_diff_clip']) / 120)

    return df

# ===============================
# selección de columnas
# ===============================
import re


def get_curated_features(df):
    """
    Devuelve todas las columnas que coinciden con la lista base,
    incluyendo variantes con sufijos como '_pca', '_scaled', etc.
    Además, incluye automáticamente los componentes PCA generados por los grupos definidos en PCA_GROUPS.
    """
    base = [
        'elo_prob', 'surface_elo_prob',
        'elo_diff_clip', 'surface_elo_diff_clip',
        'elo_diff_sq', 'surface_elo_diff_sq',
        'elo_surface_interaction', 'elo_consistency',
        'elo_vol5_diff', 'surface_elo_vol5_diff',

        'rank_diff', 'rank_ratio', 'log_rank_ratio', 'elo_rank_mismatch',

        'h2h_count', 'h2h_balance',
        'h2h_surface_count', 'h2h_surface_balance',
        'h2h_recent3_balance',

        'form_last5_diff', 'form_last10_diff',
        'surface_wr_all_diff', 'surface_wr_last5_diff',
        'streak_wins_diff',
        'matches_last7d_diff', 'matches_last14d_diff', 'days_since_last_diff',

        'matches_recent_weighted', 'days_since_last_diff_scaled', 'elo_matches_interaction',

        'match_competitiveness',

        'p1_ace_mean', 'p1_df_mean', 'p1_svpt_mean', 'p1_1stIn_mean', 'p1_1stWon_mean', 'p1_2ndWon_mean',
        'p1_bpSaved_mean', 'p1_bpFaced_mean', 'p1_ace_surface_mean',
        'p2_ace_mean', 'p2_df_mean', 'p2_svpt_mean', 'p2_1stIn_mean', 'p2_1stWon_mean', 'p2_2ndWon_mean',
        'p2_bpSaved_mean', 'p2_bpFaced_mean', 'p2_ace_surface_mean',
    ]

    curated_features = []
    # Features base y variantes
    for col in df.columns:
        for b in base:
            # Coincide exactamente o con sufijo (_pca, _scaled, etc.)
            if re.fullmatch(b + r'(_\w+)?', col):
                curated_features.append(col)
                break
    # Añadir componentes PCA por grupo
    for group in PCA_GROUPS.keys():
        for col in df.columns:
            if re.fullmatch(group + r'_pca\d+', col):
                curated_features.append(col)
    return curated_features


# ===============================
# identificadores y normalización de nombres
# ===============================
def ensure_player_identifiers(df):
    df = df.copy()
    if 'player_1' not in df.columns:
        if 'player_1_id' in df.columns:
            df['player_1'] = df['player_1_id']
        elif 'player_1_name' in df.columns:
            df['player_1'] = df['player_1_name']
    if 'player_2' not in df.columns:
        if 'player_2_id' in df.columns:
            df['player_2'] = df['player_2_id']
        elif 'player_2_name' in df.columns:
            df['player_2'] = df['player_2_name']
    missing = [c for c in ['player_1','player_2'] if c not in df.columns]
    if missing:
        raise ValueError(f"faltan columnas identificadoras de jugadores: {missing}. revisa rename_winner_loser_columns.")
    return df

def rename_winner_loser_columns(df):
    df = df.copy()
    column_map = {
        'winner_id': 'player_1_id',
        'winner_name': 'player_1',
        'winner_hand': 'player_1_hand',
        'winner_rank': 'p1_rank',
        'winner': 'player_1',
        'player1': 'player_1',
        'loser_id': 'player_2_id',
        'loser_name': 'player_2',
        'loser_hand': 'player_2_hand',
        'loser_rank': 'p2_rank',
        'loser': 'player_2',
        'player2': 'player_2',
        'w_ace': 'p1_ace', 'w_df': 'p1_df', 'w_svpt': 'p1_svpt',
        'w_1stIn': 'p1_1stIn', 'w_1stWon': 'p1_1stWon', 'w_2ndWon': 'p1_2ndWon',
        'w_bpSaved': 'p1_bpSaved','w_bpFaced': 'p1_bpFaced',
        'l_ace': 'p2_ace', 'l_df': 'p2_df', 'l_svpt': 'p2_svpt',
        'l_1stIn': 'p2_1stIn', 'l_1stWon': 'p2_1stWon', 'l_2ndWon': 'p2_2ndWon',
        'l_bpSaved': 'p2_bpSaved','l_bpFaced': 'p2_bpFaced',
        'y': 'target',
    }
    explicit = set(column_map.keys())
    for col in list(df.columns):
        if col in explicit:
            continue
        if col.startswith('w_') and col not in column_map:
            column_map[col] = 'p1_' + col[2:]
        elif col.startswith('l_') and col not in column_map:
            column_map[col] = 'p2_' + col[2:]
        elif col.startswith('winner_') and col not in column_map:
            suffix = col[len('winner_'):]
            if suffix not in ['id','name','hand','rank']:
                column_map[col] = 'p1_' + suffix
        elif col.startswith('loser_') and col not in column_map:
            suffix = col[len('loser_'):]
            if suffix not in ['id','name','hand','rank']:
                column_map[col] = 'p2_' + suffix
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    df = ensure_player_identifiers(df)
    return df

def randomize_player_order(df):
    """aleatoriza filas intercambiando todas las columnas p1_/p2_ coherentemente y corrige target."""
    df = df.copy()

    # --- Normaliza dtypes numéricos (evita FutureWarning al swap) ---
    def _maybe_coerce_numeric(s: pd.Series):
        if s.dtype != object:
            return s
        non_null = s.dropna().astype(str)
        if len(non_null) == 0:
            return s
        if non_null.str.fullmatch(r'-?\d+(\.\d+)?').all():
            return pd.to_numeric(s, errors='coerce')
        # intento conversión parcial (>=95% numérico) para limpiar strings sueltos
        parsed = pd.to_numeric(s, errors='coerce')
        if parsed.notna().mean() >= 0.95:
            return parsed
        return s

    candidate_cols = [c for c in df.columns if c.startswith(('p1_','p2_')) and not any(c.endswith(x) for x in ['name','hand'])]
    for c in candidate_cols:
        df[c] = _maybe_coerce_numeric(df[c])
    # ----------------------------------------------------------------

    mask = np.random.rand(len(df)) < 0.5
    if mask.sum() == 0:
        return df
    p1_cols = [c for c in df.columns if c.startswith('p1_')]
    p2_cols = [c for c in df.columns if c.startswith('p2_')]
    p1_suffix = {c[3:] for c in p1_cols}
    p2_suffix = {c[3:] for c in p2_cols}
    common = p1_suffix.intersection(p2_suffix)

    for suf in common:
        c1 = 'p1_' + suf
        c2 = 'p2_' + suf
        if c1 not in df.columns or c2 not in df.columns:
            continue

        # Convertir ambas columnas al mismo dtype compatible antes del swap
        col1_data = df[c1].copy()
        col2_data = df[c2].copy()

        # Intentar conversión numérica primero
        try:
            col1_numeric = pd.to_numeric(col1_data, errors='coerce')
            col2_numeric = pd.to_numeric(col2_data, errors='coerce')

            # Si ambas son mayoritariamente numéricas, usar numeric
            if (col1_numeric.notna().sum() >= len(col1_data) * 0.8 and
                col2_numeric.notna().sum() >= len(col2_data) * 0.8):
                df[c1] = col1_numeric
                df[c2] = col2_numeric
            else:
                # Convertir a string para swap seguro
                df[c1] = col1_data.astype('str')
                df[c2] = col2_data.astype('str')
        except Exception:
            # Fallback: convertir a string
            df[c1] = col1_data.astype('str')
            df[c2] = col2_data.astype('str')

        # Hacer el swap usando .values para evitar warnings
        if mask.sum() > 0:
            temp_values = df.loc[mask, c1].values.copy()
            df.loc[mask, c1] = df.loc[mask, c2].values
            df.loc[mask, c2] = temp_values

    # swap especial para columnas principales
    if {'player_1','player_2'}.issubset(df.columns):
        if mask.sum() > 0:
            temp_values = df.loc[mask, 'player_1'].values.copy()
            df.loc[mask, 'player_1'] = df.loc[mask, 'player_2'].values
            df.loc[mask, 'player_2'] = temp_values

    for pair in [('elo_p1','elo_p2'), ('surface_elo_p1','surface_elo_p2')]:
        if all(p in df.columns for p in pair):
            if mask.sum() > 0:
                temp_values = df.loc[mask, pair[0]].values.copy()
                df.loc[mask, pair[0]] = df.loc[mask, pair[1]].values
                df.loc[mask, pair[1]] = temp_values

    if 'target' not in df.columns:
        df['target'] = 1
    df.loc[mask, 'target'] = 1 - df.loc[mask, 'target']
    return df

# ===============================
# pipeline principal
# ===============================
def add_all_features(
    df,
    initial_global_elos=None,
    initial_surface_elos=None,
    initial_h2h=None,
    initial_stats=None,
    mode="train",  # 'train' o 'inference'
    fast=True,
    vif_sample_rows=5000,
    pca_state=None,            # NUEVO: permitir reutilizar PCA en inference
    return_pca_state=False,    # NUEVO: devolver estado PCA para diagnóstico
    randomize_players=None,    # NUEVO: forzar randomización también en inference si se desea balancear labels
    return_full=False          # NUEVO: devolver df completo antes de PCA/corr/VIF para gráficos
):
    """
    Pipeline completo. Añadido parámetro fast para reducir tiempo y ahora pca_state para evitar leakage.
    Si mode='inference' y se pasa pca_state entrenado, se reutilizan imputer+scaler+PCA por grupo.
    randomize_players:
        - None: se randomiza solo si mode=='train' (comportamiento original)
        - True: fuerza randomización (útil para usar conjuntos conocidos como evaluación sin leakage de PCA)
        - False: nunca randomiza
    IMPORTANTE: randomizar antes de calcular Elo/derivadas para coherencia de diffs.

    return_full=True permite conservar un dataframe completo (sin PCA/VIF ni drops correlación) para análisis y gráficos
    que necesitan columnas originales como elo_p1, elo_p2, surface_elo_p1, etc. El dataframe de modelado (df_final)
    se mantiene reducido. Orden de retorno cuando return_full=True:
      - si return_pca_state: (df_final, df_full, final_global_elos, final_surface_elos, final_h2h, final_stats, pca_state)
      - si no:               (df_final, df_full, final_global_elos, final_surface_elos, final_h2h, final_stats)
    Compatibilidad: cuando return_full=False se mantiene la firma anterior.
    """
    df = rename_winner_loser_columns(df)
    df = ensure_player_identifiers(df)
    if randomize_players is None:
        randomize_players = (mode == "train")
    if randomize_players:
        df = randomize_player_order(df)

    df, final_global_elos, match_counts_global  = compute_elo_ratings(
        df,
        k_mode="adaptive",
        k_min=24.0,
        k_max=42.0,
        half_life=50.0,
        initial_elos=initial_global_elos
    )

    df, final_surface_elos, match_counts_global, match_counts_surface = compute_surface_elo(
        df,
        k_mode="adaptive",
        k_min=24.0,
        k_max=42.0,
        half_life=50.0,
        initial_elos=initial_surface_elos,
        initial_match_counts_global = match_counts_global
    )
    df, final_h2h          = compute_h2h(df, initial_h2h=initial_h2h)
    df = add_enhanced_elo_features(df)
    df = add_recent_form_and_fatigue(df)
    df, final_stats        = compute_historic_stats_with_ratios(df, initial_stats=initial_stats)


    # Copia completa antes de PCA / reducciones para análisis
    df_full = df.copy() if return_full else None

    # PCA por grupos
    df, pca_info, pca_state = _apply_group_pca(df, variance_threshold=0.9, mode=mode, existing_pca_state=pca_state)

    # quick drop constants + ultra-high corr
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric, corr_dropped = _drop_constant_and_correlated(numeric, corr_thresh=0.995)

    vif_dropped = []
    if not fast:
        if len(numeric) > vif_sample_rows:
            sample_idx = np.random.choice(numeric.index, size=vif_sample_rows, replace=False)
            vif_df = numeric.loc[sample_idx].copy()
        else:
            vif_df = numeric.copy()
        vif_df, vif_dropped = _drop_high_vif(vif_df, thresh=10.0, max_iter=8, verbose=False)
        numeric.drop(columns=vif_dropped, errors='ignore', inplace=True)

    df_final = numeric
    print(f"[INFO] fast={fast} dropped_by_corr={len(corr_dropped)} dropped_by_vif={len(vif_dropped)} randomize={randomize_players} mode={mode}")
    parts = [df_final]
    if return_full:
        parts.append(df_full)
    parts.extend([final_global_elos, final_surface_elos, final_h2h, final_stats])
    if return_pca_state:
        parts.append(pca_state)
    return tuple(parts)


def _drop_constant_and_correlated(df, corr_thresh=0.995):
    """
    Quita columnas con var=0 y corta pares muy correlados (mantiene una de las dos).
    Devuelve df limpio y lista de columnas dropeadas.
    """
    df = df.copy()
    dropped = []

    # 1) columnas constantes o casi constantes
    variances = df.var(axis=0)
    const_cols = variances[variances == 0].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols)
        dropped.extend(const_cols)

    # 2) drop correlación muy alta (quick)
    if df.shape[1] > 1:
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
        if to_drop:
            df = df.drop(columns=to_drop)
            dropped.extend(to_drop)

    return df, dropped


def _apply_group_pca(df, variance_threshold=0.90, mode='train', existing_pca_state=None, store_loadings=True, scale_before=True):
    """
    PCA por grupos definidos en PCA_GROUPS.
    Ahora:
      - Estandariza (StandardScaler) tras imputar medianas para evitar dominación por escala.
      - Guarda explained_variance_ratio_, varianza acumulada y loadings de cada componente retenido.
      - Permite reutilizar objetos (imputer, scaler, pca) en modo inference para evitar leakage.
    Devuelve df transformado, pca_info (resumen serializable) y pca_state (objetos para reuso).
    """
    df = df.copy()
    pca_info = {}
    pca_state_out = existing_pca_state.copy() if existing_pca_state else {}

    for group_name, cols in PCA_GROUPS.items():
        existing_cols = [c for c in cols if c in df.columns]
        if len(existing_cols) < 2:
            continue
        # Recuperar estado si inference
        reuse = (mode != 'train') and (existing_pca_state is not None) and (group_name in existing_pca_state)
        if reuse:
            state = existing_pca_state[group_name]
            imputer = state['imputer']
            scaler = state.get('scaler')
            pca_full = state['pca']
            original_cols = state['cols']
            # asegurar que todas las columnas existen (crear NaN para faltantes)
            for oc in original_cols:
                if oc not in df.columns:
                    df[oc] = np.nan
            X = df[original_cols].copy()
            X_imputed = pd.DataFrame(imputer.transform(X), columns=original_cols, index=X.index)
            if scaler is not None:
                X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=original_cols, index=X.index)
            else:
                X_scaled = X_imputed
            n_components = state['n_components']
            comps = pca_full.transform(X_scaled)[:, :n_components]
            for i in range(n_components):
                df[f"{group_name}_pca{i+1}"] = comps[:, i]
            df.drop(columns=[c for c in original_cols if c in df.columns], inplace=True)
            # info resumida reuse
            pca_info[group_name] = {
                'created': n_components,
                'dropped': original_cols,
                'explained_variance_ratio': state.get('explained_variance_ratio')[:n_components],
                'variance_cumsum': state.get('variance_cumsum')[:n_components],
                'loadings': state.get('loadings'),
                'reused': True
            }
            continue
        # TRAIN: fit nuevos objetos
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = df[existing_cols].copy()
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=existing_cols, index=X.index)
        if scale_before:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=existing_cols, index=X.index)
        else:
            scaler = None
            X_scaled = X_imputed
        pca_full = PCA(svd_solver='randomized', random_state=42)
        pca_full.fit(X_scaled)
        full_var = pca_full.explained_variance_ratio_
        cumsum = np.cumsum(full_var)
        n_components = int(np.argmax(cumsum >= variance_threshold)) + 1 if any(cumsum >= variance_threshold) else len(existing_cols)
        comps = pca_full.transform(X_scaled)[:, :n_components]
        for i in range(n_components):
            df[f"{group_name}_pca{i+1}"] = comps[:, i]
        df.drop(columns=existing_cols, inplace=True)
        loadings = None
        if store_loadings:
            loadings = {
                f"pca{i+1}": {col: float(pca_full.components_[i, j]) for j, col in enumerate(existing_cols)}
                for i in range(n_components)
            }
        pca_info[group_name] = {
            'created': n_components,
            'dropped': existing_cols,
            'explained_variance_ratio': full_var[:n_components].tolist(),
            'variance_cumsum': cumsum[:n_components].tolist(),
            'loadings': loadings,
            'reused': False
        }
        pca_state_out[group_name] = {
            'imputer': imputer,
            'scaler': scaler,
            'pca': pca_full,
            'cols': existing_cols,
            'n_components': n_components,
            'explained_variance_ratio': full_var.tolist(),
            'variance_cumsum': cumsum.tolist(),
            'loadings': loadings
        }
    return df, pca_info, pca_state_out


def _drop_high_vif(df, thresh=10.0, max_iter=25, verbose=False):
    """
    Iterative VIF remover, robusto a singularidades y suprime warnings.
    1) Imputa medianas
    2) Calcula VIF; si inf o > thresh, elimina la variable con mayor VIF y repite.
    3) Limita a max_iter para evitar loops infinitos.
    Devuelve df reducido y lista de columnas dropeadas.
    """
    df = df.copy()
    imputer = SimpleImputer(strategy="median")
    cols = df.columns.tolist()
    dropped = []

    # imputar una vez para cálculo (no modificar original index/cols)
    X = pd.DataFrame(imputer.fit_transform(df[cols]), columns=cols, index=df.index)

    # --- FILTRADO PREVIO: quitar varianza cero o muy baja ---
    low_var_cols = X.columns[X.std() <= 1e-6].tolist()
    if low_var_cols:
        X.drop(columns=low_var_cols, inplace=True)
        for c in low_var_cols:
            cols.remove(c)
            dropped.append(c)
        if verbose:
            print(f"[VIF] dropped low-variance cols: {low_var_cols}")

    it = 0
    while it < max_iter and len(cols) > 1:
        it += 1
        vif_vals = []
        # suprimir warnings de divide-by-zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suprime RuntimeWarning del interno
            for i in range(len(cols)):
                try:
                    v = variance_inflation_factor(X[cols].values, i)
                except Exception:
                    v = np.inf
                vif_vals.append(v)

        max_vif = max(vif_vals)
        max_idx = int(np.argmax(vif_vals))
        if np.isinf(max_vif) or max_vif > thresh:
            col_to_drop = cols[max_idx]
            dropped.append(col_to_drop)
            cols.pop(max_idx)
            X.drop(columns=[col_to_drop], inplace=True)
            if verbose:
                print(f"[VIF] Iter {it}: drop {col_to_drop} (vif={max_vif:.2f})")
        else:
            break

    # devolver df sin las columnas dropeadas
    return df.drop(columns=dropped, errors="ignore"), dropped


def export_pca_report(pca_state):
    """Devuelve DataFrame resumen por grupo para análisis (n_components, varianza primer componente, acumulada)."""
    if not pca_state:
        return pd.DataFrame()
    rows = []
    for g, st in pca_state.items():
        var = st.get('explained_variance_ratio', [])
        cum = st.get('variance_cumsum', [])
        rows.append({
            'group': g,
            'n_components': st.get('n_components'),
            'pc1_var': var[0] if var else np.nan,
            'pc1_cum': cum[0] if cum else np.nan,
            'cum_at_n': cum[st.get('n_components')-1] if cum and st.get('n_components') else np.nan
        })
    return pd.DataFrame(rows)


def diagnose_pca_group(df_original, group_name, pca_state):
    """Diagnóstico rápido de un grupo PCA ya entrenado: devuelve loadings ordenados y varianzas."""
    if group_name not in pca_state:
        raise ValueError(f"Grupo {group_name} no en pca_state")
    st = pca_state[group_name]
    loadings = st.get('loadings', {})
    pc1 = loadings.get('pca1', {}) if loadings else {}
    pc1_sorted = sorted(pc1.items(), key=lambda x: abs(x[1]), reverse=True)
    return {
        'group': group_name,
        'n_components': st['n_components'],
        'explained_variance_ratio': st['explained_variance_ratio'][:st['n_components']],
        'variance_cumsum': st['variance_cumsum'][:st['n_components']],
        'pc1_loadings_sorted': pc1_sorted
    }

