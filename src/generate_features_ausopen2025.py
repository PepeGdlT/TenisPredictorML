import pandas as pd
import os
import json
from data_loader import BASE_DIR
from preprocess import clean_data
from features import add_all_features

def generate_ausopen_matches_dual_df():
    # --- Datos de partidos (copiado de generate_2025_data.py) ---
    matches_by_round = {
        "F": [
            ("26/01/2025", "Jannik Sinner", "3 - 0", "Alexander Zverev"),
        ],
        "SF": [
            ("24/01/2025", "Jannik Sinner", "3 - 0", "Ben Shelton"),
            ("24/01/2025", "Novak Djokovic", "0 - 1", "Alexander Zverev"),
        ],
        "QF": [
            ("22/01/2025", "Jannik Sinner", "3 - 0", "Alex De Minaur"),
            ("22/01/2025", "Ben Shelton", "3 - 1", "Lorenzo Sonego"),
            ("21/01/2025", "Novak Djokovic", "3 - 1", "Carlos Alcaraz"),
            ("21/01/2025", "Tommy Paul", "1 - 3", "Alexander Zverev"),
        ],
        "R16": [
            ("20/01/2025", "Alex Michelsen", "0 - 3", "Alex De Minaur"),
            ("20/01/2025", "Gael Monfils", "1 - 2", "Ben Shelton"),
            ("20/01/2025", "Lorenzo Sonego", "3 - 1", "Learner Tien"),
            ("20/01/2025", "Jannik Sinner", "3 - 1", "Holger Rune"),
            ("19/01/2025", "Novak Djokovic", "3 - 0", "Jiri Lehecka"),
            ("19/01/2025", "Ugo Humbert", "1 - 3", "Alexander Zverev"),
            ("19/01/2025", "Jack Draper", "0 - 2", "Carlos Alcaraz"),
            ("19/01/2025", "Alejandro Davidovich Fokina", "0 - 3", "Tommy Paul"),
        ],
        "R32": [
            ("18/01/2025", "Miomir Kecmanovic", "2 - 3", "Holger Rune"),
            ("18/01/2025", "Jannik Sinner", "3 - 0", "Marcos Giron"),
            ("18/01/2025", "Ben Shelton", "3 - 1", "Lorenzo Musetti"),
            ("18/01/2025", "Corentin Moutet", "0 - 3", "Learner Tien"),
            ("18/01/2025", "Lorenzo Sonego", "3 - 1", "Fabian Marozsan"),
            ("18/01/2025", "Francisco Cerundolo", "1 - 3", "Alex De Minaur"),
            ("18/01/2025", "Taylor Fritz", "1 - 3", "Gael Monfils"),
            ("18/01/2025", "Alex Michelsen", "3 - 0", "Karen Khachanov"),
            ("17/01/2025", "Jack Draper", "3 - 2", "Aleksandar Vukic"),
            ("17/01/2025", "Jiri Lehecka", "3 - 0", "Benjamin Bonzi"),
            ("17/01/2025", "Novak Djokovic", "3 - 0", "Tomas Machac"),
            ("17/01/2025", "Ugo Humbert", "2 - 1", "Arthur Fils"),
            ("17/01/2025", "Jakub Mensik", "2 - 3", "Alejandro Davidovich Fokina"),
            ("17/01/2025", "Jacob Fearnley", "0 - 3", "Alexander Zverev"),
            ("17/01/2025", "Nuno Borges", "1 - 3", "Carlos Alcaraz"),
            ("17/01/2025", "Roberto Carballes Baena", "0 - 3", "Tommy Paul"),
        ],
        # ... puedes añadir más rondas si lo necesitas ...
    }
    rows = []
    round_order = {'R64': 1, 'R32': 2, 'R16': 3, 'QF': 4, 'SF': 5, 'F': 6}
    for rnd, matches in matches_by_round.items():
        for date_str, p1, score, p2 in matches:
            score_parts = score.split('-')
            p1_score = int(score_parts[0].strip())
            p2_score = int(score_parts[1].strip())
            if p1_score > p2_score:
                winner, loser, final_score = p1, p2, score.replace(' ', '')
            else:
                winner, loser, final_score = p2, p1, f"{p2_score}-{p1_score}"
            date_fmt = pd.to_datetime(date_str, format='%d/%m/%Y').strftime('%Y%m%d')
            # Fila para el ganador
            rows.append({
                'tourney_id': '2025-01-580',
                'tourney_name': 'Australian Open',
                'surface': 'Hard',
                'tourney_level': 'G',
                'tourney_date': date_fmt,
                'round': rnd,
                'player': winner,
                'opponent': loser,
                'score': final_score,
                'best_of': 5,
                'target': 1,
                'winner_name': winner,
                'loser_name': loser,
                '_sort_date': date_fmt,
                '_sort_round': round_order[rnd],
            })
            # Fila para el perdedor
            rows.append({
                'tourney_id': '2025-01-580',
                'tourney_name': 'Australian Open',
                'surface': 'Hard',
                'tourney_level': 'G',
                'tourney_date': date_fmt,
                'round': rnd,
                'player': loser,
                'opponent': winner,
                'score': final_score,
                'best_of': 5,
                'target': 0,
                'winner_name': winner,
                'loser_name': loser,
                '_sort_date': date_fmt,
                '_sort_round': round_order[rnd],
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(by=['_sort_date', '_sort_round']).reset_index(drop=True)
    df = df.drop(columns=['_sort_date', '_sort_round'])
    return df

def generate_ausopen_features():
    df = generate_ausopen_matches_dual_df()
    # Cargar elos y h2h
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
    h2h_path = os.path.join(processed_data_path, "final_h2h.json")
    try:
        with open(h2h_path, "r") as f:
            initial_h2h = json.load(f)
        print("Historial H2H inicial cargado.")
    except FileNotFoundError:
        initial_h2h = None
        print("[Aviso] No se encontró historial H2H inicial. Se parte de cero.")
    # Limpiar datos
    df = clean_data(df)
    # Generar features
    features_df, _, _, final_h2h = add_all_features(
        df,
        initial_global_elos=initial_global_elos,
        initial_surface_elos=initial_surface_elos,
        initial_h2h=initial_h2h
    )
    # Guardar features (único output)
    features_path = os.path.join(processed_data_path, "features_ausopen2025.csv")
    features_df.to_csv(features_path, index=False)
    print(f"Features generadas y guardadas en: {features_path}")
    # Guardar historial h2h actualizado
    with open(h2h_path, "w") as f:
        json.dump(final_h2h, f)
    print(f"Historial H2H actualizado guardado en: {h2h_path}")

if __name__ == "__main__":
    generate_ausopen_features()
