import os

from data_loader import load_train_data, load_test_data, BASE_DIR
from preprocess import clean_data
from features import add_all_features

# este script principal carga los datos, genera features y muestra estadísticas básicas

def main():
    print("cargando datos...")
    df = load_train_data()
    df = clean_data(df)

    print("generando features...")
    df = add_all_features(df)

    print("primeras filas con features:")
    print(df[[
        "winner_name", "loser_name",
        "elo_winner", "elo_loser", "elo_diff",
        "surface_elo_winner", "surface_elo_loser", "surface_elo_diff",
        "h2h_count", "h2h_balance"
    ]].head())

    output_path = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_path, exist_ok=True)

    df.to_csv(os.path.join(output_path, "features.csv"), index=False)

    print("rango de fechas del dataset:")
    print(df["tourney_date"].min(), "→", df["tourney_date"].max())
    print("cantidad de filas:", len(df))
    print("años únicos:", df["tourney_date"].dt.year.unique())

    print("distribución de partidos por año:")
    print(df["tourney_date"].dt.year.value_counts().sort_index())

if __name__ == "__main__":
    main()
