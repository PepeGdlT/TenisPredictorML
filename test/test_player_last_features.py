import pickle

# ============ CONFIG ============ #
PKL_PATH = "outputs/player_last_features.pkl"
PLAYER_EXAMPLE = "Carlos Alcaraz"
# ================================= #

print("🔍 Cargando archivo PKL...")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

print("✅ Archivo cargado")
print(f"   Keys principales: {list(data.keys())}")
print(f"   Total jugadores: {len(data['player_data_full'])}")

# ========================
# TEST: Info legible de un jugador
# ========================
if PLAYER_EXAMPLE in data["player_data_full"]:
    print(f"\n🎾 Info legible de {PLAYER_EXAMPLE}...")

    full_info = data["player_data_full"][PLAYER_EXAMPLE]
    stats = full_info["last_features"]
    meta = full_info["last_match_meta"]

    # --- Meta información del último partido
    print("\n📌 Último partido:")
    print(f"   Torneo: {meta.get('tourney_name')}")
    print(f"   Fecha: {meta.get('tourney_date')}")
    print(f"   Superficie: {meta.get('surface')}")
    print(f"   Ronda: {meta.get('round')}")
    print(f"   Rival: {meta.get('opponent')}")

    # --- Elo ratings
    print("\n📈 Elo ratings:")
    for key in ["elo_p1", "elo_p2", "elo_diff", "surface_elo_p1", "surface_elo_p2", "surface_elo_diff", "p1_rank"]:
        if key in stats:
            print(f"   {key}: {stats[key]}")

    # --- Volatilidad Elo
    print("\n📉 Volatilidad Elo (últimos 5 partidos):")
    for key in ["elo_vol5_p1", "elo_vol5_p2", "elo_vol5_diff",
                "surface_elo_vol5_p1", "surface_elo_vol5_p2", "surface_elo_vol5_diff"]:
        if key in stats:
            print(f"   {key}: {stats[key]}")

    # --- Head-to-head
    print("\n🤜🤛 Head-to-head:")
    for key in ["h2h_wins_p1", "h2h_wins_p2", "h2h_diff"]:
        if key in stats:
            print(f"   {key}: {stats[key]}")

    # --- Fatiga
    print("\n💤 Fatiga / rendimiento reciente:")
    fatigue_cols = [k for k in stats.keys() if "fatigue" in k]
    for k in fatigue_cols[:5]:  # limitar a 5
        print(f"   {k}: {stats[k]}")

    # --- Otras features (ejemplo)
    print("\n🧾 Otras stats (ejemplo):")
    for k in list(stats.keys())[:10]:  # mostrar primeras 10 columnas
        print(f"   {k}: {stats[k]}")

else:
    print(f"⚠️ Jugador {PLAYER_EXAMPLE} no encontrado en PKL.")
