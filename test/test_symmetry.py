import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from backend.api import app
from backend.data_loader import DataLoader
from backend.feature_builder import FeatureBuilder
from backend.model import TennisPredictor

# Inicialización manual para entorno de test
import backend.api as api_globals
api_globals.data_loader = DataLoader()
api_globals.feature_builder = FeatureBuilder(api_globals.data_loader)
api_globals.data_loader.load_all_data()
api_globals.predictor = TennisPredictor()
api_globals.predictor.load_model()

from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.mark.parametrize("player1,player2,surface", [
    ("Novak Djokovic", "Rafael Nadal", "Clay"),
    ("Carlos Alcaraz", "Daniil Medvedev", "Hard"),
    ("Roger Federer", "Andy Murray", "Grass"),
])
def test_prediction_symmetry(player1, player2, surface):
    # Predicción directa
    response1 = client.post("/predict", json={
        "player1": player1,
        "player2": player2,
        "surface": surface
    })
    if response1.status_code != 200:
        print(f"\n[ERROR] {player1} vs {player2} ({surface}) -> Status: {response1.status_code}")
        print("Response:", response1.text)
    assert response1.status_code == 200
    data1 = response1.json()
    p1_win = data1["player_1_win_probability"]
    p2_win = data1["player_2_win_probability"]

    # Predicción inversa
    response2 = client.post("/predict", json={
        "player1": player2,
        "player2": player1,
        "surface": surface
    })
    if response2.status_code != 200:
        print(f"\n[ERROR] {player2} vs {player1} ({surface}) -> Status: {response2.status_code}")
        print("Response:", response2.text)
    assert response2.status_code == 200
    data2 = response2.json()
    p1_win_inv = data2["player_1_win_probability"]
    p2_win_inv = data2["player_2_win_probability"]

    # --- DEBUG: Comparar features generados en ambos órdenes ---
    debug1 = client.get("/debug_match_features", params={
        "player1": player1,
        "player2": player2,
        "surface": surface
    })
    debug2 = client.get("/debug_match_features", params={
        "player1": player2,
        "player2": player1,
        "surface": surface
    })
    if debug1.status_code == 200 and debug2.status_code == 200:
        features1 = debug1.json()
        features2 = debug2.json()
        # Imprimir cuántas features se pasan al modelo en cada predicción
        if 'all_match_features' in features1:
            print(f"[INFO] Features usados por el modelo ({player1} vs {player2}): {len(features1['all_match_features'])}")
        if 'all_match_features' in features2:
            print(f"[INFO] Features usados por el modelo ({player2} vs {player1}): {len(features2['all_match_features'])}")
        print(f"\n[DEBUG FEATURES] {player1} vs {player2}:")
        # Mostrar features diferenciales completos
        all1 = features1.get("all_match_features", {})
        all2 = features2.get("all_match_features", {})
        for k in all1:
            if k in all2 and "diff" in k:
                v1 = all1[k]
                v2 = all2[k]
                print(f"  {k}: {v1}   |   {k} (inv): {v2}   |   suma: {v1 + v2}")
        # Mostrar balances y ELOs como antes
        for k in features1:
            if k in features2:
                v1 = features1[k]
                v2 = features2[k]
                if any(s in k for s in ["balance", "elo", "h2h"]):
                    print(f"  {k}: {v1}   |   {k} (inv): {v2}")
    else:
        print("[DEBUG] No se pudo obtener features de debug para ambos órdenes.")

    # La suma de probabilidades cruzadas debe ser ≈1
    assert abs(p1_win - p2_win_inv) < 1e-6, f"P1_win({player1}) != P2_win({player2}) inv: {p1_win} vs {p2_win_inv}"
    assert abs(p2_win - p1_win_inv) < 1e-6, f"P2_win({player2}) != P1_win({player1}) inv: {p2_win} vs {p1_win_inv}"
    assert abs((p1_win + p2_win) - 1) < 1e-6, f"Probabilidades no suman 1: {p1_win} + {p2_win}"
    assert abs((p1_win_inv + p2_win_inv) - 1) < 1e-6, f"Probabilidades inversas no suman 1: {p1_win_inv} + {p2_win_inv}"
