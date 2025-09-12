import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import pytest
from backend.api import app
from backend.data_loader import DataLoader
from backend.feature_builder import FeatureBuilder
from fastapi.testclient import TestClient

# Ruta al pickle generado por el pipeline
PKL_PATH = os.path.join('outputs', 'player_last_features.pkl')

@pytest.mark.parametrize("player1, player2, surface", [
    ("Novak Djokovic", "Rafael Nadal", "Clay"),
    ("Carlos Alcaraz", "Daniil Medvedev", "Hard"),
    ("Roger Federer", "Andy Murray", "Grass"),
])
def test_api_features_vs_model_features(player1, player2, surface):
    # 1. Cargar features del modelo desde el pickle
    assert os.path.exists(PKL_PATH), f"No se encontró el pickle: {PKL_PATH}"
    with open(PKL_PATH, 'rb') as f:
        pkl = pickle.load(f)
    model_features = set(pkl['model_features'])

    # 2. Inicializar API y generar features para el partido
    data_loader = DataLoader()
    data_loader.load_all_data()
    feature_builder = FeatureBuilder(data_loader)
    match_features = feature_builder.build_match_features(player1, player2, surface)
    api_features = set(match_features.index)

    # 3. Comparar features
    extra_in_api = api_features - model_features
    missing_in_api = model_features - api_features

    print(f"\n[TEST] {player1} vs {player2} ({surface})")
    print(f"Features generadas por la API: {len(api_features)}")
    print(f"Features requeridas por el modelo: {len(model_features)}")
    if extra_in_api:
        print(f"\n⚠️ Features generadas por la API pero no usadas por el modelo: {sorted(list(extra_in_api))}")
    if missing_in_api:
        print(f"\n❌ Features requeridas por el modelo pero NO generadas por la API: {sorted(list(missing_in_api))}")
    assert not missing_in_api, f"Faltan features requeridas por el modelo: {missing_in_api}"
    # No es error grave si hay features extra en la API, pero se recomienda limpiar antes de pasar al modelo.

