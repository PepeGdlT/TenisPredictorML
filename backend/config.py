"""
Configuration settings for the Tennis Prediction API.
"""
import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    OUTPUTS_DIR = BASE_DIR / "outputs"

    # Data files
    PLAYER_FEATURES_FILE = "player_last_features.pkl"
    SURFACE_ELOS_FILE = "surface_elos_by_player.json"
    H2H_DATA_FILE = "h2h_all.json"

    # Model files (in order of preference)
    MODEL_FILES = [
        "calibrated_xgb_model.pkl",
        "best_xgb_model.pkl"
        "ensemble_model.pkl",
    ]

    FEATURE_IMPUTER_FILE = "feature_imputer.pkl"

    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Validation settings
    MIN_FEATURES_COUNT = 50
    MAX_EXTREME_VALUE = 1e6

    # Default values
    DEFAULT_ELO = 1500.0
    DEFAULT_AGE = 25.0
    DEFAULT_RANK = 100
    DEFAULT_HEIGHT = 180
    DEFAULT_DAYS_SINCE_LAST = 7
