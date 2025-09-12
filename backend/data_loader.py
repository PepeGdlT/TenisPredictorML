"""
Data loader module for tennis prediction backend.
Handles loading of player features, surface ELO, and H2H data.
"""
import pickle
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, base_path: str = "outputs"):
        self.base_path = Path(base_path)
        self.player_features = None
        self.surface_elos = None
        self.h2h_data = None
        self.feature_columns = None

    def load_player_features(self, filename: str = "player_last_features.pkl") -> pd.DataFrame:
        """Load player features from PKL file."""
        try:
            pkl_path = self.base_path / filename
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)

            logger.info(f"Loaded PKL data: {type(pkl_data)}")

            # Verificar si es la estructura del notebook (con claves player_data_final, etc.)
            if isinstance(pkl_data, dict) and 'player_data_final' in pkl_data:
                logger.info("Detected notebook-generated PKL structure")

                # Extraer los datos de jugadores
                self.player_features = pkl_data['player_data_final']

                # Extraer las columnas de features del modelo si están disponibles
                if 'model_features' in pkl_data:
                    self.feature_columns = pkl_data['model_features']
                    logger.info(f"Model features loaded: {len(self.feature_columns)}")

                # Log del summary si existe
                if 'summary' in pkl_data:
                    summary = pkl_data['summary']
                    logger.info(f"PKL Summary - Players: {summary.get('total_players')}, "
                               f"Features: {summary.get('features_count_final')}")

                logger.info(f"Player data type: {type(self.player_features)}")
                logger.info(f"Available players: {len(self.player_features) if isinstance(self.player_features, dict) else 'unknown'}")

            else:
                # Estructura antigua - datos directos
                self.player_features = pkl_data

                if isinstance(self.player_features, pd.DataFrame):
                    self.feature_columns = list(self.player_features.columns)
                    logger.info(f"DataFrame shape: {self.player_features.shape}")

            return self.player_features

        except Exception as e:
            logger.error(f"Error loading player features: {e}")
            raise

    def load_surface_elos(self, filename: str = "surface_elos_by_player.json") -> Dict:
        """Load surface ELO ratings from JSON file."""
        try:
            json_path = self.base_path / filename
            with open(json_path, 'r') as f:
                self.surface_elos = json.load(f)

            logger.info(f"Loaded surface ELOs for {len(self.surface_elos)} players")
            return self.surface_elos

        except Exception as e:
            logger.error(f"Error loading surface ELOs: {e}")
            self.surface_elos = {}
            return self.surface_elos

    def load_h2h_data(self, filename: str = "h2h_all.json") -> Dict:
        """Load head-to-head data from JSON file."""
        try:
            json_path = self.base_path / filename
            with open(json_path, 'r') as f:
                self.h2h_data = json.load(f)

            logger.info(f"Loaded H2H data with {len(self.h2h_data)} entries")
            return self.h2h_data

        except Exception as e:
            logger.error(f"Error loading H2H data: {e}")
            self.h2h_data = {}
            return self.h2h_data

    def get_player_features(self, player_name: str) -> Optional[pd.Series]:
        """Get features for a specific player."""
        if self.player_features is None:
            raise ValueError("Player features not loaded. Call load_player_features() first.")

        # LOG: Debug del jugador que estamos buscando
        logger.info(f"🔍 SEARCHING for player: '{player_name}'")

        if isinstance(self.player_features, pd.DataFrame):
            # Buscar por índice o por columna 'player_name' si existe
            if player_name in self.player_features.index:
                result = self.player_features.loc[player_name]
                logger.info(f"✅ Found {player_name} in DataFrame index")
                return result
            elif 'player_name' in self.player_features.columns:
                matches = self.player_features[self.player_features['player_name'] == player_name]
                if not matches.empty:
                    result = matches.iloc[0]
                    logger.info(f"✅ Found {player_name} in DataFrame column")
                    return result

        elif isinstance(self.player_features, dict):
            # Nueva estructura del notebook: cada jugador tiene 'last_features', 'feature_names', etc.

            # LOG: Mostrar las primeras claves disponibles para debug
            available_keys = list(self.player_features.keys())[:10]
            logger.info(f"📋 Available player keys (first 10): {available_keys}")

            if player_name in self.player_features:
                player_data = self.player_features[player_name]
                logger.info(f"✅ Found {player_name} in dict, data type: {type(player_data)}")

                if isinstance(player_data, dict):
                    # Estructura del notebook
                    if 'last_features' in player_data and 'feature_names' in player_data:
                        features_array = player_data['last_features']
                        feature_names = player_data['feature_names']

                        # LOG: Verificar datos básicos del jugador
                        if 'p1_age' in feature_names:
                            age_idx = feature_names.index('p1_age')
                            logger.info(f"🎂 {player_name} age from data: {features_array[age_idx]}")
                        if 'p1_rank' in feature_names:
                            rank_idx = feature_names.index('p1_rank')
                            logger.info(f"🏆 {player_name} rank from data: {features_array[rank_idx]}")

                        # Crear Series con los nombres de features correctos
                        if len(features_array) == len(feature_names):
                            result = pd.Series(data=features_array, index=feature_names)
                            logger.info(f"✅ Created Series for {player_name} with {len(feature_names)} features")
                            return result
                        else:
                            logger.warning(f"Mismatch: {len(features_array)} values vs {len(feature_names)} names for {player_name}")
                            return pd.Series(data=features_array)

                    # Estructura simple: dict directo
                    else:
                        logger.info(f"📊 Using direct dict structure for {player_name}")
                        return pd.Series(player_data)

                elif isinstance(player_data, pd.Series):
                    logger.info(f"📊 Using Series structure for {player_name}")
                    return player_data
                else:
                    # Array o lista
                    logger.info(f"📊 Converting array/list to Series for {player_name}")
                    return pd.Series(player_data)
            else:
                # LOG: Buscar coincidencias parciales para debug
                partial_matches = [key for key in self.player_features.keys() if player_name.lower() in key.lower()]
                if partial_matches:
                    logger.warning(f"❓ Exact match not found for '{player_name}', but found partial matches: {partial_matches[:5]}")
                else:
                    logger.warning(f"❌ No matches found for '{player_name}' at all")

        logger.error(f"❌ Player {player_name} not found in features data")
        return None

    def get_surface_elo(self, player_name: str, surface: str) -> Optional[float]:
        """Get surface-specific ELO for a player."""
        if self.surface_elos is None:
            logger.warning("Surface ELOs not loaded")
            return None

        player_elos = self.surface_elos.get(player_name, {})
        if isinstance(player_elos, dict):
            return player_elos.get(surface)

        return None

    def get_h2h_stats(self, player1: str, player2: str, surface: str = None) -> Dict[str, Any]:
        """Get head-to-head statistics between two players."""
        if self.h2h_data is None:
            logger.warning("H2H data not loaded")
            return self._get_default_h2h()

        # Estructura del notebook: 'global' y 'by_surface'
        if isinstance(self.h2h_data, dict) and 'global' in self.h2h_data:
            # Nuevo formato del notebook
            global_h2h = self.h2h_data.get('global', {})
            surface_h2h = self.h2h_data.get('by_surface', {})

            # Probar ambas combinaciones de nombres con separador |||
            key1 = f"{player1}|||{player2}"
            key2 = f"{player2}|||{player1}"

            h2h_stats = global_h2h.get(key1) or global_h2h.get(key2)

            if h2h_stats is None:
                logger.warning(f"No H2H data found for {player1} vs {player2}")
                return self._get_default_h2h()

            # Calcular estadísticas H2H
            total_matches = h2h_stats.get('count', 0)
            wins_dict = h2h_stats.get('wins', {})

            player1_wins = wins_dict.get(player1, 0)
            player2_wins = wins_dict.get(player2, 0)

            # IMPORTANTE: Balance desde perspectiva de player1
            h2h_balance = player1_wins / total_matches if total_matches > 0 else 0.5

            result = {
                'h2h_count': total_matches,
                'h2h_balance': h2h_balance,
                'h2h_recent3_balance': h2h_balance,  # Simplificación por ahora
            }

            # Estadísticas por superficie si se especifica
            if surface:
                surface_key1 = f"{player1}|||{player2}|||{surface}"
                surface_key2 = f"{player2}|||{player1}|||{surface}"

                surface_stats = surface_h2h.get(surface_key1) or surface_h2h.get(surface_key2)

                if surface_stats:
                    surface_total = surface_stats.get('count', 0)
                    surface_wins_dict = surface_stats.get('wins', {})
                    surface_p1_wins = surface_wins_dict.get(player1, 0)

                    # IMPORTANTE: Surface balance desde perspectiva de player1
                    surface_balance = surface_p1_wins / surface_total if surface_total > 0 else 0.5

                    result.update({
                        'h2h_surface_count': surface_total,
                        'h2h_surface_balance': surface_balance
                    })
                else:
                    result.update({
                        'h2h_surface_count': 0,
                        'h2h_surface_balance': 0.5
                    })
            else:
                result.update({
                    'h2h_surface_count': 0,
                    'h2h_surface_balance': 0.5
                })

            # LOG: Debug H2H calculation
            logger.info(f"H2H {player1} vs {player2}: total={total_matches}, p1_wins={player1_wins}, balance={h2h_balance}")

            return result

        else:
            # Formato antiguo (fallback)
            key1 = f"{player1}_vs_{player2}"
            key2 = f"{player2}_vs_{player1}"

            h2h_stats = self.h2h_data.get(key1) or self.h2h_data.get(key2)

            if h2h_stats is None:
                logger.warning(f"No H2H data found for {player1} vs {player2}")
                return self._get_default_h2h()

            # Si necesitamos estadísticas específicas de superficie
            if surface and isinstance(h2h_stats, dict):
                surface_stats = h2h_stats.get(f"surface_{surface.lower()}", {})
                # Combinar estadísticas generales con específicas de superficie
                result = self._get_default_h2h()
                result.update(h2h_stats)
                if surface_stats:
                    result.update({f"surface_{k}": v for k, v in surface_stats.items()})
                return result

            return h2h_stats if isinstance(h2h_stats, dict) else self._get_default_h2h()

    def _get_default_h2h(self) -> Dict[str, Any]:
        """Get default H2H statistics when data is not available."""
        return {
            'h2h_count': 0,
            'h2h_balance': 0.5,  # 50-50 si no hay historial
            'h2h_recent3_balance': 0.5,
            'h2h_surface_count': 0,
            'h2h_surface_balance': 0.5
        }

    def get_required_feature_columns(self) -> list:
        """Return ONLY the model features (49), never all columns."""
        if self.feature_columns is not None:
            logger.info(f"[DEBUG] get_required_feature_columns returns {len(self.feature_columns)} model features.")
            return self.feature_columns
        else:
            logger.warning("[DEBUG] get_required_feature_columns: No model features found, returning empty list.")
            return []

    def load_all_data(self):
        """Load all required data files."""
        logger.info("Loading all data files...")
        self.load_player_features()
        self.load_surface_elos()
        self.load_h2h_data()
        logger.info("All data files loaded successfully")
