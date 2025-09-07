"""
Servicios de predicción y análisis de tenis
NUEVO: Usa features pre-calculadas desde archivo
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import pickle

logger = logging.getLogger(__name__)

class TennisPredictor:
    """Servicio principal de predicción de tenis - Usa features pre-calculadas"""

    def __init__(self, model_cache, data_cache):
        self.model_cache = model_cache
        self.data_cache = data_cache

        # VALIDACIÓN ESTRICTA: todo debe estar presente
        self.model = model_cache.get('xgb_model')
        self.imputer = model_cache.get('imputer')
        self.training_states = model_cache.get('training_states', {})
        self.feature_columns = self.training_states.get('feature_columns', [])

        # FALLO INMEDIATO si algo no está
        if not self.model:
            raise ValueError("MODELO XGBOOST NO ENCONTRADO - La aplicación no puede funcionar")
        if not self.imputer:
            raise ValueError("IMPUTER NO ENCONTRADO - La aplicación no puede funcionar")
        if not self.feature_columns:
            raise ValueError("FEATURE COLUMNS NO ENCONTRADAS - La aplicación no puede funcionar")

        # NUEVO: Cargar features pre-calculadas de jugadores
        self.player_features = self._load_player_features()

        logger.info(f"✅ TennisPredictor inicializado con {len(self.feature_columns)} features")
        logger.info(f"📊 Features pre-calculadas para {len(self.player_features)} jugadores")

    def _load_player_features(self):
        """Cargar features pre-calculadas de jugadores desde archivo"""
        from pathlib import Path

        try:
            outputs_dir = Path(__file__).parent / 'outputs'
            player_features_path = outputs_dir / 'player_features.pkl'

            if not player_features_path.exists():
                logger.warning(f"❌ Archivo de features de jugadores no encontrado: {player_features_path}")
                logger.warning("🔧 Ejecuta el notebook hasta la sección 15 para generarlo")
                return {}

            logger.info(f"📁 Cargando features de jugadores desde {player_features_path}")
            with open(player_features_path, 'rb') as f:
                player_features = pickle.load(f)

            logger.info(f"✅ Features cargadas para {len(player_features)} jugadores")
            return player_features

        except Exception as e:
            logger.error(f"❌ Error cargando features de jugadores: {e}")
            return {}

    def predict_match(self, player1, player2, surface='Hard', tournament_level='ATP500'):
        """
        Predecir resultado usando features pre-calculadas
        """
        try:
            # Verificar que tengamos features para ambos jugadores
            if not self.player_features:
                raise ValueError("No hay features de jugadores cargadas. Ejecuta el notebook hasta la sección 15.")

            # Buscar features de ambos jugadores
            p1_features = self._get_player_features(player1)
            p2_features = self._get_player_features(player2)

            if p1_features is None:
                logger.warning(f"⚠️ Player1 '{player1}' no encontrado en features pre-calculadas")
            if p2_features is None:
                logger.warning(f"⚠️ Player2 '{player2}' no encontrado en features pre-calculadas")

            # Crear features para ambas direcciones del enfrentamiento
            features_1 = self._create_match_features(p1_features, p2_features, surface, player1, player2)
            features_2 = self._create_match_features(p2_features, p1_features, surface, player2, player1)

            # Validar dimensiones
            expected = len(self.feature_columns)
            if len(features_1) != expected or len(features_2) != expected:
                logger.error(f"MISMATCH DE FEATURES: esperado {expected}, obtenido {len(features_1)}/{len(features_2)}")
                # Ajustar features si hay diferencia
                features_1 = self._adjust_features_length(features_1, expected)
                features_2 = self._adjust_features_length(features_2, expected)

            # Preparar arrays e imputar
            arr1 = np.array([features_1])
            arr2 = np.array([features_2])
            arr1_imp = self.imputer.transform(arr1)
            arr2_imp = self.imputer.transform(arr2)

            # Predicciones del modelo
            probs1 = self.model.predict_proba(arr1_imp)[0]
            probs2 = self.model.predict_proba(arr2_imp)[0]

            # Promediado simétrico para evitar bias
            prob_p1_from_direct = float(probs1[1])
            prob_p1_from_inverted = 1.0 - float(probs2[1])
            prob_player1_wins = (prob_p1_from_direct + prob_p1_from_inverted) / 2.0
            prob_player2_wins = 1.0 - prob_player1_wins

            # Información de jugadores y H2D
            player1_info = self._get_player_detailed_info(player1, surface)
            player2_info = self._get_player_detailed_info(player2, surface)
            h2h_info = self._get_h2h_info(player1, player2)

            response = {
                'player1': player1,
                'player2': player2,
                'surface': surface,
                'predictions': {
                    'player1_win_probability': round(prob_player1_wins, 3),
                    'player2_win_probability': round(prob_player2_wins, 3),
                    'predicted_winner': player1 if prob_player1_wins > 0.5 else player2,
                    'confidence': round(abs(prob_player1_wins - 0.5) * 2, 3)
                },
                'player1_info': player1_info,
                'player2_info': player2_info,
                'head_to_head': h2h_info,
                'model_type': 'XGBoost con Features Pre-calculadas',
                'features_used': expected,
                'model_accuracy': self.model_cache.get('model_summary', {}).get('performance', {}).get('accuracy', 'N/A'),
                'status': 'success'
            }

            # Diagnósticos
            response['diagnostics'] = {
                'player1_found_in_features': p1_features is not None,
                'player2_found_in_features': p2_features is not None,
                'features_1_non_zero': int((np.array(features_1) != 0).sum()),
                'features_2_non_zero': int((np.array(features_2) != 0).sum()),
                'total_players_in_db': len(self.player_features),
                'method': 'pre_calculated_features'
            }

            return response

        except Exception as e:
            logger.error(f"ERROR CRÍTICO EN PREDICCIÓN: {e}")
            raise e

    def _get_player_features(self, player_name):
        """
        Buscar features de un jugador (con búsqueda flexible)
        """
        # Búsqueda exacta
        if player_name in self.player_features:
            return self.player_features[player_name]['model_features']

        # Búsqueda parcial por apellido
        player_lower = player_name.lower()
        name_parts = player_lower.split()

        # Buscar por apellido (última parte del nombre)
        if len(name_parts) >= 2:
            surname = name_parts[-1]
            for stored_name in self.player_features.keys():
                if surname in stored_name.lower():
                    return self.player_features[stored_name]['model_features']

        # Buscar por cualquier parte del nombre
        for stored_name in self.player_features.keys():
            stored_lower = stored_name.lower()
            if any(part in stored_lower for part in name_parts):
                return self.player_features[stored_name]['model_features']

        return None  # No encontrado

    def _create_match_features(self, p1_features, p2_features, surface, player1_name, player2_name):
        """
        Crear vector de features para un enfrentamiento específico
        """
        # Si no tenemos features de algún jugador, usar valores neutros
        if p1_features is None:
            p1_features = {col: 0.0 for col in self.feature_columns}
        if p2_features is None:
            p2_features = {col: 0.0 for col in self.feature_columns}

        # Crear vector de features alineado con feature_columns del modelo
        match_features = []

        for col in self.feature_columns:
            # Intentar obtener el valor de features del jugador
            value = 0.0  # Valor por defecto

            # Estrategia: usar features del jugador 1 como base,
            # y complementar con diferencias/ratios respecto al jugador 2
            if col in p1_features:
                value = p1_features[col]
            elif col in p2_features:
                # Si es una feature de "diferencia" o "ratio", usar p2 como referencia
                if 'diff' in col.lower() or 'ratio' in col.lower():
                    value = -p2_features[col]  # Invertir para perspectiva del jugador 1
                else:
                    value = p2_features[col]

            # Para features específicas de superficie, ajustar según superficie actual
            if surface.lower() in col.lower():
                # Dar más peso a features de la superficie específica
                value *= 1.1

            # Asegurar que es un float válido
            try:
                value = float(value)
                if not np.isfinite(value):
                    value = 0.0
            except (ValueError, TypeError):
                value = 0.0

            match_features.append(value)

        return match_features

    def _adjust_features_length(self, features, expected_length):
        """
        Ajustar longitud de features si hay diferencia
        """
        if len(features) < expected_length:
            # Rellenar con ceros
            features.extend([0.0] * (expected_length - len(features)))
        elif len(features) > expected_length:
            # Truncar
            features = features[:expected_length]
        return features

    def _get_player_detailed_info(self, player_name, surface):
        """
        Obtener información detallada de un jugador SOLO desde features pre-calculadas
        """
        # Obtener metadata del archivo de features
        player_metadata = None
        if player_name in self.player_features:
            player_metadata = self.player_features[player_name].get('metadata', {})

        # Si no encontramos metadata, buscar con búsqueda flexible
        if not player_metadata:
            # Búsqueda flexible igual que en _get_player_features
            player_lower = player_name.lower()
            name_parts = player_lower.split()

            # Buscar por apellido
            if len(name_parts) >= 2:
                surname = name_parts[-1]
                for stored_name in self.player_features.keys():
                    if surname in stored_name.lower():
                        player_metadata = self.player_features[stored_name].get('metadata', {})
                        break

            # Buscar por cualquier parte del nombre
            if not player_metadata:
                for stored_name in self.player_features.keys():
                    stored_lower = stored_name.lower()
                    if any(part in stored_lower for part in name_parts):
                        player_metadata = self.player_features[stored_name].get('metadata', {})
                        break

        # Si aún no tenemos metadata, usar valores por defecto
        if not player_metadata:
            player_metadata = {
                'last_match_date': None,
                'total_matches': 0,
                'global_elo': 1500,
                'surface_elos': {'hard': 1500, 'clay': 1500, 'grass': 1500}
            }

        # Obtener ELO para la superficie específica
        surface_elos = player_metadata.get('surface_elos', {})
        surface_elo = surface_elos.get(surface.lower(), player_metadata.get('global_elo', 1500))
        global_elo = player_metadata.get('global_elo', 1500)

        # Información básica del jugador SOLO desde metadata pre-calculada
        player_info = {
            'name': player_name,
            'elo_ratings': {
                'global_elo': int(global_elo),
                'surface_elo': int(surface_elo),
                'elo_difference': int(surface_elo - global_elo)
            },
            'career_stats': {
                'total_matches': player_metadata.get('total_matches', 0),
                'wins': int(player_metadata.get('total_matches', 0) * 0.5),  # Estimación
                'losses': int(player_metadata.get('total_matches', 0) * 0.5),  # Estimación
                'win_rate': 0.5  # Valor neutro
            },
            'recent_form': {
                'recent_matches': 0,
                'recent_wins': 0,
                'recent_win_rate': 0.5,
                'form_string': 'N/A',
                'note': 'Información desde features pre-calculadas'
            },
            'surface_performance': {
                'matches_on_surface': 0,
                'wins_on_surface': 0,
                'surface_win_rate': 0.5,
                'note': 'Información desde features pre-calculadas'
            },
            'last_match_info': {
                'last_match': {
                    'date': player_metadata.get('last_match_date', ''),
                    'tournament': player_metadata.get('last_tournament', 'Unknown'),
                    'opponent': player_metadata.get('last_opponent', 'Unknown'),
                    'surface': player_metadata.get('last_surface', 'Unknown'),
                    'result': 'Unknown',
                    'score': '',
                    'round': ''
                } if player_metadata.get('last_match_date') else None,
                'days_since_last_match': self._calculate_days_since(player_metadata.get('last_match_date'))
            }
        }

        return player_info

    def _get_h2h_info(self, player1, player2):
        """
        Obtener información head-to-head SOLO desde training_states
        """
        h2h_data = self.training_states.get('final_h2h', {})

        # Buscar H2H en training_states
        h2h_key1 = f"{player1}_vs_{player2}"
        h2h_key2 = f"{player2}_vs_{player1}"

        h2h_stats1 = h2h_data.get(h2h_key1, {})
        h2h_stats2 = h2h_data.get(h2h_key2, {})

        # Combinar estadísticas de training_states
        player1_wins = h2h_stats1.get('wins', 0) + h2h_stats2.get('losses', 0)
        player2_wins = h2h_stats1.get('losses', 0) + h2h_stats2.get('wins', 0)
        total_matches = player1_wins + player2_wins

        if total_matches == 0:
            return {
                'total_matches': 0,
                'player1_wins': 0,
                'player2_wins': 0,
                'player1_h2h_rate': 0.5,
                'note': 'No hay enfrentamientos previos registrados',
                'recent_matches': []
            }

        return {
            'total_matches': total_matches,
            'player1_wins': player1_wins,
            'player2_wins': player2_wins,
            'player1_h2h_rate': round(player1_wins / total_matches, 3),
            'player2_h2h_rate': round(player2_wins / total_matches, 3),
            'head_to_head_advantage': player1 if player1_wins > player2_wins else player2 if player2_wins > player1_wins else 'Even',
            'recent_matches': []  # No calculamos matches recientes
        }

    def _calculate_days_since(self, date_str):
        """
        Calcular días desde una fecha
        """
        try:
            if not date_str or pd.isna(date_str) or date_str == 'None':
                return None
            date_obj = pd.to_datetime(date_str)
            days_diff = (pd.Timestamp.now() - date_obj).days
            return int(days_diff)
        except:
            return None

class PlayerAnalyzer:
    """Servicio de análisis de jugadores"""

    def __init__(self, data_cache):
        self.data_cache = data_cache

    def get_player_stats(self, player_name, limit_matches=100):
        """Obtener estadísticas completas de un jugador"""
        try:
            if 'historical_data' not in self.data_cache:
                return None

            df = self.data_cache['historical_data']
            matches = self._get_player_matches(df, player_name, limit_matches)

            if not matches:
                return None

            # Calcular estadísticas básicas
            basic_stats = self._calculate_basic_stats(matches)

            # Estadísticas por superficie
            surface_stats = self._calculate_surface_stats(matches)

            # Forma reciente
            recent_form = self._calculate_recent_form(matches)

            # Rivales más enfrentados
            head_to_head = self._calculate_h2h_stats(matches)

            return {
                'player_name': player_name,
                'basic_stats': basic_stats,
                'surface_stats': surface_stats,
                'recent_form': recent_form,
                'head_to_head': head_to_head,
                'recent_matches': matches[:20]  # Últimos 20 partidos
            }

        except Exception as e:
            logger.error(f"Error analizando jugador {player_name}: {e}")
            return None

    def _get_player_matches(self, df, player_name, limit):
        """Obtener partidos de un jugador"""
        matches = []

        # Buscar en diferentes formatos de columnas
        for winner_col, loser_col in [('winner_name', 'loser_name'), ('player_1', 'player_2')]:
            if winner_col in df.columns and loser_col in df.columns:
                # Partidos ganados
                won_matches = df[df[winner_col].str.contains(player_name, case=False, na=False)]
                for _, match in won_matches.iterrows():
                    matches.append(self._format_match(match, winner_col, loser_col, 'Won'))

                # Partidos perdidos
                lost_matches = df[df[loser_col].str.contains(player_name, case=False, na=False)]
                for _, match in lost_matches.iterrows():
                    matches.append(self._format_match(match, winner_col, loser_col, 'Lost'))
                break

        # Ordenar por fecha (más reciente primero) y limitar
        matches.sort(key=lambda x: x.get('date', ''), reverse=True)
        return matches[:limit]

    def _format_match(self, match, winner_col, loser_col, result):
        """Formatear información de un partido"""
        opponent = match.get(loser_col if result == 'Won' else winner_col, 'Unknown')

        return {
            'date': match.get('tourney_date', ''),
            'tournament': match.get('tourney_name', 'Unknown'),
            'surface': match.get('surface', 'Unknown'),
            'opponent': opponent,
            'result': result,
            'score': match.get('score', ''),
            'round': match.get('round', ''),
            'tournament_level': match.get('tourney_level', '')
        }

    def _calculate_basic_stats(self, matches):
        """Calcular estadísticas básicas"""
        total = len(matches)
        wins = sum(1 for m in matches if m['result'] == 'Won')

        return {
            'total_matches': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': round(wins / total if total > 0 else 0, 3)
        }

    def _calculate_surface_stats(self, matches):
        """Calcular estadísticas por superficie"""
        surface_stats = defaultdict(lambda: {'wins': 0, 'losses': 0})

        for match in matches:
            surface = match['surface']
            if match['result'] == 'Won':
                surface_stats[surface]['wins'] += 1
            else:
                surface_stats[surface]['losses'] += 1

        # Calcular win rates
        result = {}
        for surface, stats in surface_stats.items():
            total = stats['wins'] + stats['losses']
            result[surface] = {
                'wins': stats['wins'],
                'losses': stats['losses'],
                'total': total,
                'win_rate': round(stats['wins'] / total if total > 0 else 0, 3)
            }

        return result

    def _calculate_recent_form(self, matches):
        """Calcular forma reciente (últimos 10 partidos)"""
        recent = matches[:10]
        if not recent:
            return {'wins': 0, 'losses': 0, 'win_rate': 0, 'form_string': ''}

        wins = sum(1 for m in recent if m['result'] == 'Won')
        form_string = ''.join('W' if m['result'] == 'Won' else 'L' for m in recent)

        return {
            'wins': wins,
            'losses': len(recent) - wins,
            'win_rate': round(wins / len(recent), 3),
            'form_string': form_string
        }

    def _calculate_h2h_stats(self, matches):
        """Calcular estadísticas head-to-head"""
        h2h = defaultdict(lambda: {'wins': 0, 'losses': 0})

        for match in matches:
            opponent = match['opponent']
            if opponent and opponent != 'Unknown':
                if match['result'] == 'Won':
                    h2h[opponent]['wins'] += 1
                else:
                    h2h[opponent]['losses'] += 1

        # Convertir a lista ordenada por total de enfrentamientos
        h2h_list = []
        for opponent, stats in h2h.items():
            total = stats['wins'] + stats['losses']
            if total >= 2:  # Solo mostrar rivales enfrentados al menos 2 veces
                h2h_list.append({
                    'opponent': opponent,
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'total': total,
                    'win_rate': round(stats['wins'] / total, 3)
                })

        return sorted(h2h_list, key=lambda x: x['total'], reverse=True)[:10]

class TournamentAnalyzer:
    """Servicio de análisis de torneos"""

    def __init__(self, data_cache):
        self.data_cache = data_cache

    def get_tournament_stats(self):
        """Obtener estadísticas de torneos"""
        try:
            if 'historical_data' not in self.data_cache:
                return []

            df = self.data_cache['historical_data']

            if 'tourney_name' not in df.columns:
                return []

            # Agrupar por torneo
            tournament_stats = df.groupby('tourney_name').agg({
                'tourney_name': 'count',  # Total matches
                'surface': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown',
                'tourney_level': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown',
                'tourney_date': ['min', 'max']
            }).reset_index()

            tournament_stats.columns = ['tournament', 'total_matches', 'surface', 'level', 'first_year', 'last_year']

            # Convertir a lista de diccionarios
            tournaments = []
            for _, row in tournament_stats.iterrows():
                tournaments.append({
                    'name': row['tournament'],
                    'total_matches': int(row['total_matches']),
                    'surface': row['surface'],
                    'level': row['level'],
                    'years_active': f"{row['first_year']} - {row['last_year']}"
                })

            return sorted(tournaments, key=lambda x: x['total_matches'], reverse=True)

        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de torneos: {e}")
            return []
