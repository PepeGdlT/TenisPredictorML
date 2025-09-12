"""
Feature builder module for tennis prediction backend.
Handles dynamic calculation of match-specific features between two players.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def build_match_features(self, player1: str, player2: str, surface: str,
                           tournament_info: Dict = None) -> pd.Series:
        """
        Build complete feature vector for a match between two players.
        """
        logger.info(f"Building features for {player1} vs {player2} on {surface}")

        # Obtener features base de ambos jugadores
        p1_features = self.data_loader.get_player_features(player1)
        p2_features = self.data_loader.get_player_features(player2)

        if p1_features is None or p2_features is None:
            raise ValueError(f"Features not found for one or both players: {player1}, {player2}")

        # Obtener información H2H
        h2h_stats = self.data_loader.get_h2h_stats(player1, player2, surface)

        # Crear el vector de features final
        match_features = {}

        # 1. Copiar features individuales de cada jugador (renombrar p1_ y p2_)
        match_features.update(self._extract_player_features(p1_features, "p1"))
        match_features.update(self._extract_player_features(p2_features, "p2"))

        # 2. Calcular features de diferencias y ratios
        match_features.update(self._calculate_diff_features(p1_features, p2_features))

        # 3. Integrar Surface ELO
        match_features.update(self._integrate_surface_elo(player1, player2, surface))

        # 4. Integrar H2H features
        match_features.update(self._integrate_h2h_features(h2h_stats))

        # 5. Calcular features específicas del match
        match_features.update(self._calculate_match_specific_features(
            p1_features, p2_features, surface, tournament_info))

        # 6. Agregar identificadores
        match_features['player_1_id'] = player1
        match_features['player_2_id'] = player2

        # 7. LOG: Mostrar características clave para debug de simetría
        key_features_for_debug = [
            'elo_diff_clip', 'elo_prob', 'h2h_balance', 'rank_diff',
            'streak_wins_diff', 'form_last5_diff', 'matches_last7d_diff',
            # AGREGAR MÁS CARACTERÍSTICAS PARA DEBUG COMPLETO
            'surface_elo_diff_clip', 'surface_elo_prob', 'elo_vol5_diff',
            'p1_ace_mean', 'p2_ace_mean', 'p1_1stWon_mean', 'p2_1stWon_mean'
        ]
        debug_values = {}
        for feat in key_features_for_debug:
            if feat in match_features:
                debug_values[feat] = match_features[feat]

        logger.info(f"🔧 KEY FEATURES for {player1} vs {player2}: {debug_values}")

        # 7.5. VERIFICACIÓN DE SIMETRÍA CRÍTICA
        # Si tenemos valores idénticos para características que deberían ser diferentes, hay un problema
        p1_individual_sample = {
            'p1_ace_mean': match_features.get('p1_ace_mean', 0),
            'p1_1stWon_mean': match_features.get('p1_1stWon_mean', 0),
            'p1_rank': match_features.get('p1_rank', 0)
        }
        p2_individual_sample = {
            'p2_ace_mean': match_features.get('p2_ace_mean', 0),
            'p2_1stWon_mean': match_features.get('p2_1stWon_mean', 0),
            'p2_rank': match_features.get('p2_rank', 0)
        }

        logger.info(f"🎾 P1 INDIVIDUAL FEATURES: {p1_individual_sample}")
        logger.info(f"🎾 P2 INDIVIDUAL FEATURES: {p2_individual_sample}")

        # 8. Asegurar que tenemos todas las columnas requeridas
        required_columns = self.data_loader.get_required_feature_columns()
        complete_features = self._ensure_all_columns(match_features, required_columns)

        # Filtrar solo las features requeridas (las seleccionadas para el modelo)
        filtered_features = {k: complete_features[k] for k in required_columns}
        logger.info(f"[DEBUG] Features finales para el modelo: {list(filtered_features.keys())} - Total: {len(filtered_features)}")
        print(f"[DEBUG] build_match_features returns {len(filtered_features)} features: {list(filtered_features.keys())}")
        return pd.Series(filtered_features)

    def _extract_player_features(self, player_features: pd.Series, prefix: str) -> Dict:
        """Extract and rename player-specific features."""
        extracted = {}

        # Features individuales que van directamente con prefijo
        individual_features = [
            'age', 'rank', 'rank_points', 'ht', 'streak_wins',
            'days_since_last', 'days_since_surface', 'matches_last3d',
            'matches_last7d', 'matches_last14d', 'surface_matches_last30d',
            'form_winrate_last5', 'form_winrate_last10', 'surface_wr_all',
            'surface_wr_last5', '1stIn', '1stIn_mean', '1stWon', '1stWon_mean',
            '2ndWon', '2ndWon_mean', 'SvGms', 'ace', 'ace_mean', 'ace_surface_mean',
            'bpFaced', 'bpFaced_mean', 'bpSaved', 'bpSaved_mean', 'df', 'df_mean',
            'svpt', 'svpt_mean'
        ]

        # LOG: Debug del jugador actual
        logger.info(f"Extracting features for {prefix}, available features: {len(player_features)}")

        for feature in individual_features:
            target_key = f"{prefix}_{feature}"
            value = None

            # CORRECCIÓN CRÍTICA: En el PKL, TODOS los jugadores tienen sus datos con prefijo p1_
            # Necesitamos buscar SIEMPRE con p1_ sin importar si es p1 o p2
            possible_keys = [
                f"p1_{feature}",            # SIEMPRE buscar p1_ primero (estructura del PKL)
                feature,                    # Fallback: 'age', 'rank', etc.
                f"{prefix}_{feature}",      # Fallback: 'p2_age' si ya existe
            ]

            for key in possible_keys:
                if key in player_features.index and pd.notna(player_features[key]):
                    value = player_features[key]
                    break

            # Si no encontramos la feature, usar valor por defecto
            if value is None or pd.isna(value):
                value = self._get_default_value(feature)
                logger.warning(f"Feature {feature} not found for {prefix}, using default: {value}")

            extracted[target_key] = float(value) if isinstance(value, (int, float, np.number)) else value

        # LOG: Verificar que se extrajeron features correctamente
        logger.info(f"Extracted {len(extracted)} features for {prefix}")
        logger.info(f"Sample extracted features: {dict(list(extracted.items())[:5])}")

        # LOG ADICIONAL: Verificar valores críticos para debug
        if f"{prefix}_age" in extracted:
            logger.info(f"🎂 EXTRACTED {prefix} age: {extracted[f'{prefix}_age']}")
        if f"{prefix}_rank" in extracted:
            logger.info(f"🏆 EXTRACTED {prefix} rank: {extracted[f'{prefix}_rank']}")

        return extracted

    def _calculate_diff_features(self, p1_features: pd.Series, p2_features: pd.Series) -> Dict:
        """Calculate difference and ratio features between players."""
        diff_features = {}

        # Rankings - CORREGIDO COMPLETAMENTE: cada jugador busca SUS propias características
        p1_rank = self._safe_get(p1_features, 'p1_rank', 'rank')
        p2_rank = self._safe_get(p2_features, 'p1_rank', 'rank')  # P2 busca 'p1_rank' en SUS datos (porque en su PKL está como p1_rank)

        # LOG: Debug de rankings
        logger.info(f"P1 rank found: {p1_rank}, P2 rank found: {p2_rank}")

        if p1_rank and p2_rank:
            diff_features['rank_diff'] = p1_rank - p2_rank
            diff_features['rank_ratio'] = p1_rank / p2_rank if p2_rank > 0 else 1.0
            diff_features['log_rank_ratio'] = np.log(p1_rank / p2_rank) if p2_rank > 0 else 0.0

        # Form differences - cada jugador busca en SUS propios datos
        p1_form5 = self._safe_get(p1_features, 'p1_form_winrate_last5', 'form_winrate_last5')
        p2_form5 = self._safe_get(p2_features, 'p1_form_winrate_last5', 'form_winrate_last5')
        if p1_form5 is not None and p2_form5 is not None:
            diff_features['form_last5_diff'] = p1_form5 - p2_form5

        p1_form10 = self._safe_get(p1_features, 'p1_form_winrate_last10', 'form_winrate_last10')
        p2_form10 = self._safe_get(p2_features, 'p1_form_winrate_last10', 'form_winrate_last10')
        if p1_form10 is not None and p2_form10 is not None:
            diff_features['form_last10_diff'] = p1_form10 - p2_form10

        # Diferencias de actividad - cada jugador busca en SUS propios datos
        activity_features = ['matches_last3d', 'matches_last7d', 'matches_last14d']
        for feat in activity_features:
            p1_val = self._safe_get(p1_features, f'p1_{feat}', feat)
            p2_val = self._safe_get(p2_features, f'p1_{feat}', feat)

            # LOG: Debug de activity features
            logger.info(f"Activity {feat}: P1={p1_val}, P2={p2_val}")

            if p1_val is not None and p2_val is not None:
                diff_features[f'{feat}_diff'] = p1_val - p2_val
                if feat in ['matches_last3d', 'matches_last7d']:
                    diff_features[f'{feat}_diff_scaled'] = (p1_val - p2_val) / max(1, p1_val + p2_val)

        # Days since last match difference
        p1_days = self._safe_get(p1_features, 'p1_days_since_last', 'days_since_last')
        p2_days = self._safe_get(p2_features, 'p1_days_since_last', 'days_since_last')
        if p1_days is not None and p2_days is not None:
            diff_features['days_since_last_diff'] = p1_days - p2_days
            diff_features['days_since_last_diff_scaled'] = (p1_days - p2_days) / max(1, p1_days + p2_days)

        # Days since surface match difference
        p1_surf_days = self._safe_get(p1_features, 'p1_days_since_surface', 'days_since_surface')
        p2_surf_days = self._safe_get(p2_features, 'p1_days_since_surface', 'days_since_surface')
        if p1_surf_days is not None and p2_surf_days is not None:
            diff_features['days_since_surface_diff'] = p1_surf_days - p2_surf_days
            diff_features['days_since_surface_diff_scaled'] = (p1_surf_days - p2_surf_days) / max(1, p1_surf_days + p2_surf_days)

        # Surface-specific differences
        surface_features = ['surface_matches_last30d', 'surface_wr_all', 'surface_wr_last5']
        for feat in surface_features:
            p1_val = self._safe_get(p1_features, f'p1_{feat}', feat)
            p2_val = self._safe_get(p2_features, f'p1_{feat}', feat)
            if p1_val is not None and p2_val is not None:
                if feat == 'surface_matches_last30d':
                    diff_features['surface_matches_30d_diff'] = p1_val - p2_val
                else:
                    diff_features[f'{feat}_diff'] = p1_val - p2_val

        # Streak wins difference
        p1_streak = self._safe_get(p1_features, 'p1_streak_wins', 'streak_wins')
        p2_streak = self._safe_get(p2_features, 'p1_streak_wins', 'streak_wins')
        if p1_streak is not None and p2_streak is not None:
            diff_features['streak_wins_diff'] = p1_streak - p2_streak

        # Service stats ratios - AGREGAR LOS RATIOS QUE FALTAN
        p1_svpt = self._safe_get(p1_features, 'p1_svpt_mean', 'svpt_mean')
        p2_svpt = self._safe_get(p2_features, 'p1_svpt_mean', 'svpt_mean')
        if p1_svpt and p2_svpt and p2_svpt > 0:
            diff_features['svpt_ratio'] = p1_svpt / p2_svpt

        # AGREGAR LOS RATIOS QUE ESTÁN FALTANDO Y CAUSAN WARNINGS
        # 1stIn ratio
        p1_1st_in = self._safe_get(p1_features, 'p1_1stIn_mean', '1stIn_mean', default=0.6)
        p2_1st_in = self._safe_get(p2_features, 'p1_1stIn_mean', '1stIn_mean', default=0.6)
        diff_features['1stIn_ratio'] = p1_1st_in / p2_1st_in if p2_1st_in > 0 else 1.0

        # 1stWon ratio
        p1_1st_won = self._safe_get(p1_features, 'p1_1stWon_mean', '1stWon_mean', default=0.7)
        p2_1st_won = self._safe_get(p2_features, 'p1_1stWon_mean', '1stWon_mean', default=0.7)
        diff_features['1stWon_ratio'] = p1_1st_won / p2_1st_won if p2_1st_won > 0 else 1.0

        # 2ndWon ratio
        p1_2nd_won = self._safe_get(p1_features, 'p1_2ndWon_mean', '2ndWon_mean', default=0.5)
        p2_2nd_won = self._safe_get(p2_features, 'p1_2ndWon_mean', '2ndWon_mean', default=0.5)
        diff_features['2ndWon_ratio'] = p1_2nd_won / p2_2nd_won if p2_2nd_won > 0 else 1.0

        # Ace ratio
        p1_ace = self._safe_get(p1_features, 'p1_ace_mean', 'ace_mean', default=0.05)
        p2_ace = self._safe_get(p2_features, 'p1_ace_mean', 'ace_mean', default=0.05)
        diff_features['ace_ratio'] = p1_ace / p2_ace if p2_ace > 0 else 1.0

        # DF ratio
        p1_df = self._safe_get(p1_features, 'p1_df_mean', 'df_mean', default=0.03)
        p2_df = self._safe_get(p2_features, 'p1_df_mean', 'df_mean', default=0.03)
        diff_features['df_ratio'] = p1_df / p2_df if p2_df > 0 else 1.0

        # BP Faced ratio
        p1_bp_faced = self._safe_get(p1_features, 'p1_bpFaced_mean', 'bpFaced_mean', default=0.2)
        p2_bp_faced = self._safe_get(p2_features, 'p1_bpFaced_mean', 'bpFaced_mean', default=0.2)
        diff_features['bpFaced_ratio'] = p1_bp_faced / p2_bp_faced if p2_bp_faced > 0 else 1.0

        # BP Saved ratio
        p1_bp_saved = self._safe_get(p1_features, 'p1_bpSaved_mean', 'bpSaved_mean', default=0.6)
        p2_bp_saved = self._safe_get(p2_features, 'p1_bpSaved_mean', 'bpSaved_mean', default=0.6)
        diff_features['bpSaved_ratio'] = p1_bp_saved / p2_bp_saved if p2_bp_saved > 0 else 1.0

        return diff_features

    def _integrate_surface_elo(self, player1: str, player2: str, surface: str) -> Dict:
        """Integrate surface-specific ELO ratings."""
        surface_elo_features = {}

        # Obtener ELOs de superficie
        p1_surface_elo = self.data_loader.get_surface_elo(player1, surface)
        p2_surface_elo = self.data_loader.get_surface_elo(player2, surface)

        # LOG: Debug de los Surface ELOs cargados
        logger.info(f"🏟️ Surface ELO {surface}: {player1}={p1_surface_elo}, {player2}={p2_surface_elo}")

        # Si no tenemos ELO de superficie, usar ELO general como base más realista
        if p1_surface_elo is None:
            # Usar ELO general del jugador si está disponible
            p1_general_elo = self._safe_get(self.data_loader.get_player_features(player1), 'elo_p1', 'elo', default=1500)
            p1_surface_elo = p1_general_elo  # Usar ELO general como superficie
            logger.warning(f"No surface ELO found for {player1} on {surface}, using general ELO: {p1_surface_elo}")

        if p2_surface_elo is None:
            # Usar ELO general del jugador si está disponible
            p2_general_elo = self._safe_get(self.data_loader.get_player_features(player2), 'elo_p1', 'elo', default=1500)
            p2_surface_elo = p2_general_elo  # Usar ELO general como superficie
            logger.warning(f"No surface ELO found for {player2} on {surface}, using general ELO: {p2_surface_elo}")

        surface_elo_features['surface_elo_p1'] = p1_surface_elo
        surface_elo_features['surface_elo_p2'] = p2_surface_elo

        # Calcular diferencias y probabilidades CON NORMALIZACIÓN
        elo_diff = p1_surface_elo - p2_surface_elo

        # CORRECCIÓN CRÍTICA: Normalizar la diferencia de Surface ELO para evitar dominancia extrema
        # Aplicar un factor de dampening para diferencias muy grandes
        normalized_elo_diff = elo_diff * 0.5  # Reducir el impacto a la mitad

        surface_elo_features['surface_elo_diff'] = normalized_elo_diff
        surface_elo_features['surface_elo_diff_clip'] = np.clip(normalized_elo_diff, -200, 200)  # Clip más conservador
        surface_elo_features['surface_elo_diff_sq'] = normalized_elo_diff ** 2

        # Probabilidad basada en ELO NORMALIZADO
        surface_elo_features['surface_elo_prob'] = 1 / (1 + 10**(-normalized_elo_diff/400))

        # LOG: Debug de las características calculadas
        logger.info(f"🏟️ Original ELO DIFF: {elo_diff:.2f} -> Normalized: {normalized_elo_diff:.2f}")
        logger.info(f"🏟️ Surface ELO PROB: {surface_elo_features['surface_elo_prob']:.3f}")

        # Volatilidades (usar valores por defecto si no están disponibles)
        surface_elo_features['surface_elo_vol5_p1'] = 50.0  # Default volatility
        surface_elo_features['surface_elo_vol5_p2'] = 50.0
        surface_elo_features['surface_elo_vol5_diff'] = 0.0

        return surface_elo_features

    def _integrate_h2h_features(self, h2h_stats: Dict, invert: bool = False) -> Dict:
        """Integrate head-to-head statistics. Si invert=True, invierte los balances."""
        h2h_features = {}

        h2h_features['h2h_count'] = h2h_stats.get('h2h_count', 0)
        # Invertir balances si corresponde
        balance = h2h_stats.get('h2h_balance', 0.5)
        h2h_features['h2h_balance'] = 1 - balance if invert else balance
        recent3_balance = h2h_stats.get('h2h_recent3_balance', 0.5)
        h2h_features['h2h_recent3_balance'] = 1 - recent3_balance if invert else recent3_balance
        h2h_features['h2h_surface_count'] = h2h_stats.get('h2h_surface_count', 0)
        surface_balance = h2h_stats.get('h2h_surface_balance', 0.5)
        h2h_features['h2h_surface_balance'] = 1 - surface_balance if invert else surface_balance

        return h2h_features

    def _calculate_match_specific_features(self, p1_features: pd.Series, p2_features: pd.Series,
                                         surface: str, tournament_info: Dict = None) -> Dict:
        """Calculate match-specific features like ELO, fatigue, etc."""
        match_features = {}

        # ELO features - buscar correctamente en el PKL
        # Los ELOs están almacenados como 'elo_p1', 'elo_p2' en el PKL individual de cada jugador
        p1_elo = self._safe_get(p1_features, 'elo_p1', 'elo', 'elo_rating', default=1500)
        p2_elo = self._safe_get(p2_features, 'elo_p1', 'elo_p2', 'elo', 'elo_rating', default=1500)

        # LOG: Debug de ELOs encontrados
        logger.info(f"P1 ELO found: {p1_elo} (searching in {list(p1_features.index)[:10]})")
        logger.info(f"P2 ELO found: {p2_elo} (searching in {list(p2_features.index)[:10]})")

        match_features['elo_p1'] = p1_elo
        match_features['elo_p2'] = p2_elo

        elo_diff = p1_elo - p2_elo
        match_features['elo_diff'] = elo_diff
        match_features['elo_diff_clip'] = np.clip(elo_diff, -400, 400)
        match_features['elo_diff_sq'] = elo_diff ** 2
        match_features['elo_prob'] = 1 / (1 + 10**(-elo_diff/400))

        # Volatilidades ELO - buscar en el PKL
        p1_elo_vol = self._safe_get(p1_features, 'elo_vol5_p1', 'elo_vol5', default=50.0)
        p2_elo_vol = self._safe_get(p2_features, 'elo_vol5_p1', 'elo_vol5_p2', 'elo_vol5', default=50.0)

        match_features['elo_vol5_p1'] = p1_elo_vol
        match_features['elo_vol5_p2'] = p2_elo_vol
        match_features['elo_vol5_diff'] = p1_elo_vol - p2_elo_vol

        # Features de fatiga
        p1_matches_3d = self._safe_get(p1_features, 'p1_matches_last3d', 'matches_last3d', default=0)
        p2_matches_3d = self._safe_get(p2_features, 'p1_matches_last3d', 'matches_last3d', default=0)  # CORREGIDO: ambos buscan p1_

        match_features['fatigue_immediate'] = max(p1_matches_3d, p2_matches_3d)
        match_features['fatigue_short_term'] = (p1_matches_3d + p2_matches_3d) / 2

        # ELO fatigue
        match_features['elo_fatigue_immediate'] = p1_elo * (1 - 0.1 * p1_matches_3d)

        # Recovery factor
        p1_days = self._safe_get(p1_features, 'p1_days_since_last', 'days_since_last', default=7)
        p2_days = self._safe_get(p2_features, 'p1_days_since_last', 'days_since_last', default=7)  # CORREGIDO: ambos buscan p1_
        match_features['recovery_factor'] = min(p1_days, p2_days) / 7.0

        # Tournament info - estas features NO están en el modelo entrenado, remover
        if tournament_info:
            match_features['best_of'] = tournament_info.get('best_of', 3)
            match_features['draw_size'] = tournament_info.get('draw_size', 32)
            match_features['match_num'] = tournament_info.get('match_num', 1)
            match_features['minutes'] = tournament_info.get('minutes', 120)
        else:
            match_features['best_of'] = 3
            match_features['draw_size'] = 32
            match_features['match_num'] = 1
            match_features['minutes'] = 120

        # Features adicionales
        match_features['elo_consistency'] = abs(elo_diff) / 400  # Normalized consistency measure
        match_features['elo_rank_mismatch'] = 0.0  # Default
        match_features['elo_surface_interaction'] = elo_diff * 0.1  # Simple interaction
        match_features['elo_matches_interaction'] = elo_diff * (p1_matches_3d + p2_matches_3d)
        match_features['match_competitiveness'] = 1 / (1 + abs(elo_diff) / 100)
        match_features['matches_recent_weighted'] = (p1_matches_3d + p2_matches_3d) * 0.5

        return match_features

    def _ensure_all_columns(self, match_features: Dict, required_columns: List[str]) -> Dict:
        """Ensure all required columns are present with reasonable defaults."""
        complete_features = match_features.copy()

        for col in required_columns:
            if col not in complete_features:
                complete_features[col] = self._get_default_value(col)
                logger.warning(f"Missing feature {col}, using default value")

        return complete_features

    def _safe_get(self, features: pd.Series, *keys, default=None):
        """Safely get value from features using multiple possible keys."""
        for key in keys:
            if key in features.index and pd.notna(features[key]):
                return features[key]
        return default

    def _get_default_elo(self) -> float:
        """Get default ELO rating."""
        return 1500.0

    def _get_default_value(self, feature_name: str) -> float:
        """Get appropriate default value based on feature type."""
        if 'ratio' in feature_name.lower():
            return 1.0
        elif 'prob' in feature_name.lower() or 'balance' in feature_name.lower():
            return 0.5
        elif 'elo' in feature_name.lower():
            return 1500.0
        elif 'count' in feature_name.lower() or 'matches' in feature_name.lower():
            return 0
        elif 'diff' in feature_name.lower():
            return 0.0
        elif 'age' in feature_name.lower():
            return 25.0
        elif 'rank' in feature_name.lower() and 'points' not in feature_name.lower():
            return 100
        elif 'rank_points' in feature_name.lower():
            return 1000
        elif 'ht' in feature_name.lower():
            return 180
        elif 'days' in feature_name.lower():
            return 7
        else:
            return 0.0
