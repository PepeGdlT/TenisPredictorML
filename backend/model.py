"""
Model module for tennis prediction backend.
Handles loading of trained models and making predictions.
"""
import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class TennisPredictor:
    def __init__(self, base_path: str = "outputs"):
        self.base_path = Path(base_path)
        self.model = None
        self.feature_imputer = None
        self.model_type = None
        self.feature_names = None

    def load_model(self, model_filename: str = None) -> bool:
        """Load the trained model and any associated preprocessors."""
        try:
            # Intentar cargar diferentes tipos de modelos en orden de preferencia
            model_files = [
                "calibrated_xgb_model.pkl"
                , "best_xgb_model.pkl"
                , "ensemble_model.pkl"

            ]

            if model_filename:
                model_files.insert(0, model_filename)

            model_loaded = False
            for filename in model_files:
                model_path = self.base_path / filename
                if model_path.exists():
                    try:
                        with open(model_path, 'rb') as f:
                            self.model = pickle.load(f)

                        self.model_type = filename.replace('.pkl', '')
                        logger.info(f"Successfully loaded model: {filename}")
                        model_loaded = True
                        break

                    except Exception as e:
                        logger.warning(f"Failed to load {filename}: {e}")
                        continue

            if not model_loaded:
                raise FileNotFoundError("No valid model file found")

            # Intentar cargar el imputer de features si existe
            try:
                imputer_path = self.base_path / "feature_imputer.pkl"
                if imputer_path.exists():
                    with open(imputer_path, 'rb') as f:
                        self.feature_imputer = pickle.load(f)
                    logger.info("Loaded feature imputer")
            except Exception as e:
                logger.warning(f"Could not load feature imputer: {e}")
                self.feature_imputer = None

            # Obtener nombres de features si están disponibles
            try:
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = list(self.model.feature_names_in_)
                elif hasattr(self.model, 'get_booster') and hasattr(self.model.get_booster(), 'feature_names'):
                    self.feature_names = self.model.get_booster().feature_names
            except:
                logger.warning("Could not extract feature names from model")

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def predict_match(self, match_features: pd.Series) -> Dict[str, Any]:
        """
        Make prediction for a tennis match.

        Args:
            match_features: Series with all required features for prediction

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # LOG: Mostrar features originales
            logger.info(f"Original match_features shape: {len(match_features)}")
            logger.info(f"Original features (first 10): {dict(list(match_features.head(10).items()))}")

            # Preparar features para el modelo
            X = self._prepare_features_for_model(match_features)

            # LOG: Mostrar features preparadas para el modelo
            logger.info(f"Prepared features shape: {X.shape}")
            logger.info(f"Prepared feature columns: {list(X.columns)}")
            logger.info(f"Sample prepared features (first 10): {dict(X.iloc[0].head(10))}")

            # Realizar predicción
            if hasattr(self.model, 'predict_proba'):
                # Modelo con probabilidades
                probabilities = self.model.predict_proba(X)[0]
                prediction = self.model.predict(X)[0]

                # LOG: Mostrar probabilidades raw
                logger.info(f"Raw probabilities: {probabilities}")
                logger.info(f"Raw prediction: {prediction}")

                # Asumir que clase 1 es victoria del jugador 1
                prob_p1_wins = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                prob_p2_wins = 1 - prob_p1_wins

            elif hasattr(self.model, 'predict'):
                # Modelo que devuelve solo predicción
                prediction = self.model.predict(X)[0]
                logger.info(f"Raw prediction: {prediction}")

                # Si es un valor continuo entre 0 y 1, usarlo como probabilidad
                if 0 <= prediction <= 1:
                    prob_p1_wins = prediction
                    prob_p2_wins = 1 - prediction
                else:
                    # Si es clasificación binaria (0/1)
                    prob_p1_wins = prediction
                    prob_p2_wins = 1 - prediction
            else:
                raise ValueError("Model does not support prediction")

            # Preparar resultado
            result = {
                'player_1_win_probability': float(prob_p1_wins),
                'player_2_win_probability': float(prob_p2_wins),
                'predicted_winner': 1 if prob_p1_wins > 0.5 else 2,
                'confidence': float(abs(prob_p1_wins - 0.5) * 2),  # Confidence = distance from 50%
                'model_type': self.model_type,
                'features_used': len(X.columns) if isinstance(X, pd.DataFrame) else len(X[0]),
                # AGREGAR: Información de debug
                'debug_info': {
                    'original_features_count': len(match_features),
                    'prepared_features_count': X.shape[1],
                    'feature_names': list(X.columns),
                    'sample_features': dict(X.iloc[0].head(20))  # Primeras 20 features para debug
                }
            }

            # Agregar información adicional si está disponible
            top_features = None

            # Intentar obtener feature importances del modelo
            try:
                importances = None

                # Para modelo calibrado, intentar acceder al modelo base
                if hasattr(self.model, 'base_estimator'):
                    # CalibratedClassifierCV tiene base_estimator
                    base_model = self.model.base_estimator
                    if hasattr(base_model, 'feature_importances_'):
                        importances = base_model.feature_importances_
                        logger.info(f"Got feature importances from calibrated model base estimator")
                elif hasattr(self.model, 'estimator'):
                    # Otro tipo de wrapper
                    base_model = self.model.estimator
                    if hasattr(base_model, 'feature_importances_'):
                        importances = base_model.feature_importances_
                        logger.info(f"Got feature importances from wrapped estimator")
                elif hasattr(self.model, 'feature_importances_'):
                    # Modelo XGBoost directo
                    importances = self.model.feature_importances_
                    logger.info(f"Got feature importances from direct model")
                else:
                    # Intentar acceder a través de calibrated_classifiers_
                    if hasattr(self.model, 'calibrated_classifiers_') and len(self.model.calibrated_classifiers_) > 0:
                        # Tomar el primer calibrated classifier
                        cal_clf = self.model.calibrated_classifiers_[0]
                        if hasattr(cal_clf, 'base_estimator') and hasattr(cal_clf.base_estimator, 'feature_importances_'):
                            importances = cal_clf.base_estimator.feature_importances_
                            logger.info(f"Got feature importances from first calibrated classifier")

                if importances is not None:
                    logger.info(f"Feature importances shape: {len(importances)}")

                    if len(importances) == len(X.columns):
                        top_features = self._get_top_features_from_prepared(X.iloc[0], importances)
                        logger.info(f"Successfully extracted top features from model")
                    else:
                        logger.warning(f"Feature importances mismatch: {len(importances)} vs {len(X.columns)}")
                        top_features = None
                else:
                    logger.warning(f"Could not extract feature importances from model type: {type(self.model)}")
                    top_features = None

            except Exception as e:
                logger.warning(f"Error extracting feature importances: {e}")
                top_features = None

            # Agregar top_features al resultado
            result['top_features'] = top_features

            logger.info(f"Prediction completed: P1={prob_p1_wins:.3f}, P2={prob_p2_wins:.3f}")
            return result

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def _get_top_features_from_prepared(self, prepared_features: pd.Series, importances: np.ndarray,
                         top_n: int = 10) -> Dict[str, float]:
        """Get top N most important features from prepared features."""
        try:
            feature_names = list(prepared_features.index)

            if len(feature_names) != len(importances):
                logger.warning(f"Feature names length {len(feature_names)} != importances length {len(importances)}")
                return {}

            # Crear pares feature-importance y ordenar
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            # Tomar top N con sus valores
            top_features = {}
            for feature, importance in feature_importance_pairs[:top_n]:
                top_features[feature] = {
                    'importance': float(importance),
                    'value': float(prepared_features.get(feature, 0))
                }

            return top_features

        except Exception as e:
            logger.warning(f"Could not extract top features: {e}")
            return {}

    def _prepare_features_for_model(self, match_features: pd.Series) -> pd.DataFrame:
        """Prepare features for model input."""
        # Convertir Series a DataFrame
        X = pd.DataFrame([match_features])

        # LOG: Verificar si hay columna target antes de remover
        if 'target' in X.columns:
            logger.warning(f"⚠️  TARGET COLUMN FOUND in features! Value: {X['target'].iloc[0]}")
            logger.warning("This could cause data leakage and inconsistent predictions!")

        # Remover columnas no numéricas que no necesita el modelo
        non_numeric_cols = ['player_1_id', 'player_2_id', 'target']
        for col in non_numeric_cols:
            if col in X.columns:
                logger.info(f"Removing non-numeric column: {col}")
                X = X.drop(col, axis=1)

        # LOG: Verificar features después de limpieza
        logger.info(f"Features after cleanup: {list(X.columns)[:10]}...")

        # Si tenemos el imputer, usar solo las features que espera el modelo
        if self.feature_imputer is not None:
            try:
                # Obtener las features que espera el modelo del imputer
                expected_features = self.feature_imputer.feature_names_in_

                # Filtrar solo las features que el modelo espera
                available_expected = [col for col in expected_features if col in X.columns]
                missing_expected = [col for col in expected_features if col not in X.columns]

                if missing_expected:
                    logger.warning(f"Missing expected features: {missing_expected[:5]}... (showing first 5)")
                    # Agregar features faltantes con valor 0
                    for col in missing_expected:
                        X[col] = 0.0

                # Mantener solo las features esperadas
                X = X[expected_features]

                # Aplicar el imputer
                X_imputed = self.feature_imputer.transform(X)
                if isinstance(X_imputed, np.ndarray):
                    X = pd.DataFrame(X_imputed, columns=expected_features)
                else:
                    X = X_imputed

                logger.info(f"Successfully used feature imputer with {len(expected_features)} features")

            except Exception as e:
                logger.warning(f"Could not use feature imputer: {e}")
                # Fallback: usar solo las features que conocemos que funcionan
                X = self._use_core_features_only(X)
                X = X.fillna(0)
        else:
            # Sin imputer, usar las features principales que sabemos que funcionan
            X = self._use_core_features_only(X)
            X = X.fillna(0)

        # Asegurar que todos los valores sean numéricos
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        return X

    def _use_core_features_only(self, X: pd.DataFrame) -> pd.DataFrame:
        """Use only the core features that we know work with the model."""
        # Lista de features principales que esperamos que funcionen
        core_features = [
            # ELO features
            'elo_vol5_diff', 'surface_elo_vol5_diff',
            # H2H features
            'h2h_count', 'h2h_balance', 'h2h_surface_count', 'h2h_surface_balance', 'h2h_recent3_balance',
            # ELO probability and differences
            'elo_diff_clip', 'surface_elo_diff_clip', 'elo_prob', 'surface_elo_prob',
            'elo_diff_sq', 'surface_elo_diff_sq', 'elo_surface_interaction', 'elo_consistency',
            # Ranking features
            'rank_diff', 'rank_ratio', 'log_rank_ratio', 'elo_rank_mismatch',
            # Match features
            'match_competitiveness',
            # Form features
            'form_last5_diff', 'form_last10_diff', 'surface_wr_all_diff', 'surface_wr_last5_diff',
            # Activity features
            'streak_wins_diff', 'days_since_last_diff', 'matches_last7d_diff', 'matches_last14d_diff',
            'days_since_last_diff_scaled', 'matches_recent_weighted',
            # Interaction features
            'elo_matches_interaction',
            # Individual player stats (means)
            'p1_ace_mean', 'p1_df_mean', 'p1_svpt_mean', 'p1_1stIn_mean', 'p1_1stWon_mean', 'p1_2ndWon_mean',
            'p1_bpSaved_mean', 'p1_bpFaced_mean', 'p1_ace_surface_mean',
            'p2_ace_mean', 'p2_df_mean', 'p2_svpt_mean', 'p2_1stIn_mean', 'p2_1stWon_mean', 'p2_2ndWon_mean',
            'p2_bpSaved_mean', 'p2_bpFaced_mean', 'p2_ace_surface_mean'
        ]

        # Mantener solo las features que están disponibles y son parte del core
        available_core = [col for col in core_features if col in X.columns]
        missing_core = [col for col in core_features if col not in X.columns]

        if missing_core:
            logger.warning(f"Missing core features: {missing_core[:5]}... (showing first 5)")
            # Agregar features faltantes con valor por defecto
            for col in missing_core:
                X[col] = 0.0

        # Retornar solo las features del core
        return X[core_features]

    def _get_top_features(self, match_features: pd.Series, importances: np.ndarray,
                         top_n: int = 10) -> Dict[str, float]:
        """Get top N most important features for this prediction."""
        try:
            # Obtener nombres de features (excluyendo IDs)
            feature_names = [col for col in match_features.index
                           if col not in ['player_1_id', 'player_2_id']]

            if len(feature_names) != len(importances):
                return {}

            # Crear pares feature-importance y ordenar
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            # Tomar top N con sus valores
            top_features = {}
            for feature, importance in feature_importance_pairs[:top_n]:
                top_features[feature] = {
                    'importance': float(importance),
                    'value': float(match_features.get(feature, 0))
                }

            return top_features

        except Exception as e:
            logger.warning(f"Could not extract top features: {e}")
            return {}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {'status': 'no_model_loaded'}

        info = {
            'model_type': self.model_type,
            'model_class': type(self.model).__name__,
            'has_feature_imputer': self.feature_imputer is not None
        }

        # Información adicional según el tipo de modelo
        try:
            if hasattr(self.model, 'n_features_in_'):
                info['n_features'] = self.model.n_features_in_

            if hasattr(self.model, 'classes_'):
                info['classes'] = list(self.model.classes_)

            if self.feature_names:
                info['feature_count'] = len(self.feature_names)
                info['sample_features'] = self.feature_names[:10]  # Primeras 10 features

        except Exception as e:
            logger.warning(f"Could not extract additional model info: {e}")

        return info

    def validate_features(self, match_features: pd.Series) -> Dict[str, Any]:
        """Validate that features are suitable for prediction."""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }

        # Verificar que tenemos las features mínimas requeridas
        numeric_features = [col for col in match_features.index
                          if col not in ['player_1_id', 'player_2_id']]

        if len(numeric_features) < 50:  # Mínimo esperado
            validation_result['warnings'].append(
                f"Few features available: {len(numeric_features)}, expected ~100+"
            )

        # Verificar valores extremos o faltantes
        for col in numeric_features:
            value = match_features.get(col)

            if pd.isna(value):
                validation_result['warnings'].append(f"Missing value for {col}")
            elif isinstance(value, (int, float)):
                if abs(value) > 1e6:  # Valor extremadamente grande
                    validation_result['warnings'].append(f"Extreme value for {col}: {value}")

        """
        # Verificar features críticas
        critical_features = ['elo_p1', 'elo_p2', 'rank_diff', 'surface_elo_diff']
        missing_critical = [f for f in critical_features if f not in match_features.index]

        if missing_critical:
            validation_result['errors'].append(f"Missing critical features: {missing_critical}")
            validation_result['is_valid'] = False
        """
        return validation_result
