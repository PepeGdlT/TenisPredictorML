"""
Tennis Predictor API - Backend principal
Aplicación Flask para predicción de partidos de tenis
"""
import os
import pickle
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

# Importar servicios personalizados
from tennis_services import TennisPredictor, PlayerAnalyzer, TournamentAnalyzer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar Flask app
app = Flask(__name__)
CORS(app)  # Permitir CORS para el frontend

# Configuración
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'tennis-predictor-secret-key-2025')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Paths
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / 'outputs'
DATA_DIR = BASE_DIR / 'data'

# Cache global para modelos y datos
model_cache = {}
data_cache = {}

# Servicios globales
tennis_predictor = None
player_analyzer = None
tournament_analyzer = None

def load_models():
    """Cargar SOLO el modelo XGBoost optimizado - Sin fallbacks"""
    try:
        logger.info("Cargando modelo XGBoost optimizado...")

        # Archivos OBLIGATORIOS - si no existen, falla todo
        required_files = {
            'best_xgb_model.pkl': 'Modelo XGBoost entrenado',
            'imputer.pkl': 'Imputer para preprocessing',
            'training_states.pkl': 'Estados de entrenamiento',
            'players_list.pkl': 'Lista de jugadores',
            'model_summary.json': 'Resumen del modelo'
        }

        # VALIDACIÓN ESTRICTA: todos los archivos deben existir
        missing_files = []
        for filename, description in required_files.items():
            file_path = OUTPUTS_DIR / filename
            if not file_path.exists():
                missing_files.append(f"{filename} ({description})")

        if missing_files:
            error_msg = f"ARCHIVOS REQUERIDOS FALTANTES: {', '.join(missing_files)}"
            logger.error(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)

        # CARGA ESTRICTA: si cualquiera falla, todo falla
        logger.info("📁 Cargando modelo XGBoost...")
        with open(OUTPUTS_DIR / 'best_xgb_model.pkl', 'rb') as f:
            model_cache['xgb_model'] = pickle.load(f)

        logger.info("📁 Cargando imputer...")
        with open(OUTPUTS_DIR / 'imputer.pkl', 'rb') as f:
            model_cache['imputer'] = pickle.load(f)

        logger.info("📁 Cargando estados de entrenamiento...")
        with open(OUTPUTS_DIR / 'training_states.pkl', 'rb') as f:
            model_cache['training_states'] = pickle.load(f)

        logger.info("📁 Cargando lista de jugadores...")
        with open(OUTPUTS_DIR / 'players_list.pkl', 'rb') as f:
            model_cache['players_list'] = pickle.load(f)

        logger.info("📁 Cargando resumen del modelo...")
        with open(OUTPUTS_DIR / 'model_summary.json', 'r') as f:
            model_cache['model_summary'] = json.load(f)

        # Feature importance es OPCIONAL pero si existe debe cargarse bien
        feature_importance_path = OUTPUTS_DIR / 'feature_importance.csv'
        if feature_importance_path.exists():
            logger.info("📁 Cargando feature importance...")
            feature_importance = pd.read_csv(feature_importance_path)
            model_cache['feature_importance'] = feature_importance.to_dict('records')
        else:
            logger.warning("⚠️ feature_importance.csv no encontrado - se omitirá")
            model_cache['feature_importance'] = []

        # VALIDACIONES ADICIONALES: verificar que los modelos cargados sean válidos
        xgb_model = model_cache['xgb_model']
        imputer = model_cache['imputer']
        training_states = model_cache['training_states']
        players_list = model_cache['players_list']

        # Verificar que el modelo XGBoost tenga los métodos necesarios
        if not hasattr(xgb_model, 'predict') or not hasattr(xgb_model, 'predict_proba'):
            raise ValueError("El modelo XGBoost cargado no tiene métodos predict/predict_proba")

        # Verificar que el imputer sea válido
        if not hasattr(imputer, 'transform'):
            raise ValueError("El imputer cargado no tiene método transform")

        # Verificar que training_states tenga las keys necesarias
        required_keys = ['feature_columns', 'final_global_elos', 'final_surface_elos']
        for key in required_keys:
            if key not in training_states:
                raise ValueError(f"training_states no contiene la key requerida: {key}")

        # Verificar que la lista de jugadores no esté vacía
        if not players_list or len(players_list) < 100:
            raise ValueError(f"Lista de jugadores inválida (solo {len(players_list)} jugadores)")

        # TODO OK - LOG SUCCESS
        feature_columns = training_states.get('feature_columns', [])
        logger.info("✅ MODELO XGBoost CARGADO EXITOSAMENTE")
        logger.info(f"📊 Features del modelo: {len(feature_columns)}")
        logger.info(f"👥 Jugadores en BD: {len(players_list)}")
        logger.info(f"🎯 Accuracy del modelo: {model_cache['model_summary'].get('performance', {}).get('accuracy', 'N/A')}")

        return True

    except Exception as e:
        # FALLO CRÍTICO - limpiar cache y fallar
        model_cache.clear()
        logger.error(f"❌ FALLO CRÍTICO CARGANDO MODELO: {e}")
        raise e

def load_historical_data():
    """Cargar datos históricos para análisis de jugadores"""
    try:
        logger.info("Cargando datos históricos...")

        # Cargar datos procesados si existen
        train_path = DATA_DIR / 'processed' / 'train_full.csv'
        test_path = DATA_DIR / 'processed' / 'test_full.csv'

        if train_path.exists() and test_path.exists():
            # Cargar con parámetros para evitar warnings
            train_df = pd.read_csv(train_path, low_memory=False)
            test_df = pd.read_csv(test_path, low_memory=False)

            # Combinar y procesar
            full_df = pd.concat([train_df, test_df], ignore_index=True)

            # Convertir fechas de manera más robusta
            if 'tourney_date' in full_df.columns:
                full_df['tourney_date'] = pd.to_datetime(full_df['tourney_date'], errors='coerce')
                # Filtrar fechas válidas y ordenar
                full_df = full_df.dropna(subset=['tourney_date'])
                full_df = full_df.sort_values('tourney_date').reset_index(drop=True)

            data_cache['historical_data'] = full_df
            logger.info(f"✅ Datos históricos cargados: {len(full_df):,} partidos")
            return True
        else:
            logger.warning("⚠️ No se encontraron datos procesados")
            return False

    except Exception as e:
        logger.error(f"❌ Error cargando datos históricos: {e}")
        return False

def initialize_services():
    """Inicializar servicios de análisis"""
    global tennis_predictor, player_analyzer, tournament_analyzer

    try:
        tennis_predictor = TennisPredictor(model_cache, data_cache)
        player_analyzer = PlayerAnalyzer(data_cache)
        tournament_analyzer = TournamentAnalyzer(data_cache)
        logger.info("✅ Servicios de análisis inicializados")
        return True
    except Exception as e:
        logger.error(f"❌ Error inicializando servicios: {e}")
        return False

# Cargar todo al inicio
if not load_models():
    logger.error("No se pudieron cargar los modelos. La aplicación no funcionará correctamente.")

load_historical_data()
initialize_services()

@app.route('/')
def home():
    """Página principal con información de la API"""
    return jsonify({
        'message': 'Tennis Predictor API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'GET /api/health': 'Estado de la aplicación',
            'GET /api/players': 'Lista de jugadores',
            'GET /api/player/<name>': 'Información de un jugador',
            'POST /api/predict': 'Predicción de partido',
            'GET /api/model/info': 'Información del modelo',
            'GET /api/model/features': 'Feature importance'
        }
    })

@app.route('/api/health')
def health_check():
    """Endpoint de salud"""
    is_healthy = len(model_cache) > 0
    return jsonify({
        'status': 'healthy' if is_healthy else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(model_cache),
        'data_available': 'historical_data' in data_cache
    }), 200 if is_healthy else 503

@app.route('/api/players')
def get_players():
    """Obtener lista de jugadores"""
    try:
        players = model_cache.get('players_list', [])
        search = request.args.get('search', '').lower()
        limit = min(int(request.args.get('limit', 100)), 500)  # Max 500 jugadores

        if search:
            filtered_players = [p for p in players if search in p.lower()]
        else:
            filtered_players = players

        return jsonify({
            'players': filtered_players[:limit],
            'total': len(filtered_players),
            'showing': min(len(filtered_players), limit)
        })

    except Exception as e:
        logger.error(f"Error obteniendo jugadores: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/player/<player_name>')
def get_player_info(player_name):
    """Obtener información detallada de un jugador"""
    try:
        if not player_analyzer:
            return jsonify({'error': 'Servicio de análisis no disponible'}), 503

        player_stats = player_analyzer.get_player_stats(player_name)

        if not player_stats:
            return jsonify({'error': f'Jugador "{player_name}" no encontrado'}), 404

        return jsonify(player_stats)

    except Exception as e:
        logger.error(f"Error obteniendo info del jugador {player_name}: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """Predecir resultado de un partido"""
    try:
        data = request.get_json()

        if not data:
            raise BadRequest("No se proporcionaron datos")

        player1 = data.get('player1', '').strip()
        player2 = data.get('player2', '').strip()
        surface = data.get('surface', 'Hard')
        tournament_level = data.get('tournament_level', 'ATP500')

        if not player1 or not player2:
            raise BadRequest("Se requieren ambos jugadores")

        if player1.lower() == player2.lower():
            raise BadRequest("Los jugadores deben ser diferentes")

        # Verificar que el predictor esté disponible
        if not tennis_predictor:
            return jsonify({'error': 'Servicio de predicción no disponible'}), 503

        # Hacer predicción
        prediction = tennis_predictor.predict_match(player1, player2, surface, tournament_level)

        # Agregar timestamp y metadata
        prediction.update({
            'timestamp': datetime.now().isoformat(),
            'surface': surface,
            'tournament_level': tournament_level
        })

        return jsonify(prediction)

    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({'error': 'Error interno del servidor', "detail": traceback.format_exc()}), 500

@app.route('/api/tournaments')
def get_tournaments():
    """Obtener lista de torneos disponibles"""
    try:
        if not tournament_analyzer:
            return jsonify({'error': 'Servicio de torneos no disponible'}), 503

        tournaments = tournament_analyzer.get_tournament_stats()

        # Aplicar filtros si se proporcionan
        surface_filter = request.args.get('surface')
        level_filter = request.args.get('level')
        limit = min(int(request.args.get('limit', 50)), 100)

        if surface_filter:
            tournaments = [t for t in tournaments if t['surface'].lower() == surface_filter.lower()]

        if level_filter:
            tournaments = [t for t in tournaments if level_filter.lower() in t['level'].lower()]

        return jsonify({
            'tournaments': tournaments[:limit],
            'total': len(tournaments),
            'filters_applied': {
                'surface': surface_filter,
                'level': level_filter,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error obteniendo torneos: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/surfaces')
def get_surfaces():
    """Obtener tipos de superficie disponibles"""
    return jsonify({
        'surfaces': ['Hard', 'Clay', 'Grass', 'Carpet'],
        'descriptions': {
            'Hard': 'Pista dura - superficie más común en el tour',
            'Clay': 'Tierra batida - superficie lenta, favorece el juego defensivo',
            'Grass': 'Césped - superficie rápida, favorece el saque y volea',
            'Carpet': 'Alfombra - superficie rápida, ya no se usa en el tour profesional'
        }
    })

@app.route('/api/rankings/top')
def get_top_players():
    """Obtener jugadores más exitosos basado en datos históricos"""
    try:
        if 'historical_data' not in data_cache:
            return jsonify({'error': 'Datos históricos no disponibles'}), 503

        df = data_cache['historical_data']

        # Contar victorias por jugador
        player_wins = {}

        for winner_col in ['winner_name', 'player_1']:
            if winner_col in df.columns:
                wins = df[winner_col].value_counts()
                for player, count in wins.items():
                    if pd.notna(player) and player != 'Unknown':
                        player_wins[player] = player_wins.get(player, 0) + count
                break

        # Ordenar por victorias
        top_players = sorted(player_wins.items(), key=lambda x: x[1], reverse=True)[:50]

        result = []
        for i, (player, wins) in enumerate(top_players, 1):
            result.append({
                'rank': i,
                'player': player,
                'wins': int(wins),
                'estimated_total_matches': int(wins * 1.6)  # Estimación basada en win rate promedio
            })

        return jsonify({
            'top_players': result,
            'note': 'Ranking basado en datos históricos disponibles'
        })

    except Exception as e:
        logger.error(f"Error obteniendo top players: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/stats/overview')
def get_overview_stats():
    """Obtener estadísticas generales del dataset"""
    try:
        if 'historical_data' not in data_cache:
            return jsonify({'error': 'Datos históricos no disponibles'}), 503

        df = data_cache['historical_data']

        # Estadísticas básicas
        total_matches = len(df)

        # Fechas
        date_range = {}
        if 'tourney_date' in df.columns:
            dates = pd.to_datetime(df['tourney_date'], errors='coerce')
            date_range = {
                'earliest': dates.min().strftime('%Y-%m-%d') if dates.min() else 'Unknown',
                'latest': dates.max().strftime('%Y-%m-%d') if dates.max() else 'Unknown'
            }

        # Superficies
        surface_dist = {}
        if 'surface' in df.columns:
            surface_dist = df['surface'].value_counts().to_dict()

        # Torneos únicos
        unique_tournaments = 0
        if 'tourney_name' in df.columns:
            unique_tournaments = df['tourney_name'].nunique()

        # Jugadores únicos
        unique_players = len(model_cache.get('players_list', []))

        return jsonify({
            'total_matches': total_matches,
            'date_range': date_range,
            'unique_players': unique_players,
            'unique_tournaments': unique_tournaments,
            'surface_distribution': surface_dist,
            'model_info': model_cache.get('model_summary', {}),
            'last_updated': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas generales: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/model/info')
def get_model_info():
    """Información del modelo"""
    try:
        model_summary = model_cache.get('model_summary', {})
        return jsonify(model_summary)
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/api/model/features')
def get_feature_importance():
    """Feature importance del modelo"""
    try:
        features = model_cache.get('feature_importance', [])
        return jsonify({
            'features': features[:20],  # Top 20 features
            'total_features': len(features)
        })
    except Exception as e:
        logger.error(f"Error obteniendo features: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno: {error}")
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"🎾 Iniciando Tennis Predictor API en puerto {port}")
    logger.info(f"🔧 Modo debug: {debug}")

    app.run(host='0.0.0.0', port=port, debug=True)
