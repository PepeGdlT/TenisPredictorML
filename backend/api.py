"""
FastAPI backend for tennis prediction API.
Main API module that exposes REST endpoints.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime

from .data_loader import DataLoader
from .feature_builder import FeatureBuilder
from .model import TennisPredictor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Crear la aplicación FastAPI
app = FastAPI(
    title="Tennis Prediction API",
    description="API REST para predicción de partidos de tenis usando ML",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancias globales
data_loader = None
feature_builder = None
predictor = None

# Modelos Pydantic para requests/responses
class PredictionRequest(BaseModel):
    player1: str = Field(..., description="Nombre del primer jugador")
    player2: str = Field(..., description="Nombre del segundo jugador")
    surface: str = Field(..., description="Superficie del partido (Hard, Clay, Grass)")
    tournament_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Información adicional del torneo (best_of, draw_size, etc.)"
    )

class PlayerFeatureInfo(BaseModel):
    name: str
    available: bool
    feature_count: Optional[int] = None
    sample_features: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    player1: str
    player2: str
    surface: str
    player_1_win_probability: float
    player_2_win_probability: float
    predicted_winner: int
    confidence: float
    model_type: str
    features_used: int
    top_features: Optional[Dict[str, Dict[str, float]]] = None
    prediction_timestamp: str
    validation_warnings: Optional[List[str]] = None

class SystemStatus(BaseModel):
    status: str
    data_loaded: bool
    model_loaded: bool
    model_info: Dict[str, Any]
    available_players: int
    last_updated: str

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    global data_loader, feature_builder, predictor

    try:
        logger.info("Initializing Tennis Prediction API...")

        # Inicializar componentes
        data_loader = DataLoader()
        feature_builder = FeatureBuilder(data_loader)
        predictor = TennisPredictor()

        # Cargar todos los datos
        logger.info("Loading data files...")
        data_loader.load_all_data()

        # Cargar modelo
        logger.info("Loading prediction model...")
        if not predictor.load_model():
            logger.error("Failed to load model")
            raise RuntimeError("Could not load prediction model")

        logger.info("API initialized successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Tennis Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health check."""
    try:
        player_count = 0
        if data_loader and data_loader.player_features is not None:
            if hasattr(data_loader.player_features, 'shape'):
                player_count = data_loader.player_features.shape[0]
            elif isinstance(data_loader.player_features, dict):
                player_count = len(data_loader.player_features)

        model_info = predictor.get_model_info() if predictor else {}

        return SystemStatus(
            status="healthy" if (data_loader and predictor) else "unhealthy",
            data_loaded=data_loader is not None and data_loader.player_features is not None,
            model_loaded=predictor is not None and predictor.model is not None,
            model_info=convert_numpy_types(model_info),
            available_players=player_count,
            last_updated=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    """
    Predict the outcome of a tennis match between two players.
    """
    try:
        logger.info(f"Prediction request: {request.player1} vs {request.player2} on {request.surface}")

        if not data_loader or not feature_builder or not predictor:
            raise HTTPException(status_code=500, detail="System not properly initialized")

        # Validar superficie
        valid_surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
        if request.surface not in valid_surfaces:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid surface. Must be one of: {valid_surfaces}"
            )

        # Construir features para el match
        try:
            match_features = feature_builder.build_match_features(
                request.player1,
                request.player2,
                request.surface,
                request.tournament_info
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Validar features
        validation = predictor.validate_features(match_features)
        if not validation['is_valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid features for prediction: {validation['errors']}"
            )

        # Realizar predicción
        prediction_result = predictor.predict_match(match_features)

        # Preparar respuesta
        response = PredictionResponse(
            player1=request.player1,
            player2=request.player2,
            surface=request.surface,
            player_1_win_probability=prediction_result['player_1_win_probability'],
            player_2_win_probability=prediction_result['player_2_win_probability'],
            predicted_winner=prediction_result['predicted_winner'],
            confidence=prediction_result['confidence'],
            model_type=prediction_result['model_type'],
            features_used=prediction_result['features_used'],
            top_features=prediction_result.get('top_features'),
            prediction_timestamp=datetime.now().isoformat(),
            validation_warnings=validation.get('warnings')
        )

        logger.info(f"Prediction completed successfully for {request.player1} vs {request.player2}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/players/{player_name}/features", response_model=PlayerFeatureInfo)
async def get_player_features(player_name: str):
    """Get available features for a specific player."""
    try:
        if not data_loader:
            raise HTTPException(status_code=500, detail="Data loader not initialized")

        player_features = data_loader.get_player_features(player_name)

        if player_features is None:
            return PlayerFeatureInfo(
                name=player_name,
                available=False
            )

        # Obtener muestra de features (sin valores sensibles)
        sample_features = {}
        if isinstance(player_features, pd.Series):
            feature_names = list(player_features.index)[:10]  # Primeros 10
            for feature in feature_names:
                if not feature.endswith('_id'):  # Excluir IDs
                    value = player_features[feature]
                    if pd.notna(value):
                        sample_features[feature] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
                    else:
                        sample_features[feature] = None
        else:
            # Si no es Series, convertir a dict simple
            sample_features = {"error": "Could not extract features"}

        return PlayerFeatureInfo(
            name=player_name,
            available=True,
            feature_count=len(player_features) if player_features is not None else 0,
            sample_features=convert_numpy_types(sample_features)
        )

    except Exception as e:
        logger.error(f"Error getting player features: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving player features: {str(e)}")

@app.get("/players", response_model=List[str])
async def list_available_players(limit: int = 100):
    """List available players in the system."""
    try:
        if not data_loader or data_loader.player_features is None:
            raise HTTPException(status_code=500, detail="Player data not loaded")

        if hasattr(data_loader.player_features, 'index'):
            players = list(data_loader.player_features.index)[:limit]
        elif isinstance(data_loader.player_features, dict):
            players = list(data_loader.player_features.keys())[:limit]
        else:
            players = []

        return players

    except Exception as e:
        logger.error(f"Error listing players: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing players: {str(e)}")

@app.get("/surfaces", response_model=List[str])
async def list_supported_surfaces():
    """List supported tennis surfaces."""
    return ['Hard', 'Clay', 'Grass', 'Carpet']

@app.post("/reload-data")
async def reload_data(background_tasks: BackgroundTasks):
    """Reload all data files (for updating the system)."""
    try:
        if not data_loader:
            raise HTTPException(status_code=500, detail="Data loader not initialized")

        def reload_task():
            try:
                logger.info("Reloading all data files...")
                data_loader.load_all_data()
                logger.info("Data reload completed")
            except Exception as e:
                logger.error(f"Error during data reload: {e}")

        background_tasks.add_task(reload_task)
        return {"message": "Data reload initiated in background"}

    except Exception as e:
        logger.error(f"Error initiating data reload: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading data: {str(e)}")

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get detailed information about the loaded model."""
    try:
        if not predictor:
            raise HTTPException(status_code=500, detail="Predictor not initialized")

        model_info = predictor.get_model_info()
        return convert_numpy_types(model_info)

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# Endpoint de ejemplo para testing
@app.get("/test-prediction")
async def test_prediction():
    """Test endpoint with sample players (for development/testing)."""
    try:
        # Usar jugadores de ejemplo (ajustar según tu dataset)
        sample_request = PredictionRequest(
            player1="Novak Djokovic",
            player2="Rafael Nadal",
            surface="Clay"
        )

        return await predict_match(sample_request)

    except Exception as e:
        return {"error": f"Test failed: {str(e)}", "suggestion": "Check available players with /players endpoint"}

@app.get("/debug/player/{player_name}")
async def debug_player_features(player_name: str):
    """Debug endpoint to show all features for a specific player."""
    try:
        if not data_loader:
            raise HTTPException(status_code=500, detail="Data loader not initialized")

        player_features = data_loader.get_player_features(player_name)

        if player_features is None:
            return {"error": f"Player {player_name} not found"}

        # Convertir todas las features a dict para debug
        all_features = {}
        for idx, value in player_features.items():
            try:
                all_features[str(idx)] = float(value) if pd.notna(value) and isinstance(value, (int, float, np.number)) else str(value)
            except:
                all_features[str(idx)] = str(value)

        return {
            "player": player_name,
            "total_features": len(all_features),
            "features": convert_numpy_types(all_features),
            "sample_elo_features": {
                k: v for k, v in all_features.items()
                if 'elo' in k.lower()
            },
            "sample_rank_features": {
                k: v for k, v in all_features.items()
                if 'rank' in k.lower()
            },
            "sample_matches_features": {
                k: v for k, v in all_features.items()
                if 'matches' in k.lower()
            }
        }

    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.get("/debug/match/{player1}/{player2}/{surface}")
@app.get("/debug_match_features")
async def debug_match_features(player1: str, player2: str, surface: str):
    """Debug endpoint to show the complete feature construction process."""
    try:
        if not data_loader or not feature_builder:
            raise HTTPException(status_code=500, detail="System not initialized")

        # Obtener features base de ambos jugadores (filtradas SOLO a las del modelo)
        required_features = data_loader.get_required_feature_columns()
        p1_features_full = data_loader.get_player_features(player1)
        p2_features_full = data_loader.get_player_features(player2)
        if p1_features_full is None or p2_features_full is None:
            return {"error": f"Features not found for one or both players"}
        p1_features = {k: v for k, v in p1_features_full.items() if k in required_features}
        p2_features = {k: v for k, v in p2_features_full.items() if k in required_features}

        # Construir features del match (ya filtradas, SOLO las del modelo)
        match_features = feature_builder.build_match_features(player1, player2, surface)
        filtered_features = dict(match_features)  # Esto ya son solo las 49 features finales

        # Obtener H2H y Surface ELO info
        h2h_stats = data_loader.get_h2h_stats(player1, player2, surface)
        p1_surface_elo = data_loader.get_surface_elo(player1, surface)
        p2_surface_elo = data_loader.get_surface_elo(player2, surface)

        # DEBUG: Print the keys and count of filtered_features before returning
        print(f"[DEBUG] API returns {len(filtered_features)} features: {list(filtered_features.keys())}")
        return {
            "match": f"{player1} vs {player2} on {surface}",
            "player1_feature_count": len(p1_features),
            "player2_feature_count": len(p2_features),
            "match_features_count": len(filtered_features),
            "all_match_features_count": len(filtered_features),
            "all_match_features": convert_numpy_types(filtered_features),
            "h2h_stats": convert_numpy_types(h2h_stats),
            "surface_elos": {
                "player1": p1_surface_elo,
                "player2": p2_surface_elo
            },
            "critical_features": {
                k: convert_numpy_types(filtered_features.get(k, "MISSING"))
                for k in ['elo_p1', 'elo_p2', 'elo_diff', 'surface_elo_p1', 'surface_elo_p2',
                         'surface_elo_diff', 'rank_diff', 'matches_last7d_diff']
            }
        }

    except Exception as e:
        logger.error(f"Error in debug match endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.get("/debug/pure-player/{player_name}")
async def get_pure_player_features(player_name: str):
    """Get pure individual features for a player without opponent contamination."""
    try:
        if not data_loader:
            raise HTTPException(status_code=500, detail="Data loader not initialized")

        player_features = data_loader.get_player_features(player_name)

        if player_features is None:
            return {"error": f"Player {player_name} not found"}

        # Extraer solo features del JUGADOR (p1_*), no del oponente (p2_*)
        pure_features = {}
        opponent_features = {}

        for feature_name, value in player_features.items():
            if feature_name.startswith('p1_'):
                # Estas son las features del jugador actual
                pure_features[feature_name] = convert_numpy_types(value)
            elif feature_name.startswith('p2_'):
                # Estas son las features del oponente (contaminación)
                opponent_features[feature_name] = convert_numpy_types(value)
            elif not feature_name.startswith(('player_', 'target', 'elo_', 'surface_', 'h2h_')):
                # Features generales del partido
                pure_features[feature_name] = convert_numpy_types(value)

        return {
            "player": player_name,
            "pure_features_count": len(pure_features),
            "contaminated_features_count": len(opponent_features),
            "pure_features": pure_features,
            "sample_opponent_contamination": dict(list(opponent_features.items())[:5]),
            "has_contamination": len(opponent_features) > 0,
            "recommendation": "Use only p1_* features for individual player data"
        }

    except Exception as e:
        logger.error(f"Error in pure player endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
