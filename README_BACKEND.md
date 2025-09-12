# Tennis Prediction API Backend

Backend completo con API REST en FastAPI para predicción de partidos de tenis usando Machine Learning.

## 🏗️ Arquitectura

El backend está organizado en módulos especializados:

```
backend/
├── __init__.py           # Inicialización del paquete
├── data_loader.py        # Carga de datos (PKL, JSONs)
├── feature_builder.py    # Construcción dinámica de features
├── model.py             # Carga y predicción del modelo ML
├── api.py               # Endpoints REST con FastAPI
└── config.py            # Configuración del sistema
```

## 🚀 Funcionalidades

### ✅ Carga de Datos Base
- **Player Features**: Carga automática del PKL `player_last_features.pkl`
- **Surface ELO**: Integración del JSON `surface_elos_by_player.json`
- **Head-to-Head**: Datos históricos desde `h2h_all.json`

### ✅ Construcción Dinámica de Features
- Extrae features individuales de cada jugador (prefijos p1_, p2_)
- Calcula diferencias y ratios entre jugadores automáticamente
- Integra ELO de superficie específica para el partido
- Computa estadísticas H2H (general y por superficie)
- Maneja valores faltantes con defaults inteligentes

### ✅ Integración de Modelo ML
- Carga automática de modelos entrenados (ensemble, calibrated, best)
- Soporte para XGBoost y otros modelos de scikit-learn
- Manejo de imputación de features si está disponible
- Validación de features antes de predicción

### ✅ API REST Completa
- **POST /predict**: Predicción de partidos
- **GET /status**: Estado del sistema
- **GET /players**: Lista de jugadores disponibles
- **GET /players/{name}/features**: Features de un jugador
- **GET /model/info**: Información del modelo cargado

## 📋 Columnas de Features Soportadas

El sistema maneja automáticamente todas las 129 columnas especificadas:

```python
# Features individuales por jugador (p1_, p2_)
'p1_age', 'p1_rank', 'p1_rank_points', 'p1_ht', 'p1_streak_wins',
'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_ace', 'p1_df', ...

# Features calculadas dinámicamente
'elo_diff', 'rank_diff', 'form_last5_diff', 'surface_elo_diff',
'h2h_balance', 'h2h_surface_balance', ...

# Ratios y transformaciones
'1stIn_ratio', '1stWon_ratio', 'ace_ratio', 'svpt_ratio', ...
```

## 🔧 Instalación y Ejecución

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Verificar archivos de datos
Asegúrate de tener en `outputs/`:
- `player_last_features.pkl`
- `surface_elos_by_player.json` 
- `h2h_all.json`
- `best_xgb_model.pkl` (o ensemble_model.pkl)

### 3. Ejecutar el servidor
```bash
python main.py
```

El servidor estará disponible en `http://localhost:8000`

### 4. Probar la API
```bash
python test_api.py
```

## 📚 Ejemplos de Uso

### Predicción básica
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "player1": "Novak Djokovic",
    "player2": "Rafael Nadal", 
    "surface": "Clay"
})

result = response.json()
print(f"Probabilidad P1: {result['player_1_win_probability']:.1%}")
print(f"Probabilidad P2: {result['player_2_win_probability']:.1%}")
```

### Con información de torneo
```python
response = requests.post("http://localhost:8000/predict", json={
    "player1": "Carlos Alcaraz",
    "player2": "Jannik Sinner",
    "surface": "Hard",
    "tournament_info": {
        "best_of": 5,
        "draw_size": 128,
        "match_num": 1
    }
})
```

### Consultar jugadores disponibles
```python
players = requests.get("http://localhost:8000/players?limit=50").json()
print(f"Jugadores disponibles: {len(players)}")
```

## 🛡️ Robustez del Sistema

### Manejo de Datos Faltantes
- **Jugador no encontrado**: Excepción HTTP 404 con mensaje claro
- **ELO de superficie faltante**: Usa ELO general o valor default (1500)
- **H2H sin datos**: Aplica balance 50-50 como default
- **Features faltantes**: Valores por defecto inteligentes según tipo

### Logging Completo
```python
# Ejemplos de logs generados
INFO - Loading player features: DataFrame shape: (2500, 150)
INFO - Built 129 features for match
WARNING - No surface ELO found for Player X on Clay, using default
INFO - Prediction completed: P1=0.654, P2=0.346
```

### Validación de Features
- Verifica features críticas (elo_p1, elo_p2, rank_diff)
- Detecta valores extremos (>1e6)
- Alerta sobre features faltantes
- Validación antes de cada predicción

## 🔍 Endpoints Detallados

### POST /predict
**Request:**
```json
{
    "player1": "string",
    "player2": "string", 
    "surface": "Hard|Clay|Grass|Carpet",
    "tournament_info": {
        "best_of": 3,
        "draw_size": 32,
        "match_num": 1
    }
}
```

**Response:**
```json
{
    "player1": "Novak Djokovic",
    "player2": "Rafael Nadal",
    "surface": "Clay",
    "player_1_win_probability": 0.35,
    "player_2_win_probability": 0.65,
    "predicted_winner": 2,
    "confidence": 0.30,
    "model_type": "ensemble_model",
    "features_used": 129,
    "top_features": {
        "surface_elo_diff": {"importance": 0.15, "value": -120},
        "h2h_surface_balance": {"importance": 0.12, "value": 0.3}
    },
    "prediction_timestamp": "2025-01-10T15:30:00",
    "validation_warnings": []
}
```

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
LOG_LEVEL=INFO
```

### Archivos de Configuración
Modifica `backend/config.py` para ajustar:
- Rutas de archivos de datos
- Valores por defecto para features
- Configuración de logging
- Parámetros de validación

## 🧪 Testing y Desarrollo

### Script de Pruebas Automatizadas
```bash
python test_api.py
```

Ejecuta un test completo que verifica:
- ✅ Conexión al servidor
- ✅ Estado del sistema
- ✅ Carga de datos
- ✅ Funcionalidad del modelo
- ✅ Predicciones en todas las superficies

### Documentación Interactiva
Una vez ejecutando el servidor:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## ⚡ Rendimiento

- **Tiempo de carga inicial**: ~5-10 segundos
- **Tiempo por predicción**: ~50-200ms
- **Memory footprint**: ~500MB-1GB (según tamaño de datos)
- **Concurrencia**: Soporta múltiples requests simultáneos

## 🔄 Mantenimiento

### Actualizar Datos
```bash
curl -X POST http://localhost:8000/reload-data
```

### Monitoreo
```bash
curl http://localhost:8000/status
```

### Logs
Los logs se generan automáticamente con información detallada de cada operación.
