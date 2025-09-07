# 🎾 Tennis Predictor API

Backend completo para predicción de partidos de tenis usando Machine Learning.

## 🚀 Características

- **Predicción de partidos** usando modelo XGBoost entrenado con datos históricos ATP
- **Análisis de jugadores** con estadísticas detalladas y forma reciente
- **API REST completa** con documentación integrada
- **Análisis de torneos** y superficies
- **Rankings históricos** basados en datos reales
- **Modelo calibrado** para mejores probabilidades

## 📋 Endpoints Disponibles

### 🏠 General
- `GET /` - Información de la API
- `GET /api/health` - Estado de salud del servicio

### 👥 Jugadores
- `GET /api/players?search={nombre}&limit={número}` - Lista de jugadores
- `GET /api/player/{nombre}` - Información detallada de un jugador
- `GET /api/rankings/top` - Top jugadores por victorias

### 🎯 Predicciones
- `POST /api/predict` - Predecir resultado de un partido

```json
{
  "player1": "Novak Djokovic",
  "player2": "Rafael Nadal", 
  "surface": "Clay",
  "tournament_level": "ATP1000"
}
```

### 🏆 Torneos y Estadísticas
- `GET /api/tournaments?surface={superficie}&level={nivel}` - Lista de torneos
- `GET /api/surfaces` - Tipos de superficie disponibles
- `GET /api/stats/overview` - Estadísticas generales del dataset

### 🤖 Información del Modelo
- `GET /api/model/info` - Información del modelo ML
- `GET /api/model/features` - Features más importantes

## 🛠️ Instalación Local

### 1. Clonar y preparar entorno
```bash
git clone <tu-repo>
cd tennis-predictor
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements_backend.txt
```

### 3. Configurar variables de entorno
Crea un archivo `.env`:
```
FLASK_ENV=development
SECRET_KEY=tu-secret-key
PORT=5000
DEBUG=True
```

### 4. Ejecutar
```bash
python app.py
```

La API estará disponible en `http://localhost:5000`



## 📁 Estructura del Proyecto

```
tennis-predictor/
├── app.py                 # Aplicación Flask principal
├── tennis_services.py     # Lógica de negocio
├── requirements_backend.txt # Dependencias
├── Procfile              # Configuración Heroku
├── runtime.txt           # Versión Python
├── .env                  # Variables de entorno (local)
├── outputs/              # Modelos entrenados
│   ├── best_xgb_model.pkl
│   ├── calibrated_model.pkl
│   ├── training_states.pkl
│   └── players_list.pkl
└── data/                 # Datos procesados
    └── processed/
        ├── train_full.csv
        └── test_full.csv
```



## 📊 Modelo de Machine Learning

### Características del Modelo
- **Algoritmo**: XGBoost optimizado con GridSearch
- **Accuracy**: ~81% en datos de test
- **AUC**: ~0.90
- **Features**: 36 características engineered incluyendo:
  - ELO ratings (global y por superficie)
  - Estadísticas de fatiga y forma reciente
  - Head-to-head histórico
  - Rankings ATP
  - Características del torneo

### Calibración
El modelo incluye calibración de probabilidades usando CalibratedClassifierCV para obtener probabilidades más realistas.

## 🐛 Troubleshooting

### Error: "Modelos no cargados"
- Verificar que la carpeta `outputs/` contenga los archivos .pkl
- Ejecutar el notebook de entrenamiento si faltan modelos

### Error: "Datos históricos no disponibles"  
- Verificar que `data/processed/` contenga los archivos CSV
- Ejecutar el pipeline de procesamiento de datos

### Error de memoria en deployment
- Los archivos de modelo son grandes (~100MB)
- Considerar usar servicios con más RAM
- Optimizar carga de datos con lazy loading

## 🤝 Contribución

1. Fork del proyecto
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request



