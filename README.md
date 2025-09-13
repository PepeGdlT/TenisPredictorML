# Tennis Match Predictor

Este proyecto utiliza machine learning avanzado y feature engineering para predecir partidos de tenis profesional con alta precisión. El pipeline incluye reducción de dimensionalidad con PCA, selección de variables con VIF, visualización avanzada y auditoría de fuga de información.

## Objetivos
- Desarrollar un modelo predictivo robusto para partidos de tenis
- Analizar la evolución histórica de jugadores top
- Identificar los factores más determinantes en la victoria
- Comparar diferentes algoritmos de machine learning
- Visualizar y auditar el proceso de modelado

## Datos
- 55+ años de historia del tenis profesional (1968-2024)
- Dataset ATP con miles de partidos, jugadores y estadísticas

## Pipeline
1. **Carga y limpieza de datos**: Unificación y limpieza de datos históricos
2. **Feature Engineering**: Cálculo de ELO global, ELO por superficie, ranking, H2H, y más de 20 features avanzadas
3. **Reducción de dimensionalidad**: PCA por grupos de features
4. **Selección de variables**: Eliminación de multicolinealidad con VIF
5. **Balanceo de dataset**: Duplicación de filas para targets 0/1
6. **Modelado**: XGBoost optimizado, Random Forest, Gradient Boosting, Ensemble
7. **Validación**: Split temporal (train: 1968-2022, test: 2023-2024), validación cruzada, auditoría de fuga
8. **Interpretabilidad y visualización**: Análisis de importancia de features, gráficos de evolución, matriz de confusión, análisis de errores

## Features principales
El dataset generado incluye más de 20 features avanzadas, agrupadas y procesadas para maximizar la capacidad predictiva y evitar fuga de información. Las principales categorías y ejemplos son:

- **ELO global y por superficie**: rating dinámico de cada jugador, ajustado tras cada partido, tanto a nivel general como específico de superficie (hard, clay, grass). Se calcula usando el histórico de partidos previos.
- **Ranking ATP**: diferencia y razón de ranking entre ambos jugadores, log-ratio y desajuste entre ranking y ELO.
- **Head-to-Head (H2H)**: historial de enfrentamientos directos global, por superficie y recientes (últimos 3 partidos).
- **Estadísticas históricas**: medias móviles previas de aces, dobles faltas, puntos de saque, break points, etc., tanto globales como por superficie.
- **Forma reciente y fatiga**: winrate en los últimos 5 y 10 partidos, winrate por superficie, rachas de victorias, días desde el último partido, partidos jugados en los últimos 7 y 14 días, y combinaciones escaladas de fatiga.
- **Volatilidad ELO**: variabilidad reciente en el rating ELO de cada jugador.
- **Contexto del partido**: competitividad estimada, tamaño del cuadro, ronda, tipo de torneo.
- **Reducción PCA**: para cada grupo de features (fatiga, ranking, forma reciente, H2H, ELO, etc.) se aplica PCA para obtener componentes principales que capturan la mayor varianza y reducen la dimensionalidad.
- **Selección VIF**: se eliminan automáticamente variables con multicolinealidad alta para robustez del modelo.

### Ejemplo de features generadas
- `elo_p1`, `elo_p2`, `elo_diff`, `surface_elo_p1`, `surface_elo_p2`, `surface_elo_diff`
- `rank_diff`, `rank_ratio`, `log_rank_ratio`, `elo_rank_mismatch`
- `h2h_count`, `h2h_balance`, `h2h_surface_count`, `h2h_surface_balance`, `h2h_recent3_balance`
- `form_last5_diff`, `form_last10_diff`, `surface_wr_all_diff`, `surface_wr_last5_diff`, `streak_wins_diff`
- `matches_last7d_diff`, `matches_last14d_diff`, `days_since_last_diff`, `matches_recent_weighted`, `elo_matches_interaction`
- `fatigue_pca1`, `fatigue_raw_pca1`, `elo_core_pca1`, `ranking_pca1`, `recent_form_pca1`, etc.

Cada feature se obtiene a partir del histórico de partidos previos, asegurando que solo se usan datos disponibles hasta el momento del partido (sin leakage futuro). El pipeline automatiza el cálculo, escalado, reducción y selección de todas las variables.

## Resultados recientes
- **Grid Search XGBoost**: mejores parámetros {'colsample_bytree': 0.5, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 200, 'subsample': 0.9}
- **Mejor CV score (log loss)**: -0.3880
- **Modelo optimizado**: Accuracy 0.8063, AUC 0.9012, LogLoss 0.3930, Brier 0.1280
- **Modelo calibrado**: LogLoss 0.3970, Brier 0.1293
- **Top 8 features explican el 80% de la importancia**
- **Top features**: fatigue_pca1, fatigue_raw_pca1, elo_core_pca1, elo_diff, surface_elo_diff, ranking_pca1, ranking_pca2, recent_form_pca2
- **No se detectó fuga de información significativa**

## Visualizaciones y auditoría
- Evolución de partidos por año y superficie
- Estadísticas y evolución de jugadores legendarios (Big 3)
- Matriz de correlación y distribución de features clave
- Comparación de algoritmos y ensemble
- Matriz de confusión y análisis de errores
- Auditoría de correlación, KS test, permutación de target y ablation

## Stack Tecnológico
- Python 3.11+
- pandas, scikit-learn, XGBoost, matplotlib, seaborn

## Recomendaciones
- Usar XGBoost optimizado para predicciones en producción
- Enfocar feature engineering en rankings, fatiga y diferencias de nivel
- Validar con datos de torneos futuros (2025+)

## Aplicaciones
- Sistema de predicción en tiempo real
- Análisis para casas de apuestas
- Comentarios deportivos automatizados
- Análisis estratégico para jugadores/entrenadores


