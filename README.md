# Tennis Match Predictor

Este proyecto utiliza machine learning avanzado y feature engineering para predecir partidos de tenis profesional con alta precisión.

## Objetivos
- Desarrollar un modelo predictivo de alta precisión para partidos de tenis
- Analizar la evolución histórica de jugadores top
- Identificar los factores más determinantes en la victoria
- Comparar diferentes algoritmos de machine learning

## Datos
- 55+ años de historia del tenis profesional (1968-2024)
- Dataset ATP con miles de partidos, jugadores y estadísticas

## Pipeline
1. **Carga y limpieza de datos**: Unificación y limpieza de datos históricos
2. **Feature Engineering**: Cálculo de ELO global, ELO por superficie, ranking, H2H, y más de 20 features avanzadas
3. **Balanceo de dataset**: Duplicación de filas para targets 0/1
4. **Modelado**: XGBoost optimizado, Random Forest, Gradient Boosting, Ensemble
5. **Validación**: Split temporal (train: 1968-2022, test: 2023-2024)
6. **Interpretabilidad**: Análisis de importancia de features y errores

## Resultados
- **Accuracy XGBoost optimizado**: ~99% en test
- **Top features**: elo_advantage, tier_diff, rank_advantage, rank_diff, surface_elo_advantage
- **El H2H tiene impacto mínimo en predicciones generales**
- **El ranking ATP y las diferencias de nivel son más informativos que el ELO calculado**
- **El modelo es altamente preciso para datos históricos y generalizable a nuevos jugadores**

## Visualizaciones
- Evolución de partidos por año y superficie
- Estadísticas y evolución de jugadores legendarios (Big 3)
- Matriz de correlación y distribución de features clave
- Comparación de algoritmos y ensemble
- Matriz de confusión y análisis de errores

## Stack Tecnológico
- Python 3.11+
- pandas, scikit-learn, XGBoost, matplotlib, seaborn

## Recomendaciones
- Usar XGBoost optimizado para predicciones en producción
- Enfocar feature engineering en rankings y diferencias de nivel
- Validar con datos de torneos futuros (2025+)

## Aplicaciones
- Sistema de predicción en tiempo real
- Análisis para casas de apuestas
- Comentarios deportivos automatizados
- Análisis estratégico para jugadores/entrenadores

---
*Desarrollado para el análisis predictivo del tenis profesional*

