from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def train_model(X, y):
    """Entrena un XGBoost altamente optimizado para máximo accuracy."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔥 MODELO ALTAMENTE OPTIMIZADO PARA MAYOR ACCURACY
    model = XGBClassifier(
        # Parámetros principales optimizados
        n_estimators=300,           # Más árboles para mejor aprendizaje
        max_depth=8,                # Más profundidad para capturar patrones complejos
        learning_rate=0.05,         # Learning rate más bajo para convergencia estable

        # Parámetros de regularización
        subsample=0.8,              # Submuestreo para evitar overfitting
        colsample_bytree=0.8,       # Submuestreo de features
        reg_alpha=1,                # Regularización L1
        reg_lambda=1,               # Regularización L2

        # Parámetros de optimización
        min_child_weight=3,         # Mínimo peso en hojas
        gamma=0.1,                  # Mínima pérdida para split

        # Configuración técnica
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,   # Para evitar overfitting
        n_jobs=-1                   # Usar todos los cores
    )

    # Entrenamiento con validación temprana
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return model, accuracy
