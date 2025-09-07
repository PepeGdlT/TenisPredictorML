# src/enhanced_validation.py
"""
Validación mejorada para detectar overfitting y problemas de generalización
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt

def advanced_temporal_validation(X, y, dates, model, n_splits=5, test_months=3):
    """
    Validación temporal más robusta con múltiples ventanas de test
    """
    df_temp = pd.DataFrame({'date': pd.to_datetime(dates), 'target': y})
    df_temp = df_temp.join(X)
    df_temp = df_temp.sort_values('date')

    results = []
    date_min = df_temp['date'].min()
    date_max = df_temp['date'].max()

    # Crear múltiples splits temporales
    for i in range(n_splits):
        # Punto de corte progresivo
        split_point = date_min + pd.DateOffset(months=6*(i+1))
        test_end = split_point + pd.DateOffset(months=test_months)

        if test_end > date_max:
            break

        train_mask = df_temp['date'] < split_point
        test_mask = (df_temp['date'] >= split_point) & (df_temp['date'] < test_end)

        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            continue

        X_train_split = df_temp[train_mask].drop(['date', 'target'], axis=1)
        y_train_split = df_temp[train_mask]['target']
        X_test_split = df_temp[test_mask].drop(['date', 'target'], axis=1)
        y_test_split = df_temp[test_mask]['target']

        # Alinear columnas
        X_test_split = X_test_split.reindex(columns=X_train_split.columns, fill_value=0)

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_split, y_train_split)

        y_pred = model_clone.predict_proba(X_test_split)[:, 1]
        auc = roc_auc_score(y_test_split, y_pred)
        logloss = log_loss(y_test_split, y_pred)

        results.append({
            'split': i+1,
            'train_end': split_point.strftime('%Y-%m'),
            'test_start': split_point.strftime('%Y-%m'),
            'test_end': test_end.strftime('%Y-%m'),
            'train_size': train_mask.sum(),
            'test_size': test_mask.sum(),
            'auc': auc,
            'logloss': logloss
        })

    return pd.DataFrame(results)

def stability_analysis(X, y, model, n_bootstrap=50, sample_frac=0.8):
    """
    Analiza estabilidad del modelo con bootstrap sampling
    """
    np.random.seed(42)
    aucs = []

    for i in range(n_bootstrap):
        # Sample con reemplazo
        n_samples = int(len(X) * sample_frac)
        idx = np.random.choice(len(X), n_samples, replace=True)

        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        # Split temporal dentro del bootstrap
        split_idx = int(len(X_boot) * 0.8)
        X_train = X_boot.iloc[:split_idx]
        y_train = y_boot.iloc[:split_idx]
        X_test = X_boot.iloc[split_idx:]
        y_test = y_boot.iloc[split_idx:]

        if len(X_test) < 50:
            continue

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)

        y_pred = model_clone.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        aucs.append(auc)

    return {
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'min_auc': np.min(aucs),
        'max_auc': np.max(aucs),
        'aucs': aucs
    }

def feature_importance_stability(X, y, model, n_iterations=20):
    """
    Analiza estabilidad de importancia de features
    """
    np.random.seed(42)
    importance_history = []

    for i in range(n_iterations):
        # Bootstrap sample
        idx = np.random.choice(len(X), len(X), replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_boot, y_boot)

        if hasattr(model_clone, 'feature_importances_'):
            importances = dict(zip(X.columns, model_clone.feature_importances_))
            importance_history.append(importances)

    # Calcular estadísticas por feature
    feature_stats = {}
    for col in X.columns:
        values = [imp_dict.get(col, 0) for imp_dict in importance_history]
        feature_stats[col] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'cv': np.std(values) / (np.mean(values) + 1e-8)  # coeficiente de variación
        }

    return pd.DataFrame(feature_stats).T.sort_values('mean', ascending=False)

def detect_feature_interactions_leakage(X, y, suspicious_threshold=0.95):
    """
    Busca combinaciones de features que puedan causar leakage
    """
    from sklearn.tree import DecisionTreeClassifier
    from itertools import combinations

    leakage_candidates = []

    # Probar pares de features altamente correlacionadas con target
    correlations = {}
    for col in X.select_dtypes(include=[np.number]).columns:
        try:
            corr = np.corrcoef(X[col].fillna(X[col].median()), y)[0, 1]
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        except:
            continue

    high_corr_features = [k for k, v in correlations.items() if v > 0.3]

    # Probar combinaciones de 2-3 features
    for combo_size in [2, 3]:
        if len(high_corr_features) < combo_size:
            continue

        for combo in combinations(high_corr_features[:10], combo_size):  # Limitar para eficiencia
            try:
                X_combo = X[list(combo)].fillna(0)

                # Tree simple para ver si puede predecir perfectamente
                tree = DecisionTreeClassifier(max_depth=5, random_state=42)
                tree.fit(X_combo, y)

                y_pred = tree.predict_proba(X_combo)[:, 1]
                auc = roc_auc_score(y, y_pred)

                if auc > suspicious_threshold:
                    leakage_candidates.append({
                        'features': combo,
                        'auc': auc,
                        'individual_corrs': [correlations.get(f, 0) for f in combo]
                    })
            except:
                continue

    return sorted(leakage_candidates, key=lambda x: x['auc'], reverse=True)

def comprehensive_model_audit(X_train, y_train, X_test, y_test, model, dates_train=None):
    """
    Auditoría completa del modelo
    """
    print("🔍 AUDITORÍA COMPLETA DEL MODELO")
    print("=" * 50)

    # 1. Validación temporal avanzada
    if dates_train is not None:
        print("\n1. Validación Temporal Avanzada:")
        temporal_results = advanced_temporal_validation(X_train, y_train, dates_train, model)
        print(temporal_results)
        print(f"AUC promedio temporal: {temporal_results['auc'].mean():.4f} ±{temporal_results['auc'].std():.4f}")

    # 2. Estabilidad con bootstrap
    print("\n2. Análisis de Estabilidad (Bootstrap):")
    stability = stability_analysis(X_train, y_train, model)
    print(f"AUC medio: {stability['mean_auc']:.4f} ±{stability['std_auc']:.4f}")
    print(f"Rango: [{stability['min_auc']:.4f}, {stability['max_auc']:.4f}]")

    # 3. Estabilidad de importancias
    print("\n3. Estabilidad de Feature Importance:")
    feat_stability = feature_importance_stability(X_train, y_train, model)
    print("Top 10 features más estables (menor CV):")
    print(feat_stability.head(10)[['mean', 'cv']])

    # 4. Detección de interacciones sospechosas
    print("\n4. Búsqueda de Combinaciones con Leakage:")
    leakage_combos = detect_feature_interactions_leakage(X_train, y_train)
    if leakage_combos:
        print("⚠️  Combinaciones sospechosas encontradas:")
        for combo in leakage_combos[:5]:
            print(f"  - {combo['features']}: AUC={combo['auc']:.4f}")
    else:
        print("✅ No se encontraron combinaciones altamente sospechosas")

    # 5. Performance en test
    print(f"\n5. Performance Final en Test:")
    model.fit(X_train, y_train)
    y_pred_test = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_test)
    test_logloss = log_loss(y_test, y_pred_test)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test LogLoss: {test_logloss:.4f}")

    return {
        'temporal_validation': temporal_results if dates_train else None,
        'stability': stability,
        'feature_stability': feat_stability,
        'leakage_combinations': leakage_combos,
        'final_test_auc': test_auc,
        'final_test_logloss': test_logloss
    }
