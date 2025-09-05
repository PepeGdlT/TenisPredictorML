# src/audit_leakage.py
"""
script para auditar fuga de información en features: correlación con el target.
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

def audit_feature_leakage(X, y):
    """
    calcula la correlación absoluta de cada feature con el target y avisa si alguna es sospechosa (>0.95).
    Ignora columnas no numéricas, constantes o completamente NaN.
    """
    corrs = {}
    for col in X.columns:
        s = X[col]
        if not np.issubdtype(s.dtype, np.number):
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        if s.notna().sum() < 5:
            continue
        try:
            cor = np.corrcoef(s.fillna(s.median()), y)[0, 1]
        except Exception:
            cor = np.nan
        corrs[col] = cor
    corrs = pd.Series(corrs).dropna().abs().sort_values(ascending=False)
    print("\ncorrelación absoluta de cada feature con el target (filtrada):")
    print(corrs)
    high = corrs[corrs > 0.95]
    print("\nfeatures con correlación > 0.95 (posible fuga de información):")
    print(high if not high.empty else "(ninguna)")
    return corrs

def check_index_overlap(X_train, X_test):
    overlap = X_train.index.intersection(X_test.index)
    print("Overlap índices train/test:", len(overlap))
    if len(overlap) > 0:
        print("Ejemplo índices overlapped:", overlap[:10].tolist())

def check_temporal_split(df_full, train_idx, test_idx, date_cols=['match_date','tourney_date','date']):
    date_col = next((c for c in date_cols if c in df_full.columns), None)
    if date_col is None:
        print("No se encontró columna de fecha en df_full. No puedo comprobar split temporal.")
        return
    s = pd.to_datetime(df_full[date_col], errors='coerce')
    train_dates = s.loc[train_idx].dropna()
    test_dates = s.loc[test_idx].dropna()
    if train_dates.empty or test_dates.empty:
        print("Fechas faltantes en train/test para comprobar ordering.")
        return
    print("Máx fecha train:", train_dates.max(), "Min fecha test:", test_dates.min())
    print("¿train_max < test_min? ->", train_dates.max() < test_dates.min())

def compare_feature_distributions(X_train, X_test, features, alpha=0.001):
    for f in features:
        if f not in X_train.columns or f not in X_test.columns:
            print(f"{f}: no está en ambos conjuntos")
            continue
        a = X_train[f].dropna().values
        b = X_test[f].dropna().values
        if len(a) < 10 or len(b) < 10:
            print(f"{f}: muestras pequeñas")
            continue
        stat, p = ks_2samp(a, b)
        print(f"{f}: KS={stat:.3f}, p={p:.3e} -> {'distinta' if p<alpha else 'similar'}")

def quick_permutation_target_test(model, X_train, y_train, X_test, y_test, n_trials=1, random_state=None):
    """Entrena modelo base, evalúa y repite con labels permutados para sanity check.
    Alinea columnas de test a las de train para evitar mismatch (XGBoost exige coincidencia exacta)."""
    # Alinear columnas (intersección implícita) manteniendo orden de treino
    X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
    rng = np.random.RandomState(random_state)
    base = clone(model)
    base.fit(X_train, y_train)
    y_proba = base.predict_proba(X_test_aligned)[:, 1]
    print("Original test AUC:", roc_auc_score(y_test, y_proba), "LogLoss:", log_loss(y_test, y_proba))
    for i in range(n_trials):
        y_sh = rng.permutation(y_train)
        m = clone(model)
        m.fit(X_train, y_sh)
        y_p = m.predict_proba(X_test_aligned)[:, 1]
        print(f"Permutado trial {i+1} -> AUC: {roc_auc_score(y_test, y_p):.4f}, LogLoss: {log_loss(y_test, y_p):.4f}")

def feature_shuffle_ablation(model, X_train, y_train, X_test, y_test, top_features, random_state=None):
    """Baraja (permuta) internamente top_features en train y mide caída de AUC.
    Alinea columnas test-train antes de evaluar para evitar errores."""
    X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
    rng = np.random.RandomState(random_state)
    base = clone(model)
    base.fit(X_train, y_train)
    base_score = roc_auc_score(y_test, base.predict_proba(X_test_aligned)[:, 1])
    print("Baseline AUC:", base_score)
    X_train_sh = X_train.copy()
    for f in top_features:
        if f in X_train_sh.columns:
            X_train_sh[f] = rng.permutation(X_train_sh[f].values)
    m = clone(model)
    m.fit(X_train_sh, y_train)
    score_sh = roc_auc_score(y_test, m.predict_proba(X_test_aligned)[:, 1])
    print("AUC tras shuffle top features:", score_sh, "Delta:", base_score - score_sh)
