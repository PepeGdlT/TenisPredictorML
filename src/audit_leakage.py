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
    ignora columnas no numéricas, constantes o completamente nan.
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
    print("overlap índices train/test:", len(overlap))
    if len(overlap) > 0:
        print("ejemplo índices overlapped:", overlap[:10].tolist())

def check_temporal_split(df_full, train_idx, test_idx, date_cols=['match_date','tourney_date','date']):
    date_col = next((c for c in date_cols if c in df_full.columns), None)
    if date_col is None:
        print("no se encontró columna de fecha en df_full. no puedo comprobar split temporal.")
        return
    s = pd.to_datetime(df_full[date_col], errors='coerce')
    train_dates = s.loc[train_idx].dropna()
    test_dates = s.loc[test_idx].dropna()
    if train_dates.empty or test_dates.empty:
        print("fechas faltantes en train/test para comprobar ordering.")
        return
    print("máx fecha train:", train_dates.max(), "min fecha test:", test_dates.min())
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
    """entrena modelo base, evalúa y repite con labels permutados para sanity check.
    alinea columnas de test a las de train para evitar mismatch (xgboost exige coincidencia exacta)."""
    # alinear columnas (intersección implícita) manteniendo orden de treino
    X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
    rng = np.random.RandomState(random_state)
    base = clone(model)
    base.fit(X_train, y_train)
    y_proba = base.predict_proba(X_test_aligned)[:, 1]
    print("original test auc:", roc_auc_score(y_test, y_proba), "logloss:", log_loss(y_test, y_proba))
    for i in range(n_trials):
        y_sh = rng.permutation(y_train)
        m = clone(model)
        m.fit(X_train, y_sh)
        y_p = m.predict_proba(X_test_aligned)[:, 1]
        print(f"permutado trial {i+1} -> auc: {roc_auc_score(y_test, y_p):.4f}, logloss: {log_loss(y_test, y_p):.4f}")

def feature_shuffle_ablation(model, X_train, y_train, X_test, y_test, top_features, random_state=None):
    """baraja (permuta) internamente top_features en train y mide caída de auc.
    alinea columnas test-train antes de evaluar para evitar errores."""
    X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
    rng = np.random.RandomState(random_state)
    base = clone(model)
    base.fit(X_train, y_train)
    base_score = roc_auc_score(y_test, base.predict_proba(X_test_aligned)[:, 1])
    print("baseline auc:", base_score)
    X_train_sh = X_train.copy()
    for f in top_features:
        if f in X_train_sh.columns:
            idx = rng.permutation(X_train_sh.index)
            X_train_sh[f] = X_train_sh[f].values[idx]
            m = clone(model)
            m.fit(X_train_sh, y_train)
            score = roc_auc_score(y_test, m.predict_proba(X_test_aligned)[:, 1])
            print(f"{f}: auc tras shuffle: {score:.4f} (caída: {base_score-score:.4f})")
