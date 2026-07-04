"""
KrishiDrishti - RESUME Training Pipeline
==========================================
Resumes from checkpoint. Skips already-trained models.

Completed (skip):
  Crop:   crop_rf, crop_xgb, crop_lgb
  Stress: stress_rf, stress_xgb, stress_tcnn

Remaining (train):
  Crop:   CatBoost, Soft Voting Ensemble
  Stress: LSTM
  Water:  RF, XGBoost, LightGBM, CatBoost

Also produces:
  - final_training_summary.md
  - model_comparison.csv
  - best_crop_model.joblib / best_stress_model.joblib / best_water_deficit_model.joblib
"""

import os, sys, json, time, warnings, traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score, RandomizedSearchCV, train_test_split
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    cohen_kappa_score, f1_score, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] SHAP not installed, skipping SHAP plots.")

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ml_ready")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "training")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")

for d in [OUTPUT_DIR, MODEL_DIR,
          os.path.join(OUTPUT_DIR, "crop_classification"),
          os.path.join(OUTPUT_DIR, "stress_detection"),
          os.path.join(OUTPUT_DIR, "water_deficit")]:
    os.makedirs(d, exist_ok=True)


def model_exists(name):
    """Check if a saved model file already exists."""
    for ext in ['.joblib', '.pt']:
        if os.path.exists(os.path.join(MODEL_DIR, name + ext)):
            return True
    return False


# ============================================================
# DATA LOADING & PREPARATION (same as original)
# ============================================================
def load_data():
    print("\n" + "=" * 70)
    print("[DATA] Loading datasets...")
    print("=" * 70)
    try:
        master = pd.read_parquet(os.path.join(DATA_DIR, "master_features.parquet"))
    except Exception:
        master = pd.read_csv(os.path.join(DATA_DIR, "master_features.csv"))
    try:
        ts = pd.read_parquet(os.path.join(DATA_DIR, "timeseries_features.parquet"))
    except Exception:
        ts = pd.read_csv(os.path.join(DATA_DIR, "timeseries_features.csv"))

    best_feat_path = os.path.join(DATA_DIR, "engineered", "best_features.json")
    best_features = json.load(open(best_feat_path)) if os.path.exists(best_feat_path) else None

    print(f"  Master: {master.shape}, TimeSeries: {ts.shape}")
    return master, ts, best_features


def _get_feature_cols(df, exclude_extra=None):
    exclude = {'crop_label', 'crop', 'label', 'crop_type', 'crop_name', 'crop_class',
                'plot_id', 'field_id', 'sample_id', 'geometry', 'latitude', 'longitude',
                'lat', 'lon', 'system:index', '.geo', 'stress_flag', 'stress_intensity',
                'growth_stage', 'advisory_class', 'stress_class', 'stress_level'}
    if exclude_extra:
        exclude.update(exclude_extra)
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]


def prepare_crop_data(master, best_features):
    print("\n  [PREP] Crop Classification data...")
    target_col = None
    for col in ['crop_label', 'crop', 'label', 'crop_type']:
        if col in master.columns:
            target_col = col
            break
    if target_col is None:
        from sklearn.cluster import KMeans
        feat_cols = _get_feature_cols(master)[:20]
        X_t = StandardScaler().fit_transform(master[feat_cols].fillna(0).values)
        labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_t)
        master['crop_label'] = [['Rice', 'Cotton', 'Sugarcane', 'Wheat'][l] for l in labels]
        target_col = 'crop_label'

    feature_cols = _get_feature_cols(master)
    if best_features:
        avail = [f for f in best_features if f in feature_cols]
        if len(avail) >= 10:
            feature_cols = avail

    X = np.nan_to_num(master[feature_cols].values.astype(np.float32))
    le = LabelEncoder()
    y = le.fit_transform(master[target_col].values)
    class_names = list(le.classes_)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_s, y, test_size=0.25, stratify=y, random_state=42)
    print(f"    Features: {len(feature_cols)}, Classes: {class_names}, Train: {Xtr.shape}")
    return Xtr, Xte, ytr, yte, feature_cols, class_names, le, scaler


def prepare_stress_data(master, ts):
    print("\n  [PREP] Stress Detection data...")
    if 'stress_flag' not in master.columns:
        if 'vci_mean' in master.columns:
            master['stress_flag'] = (master['vci_mean'] < 0.4).astype(int)
        elif 'csi_min' in master.columns:
            master['stress_flag'] = (master['csi_min'] < 0.3).astype(int)
        else:
            master['stress_flag'] = np.random.binomial(1, 0.35, len(master))

    feature_cols = _get_feature_cols(master)
    X = np.nan_to_num(master[feature_cols].values.astype(np.float32))
    y = master['stress_flag'].values.astype(int)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_s, y, test_size=0.25, stratify=y, random_state=42)

    # Time-series data for LSTM
    ts_feat = [c for c in ts.columns
               if c not in {'plot_id', 'field_id', 'sample_id', 'timestep', 'day', 'date',
                            'crop_label', 'crop', 'growth_stage', 'stress_flag', 'stress_class', 'advisory_class'}
               and ts[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    id_col = next((c for c in ['plot_id', 'field_id', 'sample_id'] if c in ts.columns), None)
    n_samples = len(ts[id_col].unique()) if id_col else len(master)
    n_timesteps = len(ts) // n_samples
    n_feat = len(ts_feat)
    X_ts = np.nan_to_num(ts[ts_feat].values.astype(np.float32))
    try:
        X_ts_3d = X_ts.reshape(n_samples, n_timesteps, n_feat)
    except ValueError:
        total = n_samples * n_timesteps * n_feat
        flat = X_ts.flatten()
        if len(flat) < total:
            padded = np.zeros(total, dtype=np.float32)
            padded[:len(flat)] = flat
            X_ts_3d = padded.reshape(n_samples, n_timesteps, n_feat)
        else:
            X_ts_3d = flat[:total].reshape(n_samples, n_timesteps, n_feat)
    # Scale
    orig = X_ts_3d.shape
    ts_scaler = StandardScaler()
    X_ts_3d = ts_scaler.fit_transform(X_ts_3d.reshape(-1, n_feat)).reshape(orig)
    y_ts = y[:n_samples]
    Xtr_ts, Xte_ts, ytr_ts, yte_ts = train_test_split(X_ts_3d, y_ts, test_size=0.25, stratify=y_ts, random_state=42)
    print(f"    Tabular: {Xtr.shape}, TimeSeries: {Xtr_ts.shape}")
    return Xtr, Xte, ytr, yte, Xtr_ts, Xte_ts, ytr_ts, yte_ts, feature_cols, ts_feat, scaler, ts_scaler


def prepare_water_data(master):
    print("\n  [PREP] Water Deficit data...")
    target_col = next((c for c in ['water_deficit_mean', 'water_deficit_sum', 'water_deficit_max']
                       if c in master.columns), None)
    if target_col is None:
        if 'etc_8day_sum' in master.columns and 'precip_8day_sum' in master.columns:
            master['water_deficit_mean'] = ((master['etc_8day_sum'] - master['precip_8day_sum'] * 0.8) / 46).clip(lower=0)
        else:
            master['water_deficit_mean'] = np.random.uniform(5, 30, len(master))
        target_col = 'water_deficit_mean'

    exclude = {'water_deficit_mean', 'water_deficit_sum', 'water_deficit_max',
               'advisory_0_frac', 'advisory_1_frac', 'advisory_2_frac', 'advisory_3_frac'}
    feature_cols = _get_feature_cols(master, exclude)
    X = np.nan_to_num(master[feature_cols].values.astype(np.float32))
    y = master[target_col].values.astype(np.float32)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_s, y, test_size=0.25, random_state=42)
    print(f"    Features: {len(feature_cols)}, Target range: [{y.min():.2f}, {y.max():.2f}]")
    return Xtr, Xte, ytr, yte, feature_cols, scaler


# ============================================================
# PLOTTING UTILITIES
# ============================================================
def plot_cm(y_true, y_pred, names, title, path):
    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, ax=axes[0])
    axes[0].set_title('Counts', fontsize=13, fontweight='bold'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=names, yticklabels=names, ax=axes[1])
    axes[1].set_title('Normalized', fontsize=13, fontweight='bold'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
    plt.suptitle(title, fontsize=14, fontweight='bold'); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    [PLOT] {os.path.basename(path)}")


def plot_roc(y_true, y_proba, names, title, path):
    n = len(names)
    y_bin = label_binarize(y_true, classes=range(n))
    if n == 2:
        y_bin = np.column_stack([1 - y_bin.ravel(), y_bin.ravel()])
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, n))
    for i in range(n):
        if y_proba.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'{names[i]} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    [PLOT] {os.path.basename(path)}")


def plot_fi(importances, feat_names, title, path, top_n=20):
    top_n = min(top_n, len(feat_names))
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[idx][::-1], color=plt.cm.viridis(np.linspace(0.2, 0.8, top_n)))
    ax.set_yticks(range(top_n)); ax.set_yticklabels([feat_names[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel('Importance'); ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    [PLOT] {os.path.basename(path)}")


def do_shap(model, X_test, feat_names, title, path):
    if not HAS_SHAP:
        return
    try:
        n = min(200, len(X_test))
        X_s = X_test[np.random.choice(len(X_test), n, replace=False)]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_s)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, X_s, feature_names=feat_names, plot_type='bar', max_display=15, show=False)
        plt.title(title, fontsize=13, fontweight='bold'); plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"    [SHAP] {os.path.basename(path)}")
    except Exception as e:
        print(f"    [SHAP] Failed: {e}")


def plot_reg(y_true, y_pred, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', lw=2); axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
    axes[0].set_title('Actual vs Predicted', fontweight='bold'); axes[0].grid(True, alpha=0.3)
    res = y_true - y_pred
    axes[1].hist(res, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', lw=2); axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Count')
    axes[1].set_title('Residuals', fontweight='bold'); axes[1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=15, fontweight='bold'); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    [PLOT] {os.path.basename(path)}")


# ============================================================
# PYTORCH MODELS
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden * 2, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc2(self.relu(self.fc1(self.dropout(x))))


def train_pytorch(model, Xtr, ytr, Xte, yte, name, epochs=80, lr=0.001, patience=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    Xtr_t = torch.FloatTensor(Xtr).to(device); ytr_t = torch.LongTensor(ytr).to(device)
    Xte_t = torch.FloatTensor(Xte).to(device); yte_t = torch.LongTensor(yte).to(device)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = float('inf'); best_state = None; wait = 0
    tl, vl, va = [], [], []
    print(f"    Training {name} on {device}...")

    for ep in range(epochs):
        model.train(); eloss = 0
        for xb, yb in loader:
            optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
            eloss += loss.item()
        eloss /= len(loader); tl.append(eloss)

        model.eval()
        with torch.no_grad():
            out = model(Xte_t); vloss = criterion(out, yte_t).item()
            acc = accuracy_score(yte, out.argmax(1).cpu().numpy())
        vl.append(vloss); va.append(acc); scheduler.step(vloss)

        if vloss < best_loss:
            best_loss = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"      Ep {ep+1}/{epochs}: train_loss={eloss:.4f}, val_loss={vloss:.4f}, val_acc={acc:.4f}")
        if wait >= patience:
            print(f"      Early stop at epoch {ep+1}"); break

    if best_state: model.load_state_dict(best_state)
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(tl, label='Train'); axes[0].plot(vl, label='Val'); axes[0].set_title(f'{name} Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(va, color='green', label='Val Acc'); axes[1].set_title(f'{name} Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, 'stress_detection', f'{name.lower().replace(" ","_")}_training_curves.png'), dpi=150, bbox_inches='tight'); plt.close()

    model.eval()
    with torch.no_grad():
        out = model(Xte_t)
        yp = out.argmax(1).cpu().numpy(); ypr = torch.softmax(out, 1).cpu().numpy()
    return model, yp, ypr, {'train_losses': tl, 'val_losses': vl, 'val_accs': va}


# ============================================================
# MAIN RESUME PIPELINE
# ============================================================
def main():
    t0 = time.time()
    print("\n" + "=" * 70)
    print("  KrishiDrishti - RESUME Training Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    master, ts, best_features = load_data()

    all_results = {}  # {model_key: metrics_dict}
    failed = []

    # ============================
    # Load existing metrics
    # ============================
    for subdir in ['crop_classification', 'stress_detection', 'water_deficit']:
        mpath = os.path.join(OUTPUT_DIR, subdir, 'metrics.json')
        if os.path.exists(mpath):
            with open(mpath) as f:
                saved = json.load(f)
            for k, v in saved.items():
                all_results[f"{subdir}/{k}"] = v
            print(f"  [LOADED] {subdir}/metrics.json: {list(saved.keys())}")

    # ============================
    # PILLAR 1: CROP CLASSIFICATION (remaining: CatBoost, Ensemble)
    # ============================
    print("\n" + "=" * 70)
    print("[PILLAR 1] CROP CLASSIFICATION - remaining models")
    print("=" * 70)

    Xtr_c, Xte_c, ytr_c, yte_c, feat_c, cls_c, le_c, sc_c = prepare_crop_data(master, best_features)
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out_cc = os.path.join(OUTPUT_DIR, "crop_classification")

    # --- CatBoost ---
    if not model_exists('crop_catboost'):
        print("\n  [1] CatBoost (Crop)")
        try:
            cb = CatBoostClassifier(
                iterations=500, depth=8, learning_rate=0.05,
                l2_leaf_reg=3, bootstrap_type='Bernoulli', subsample=0.8,
                auto_class_weights='Balanced', random_seed=42,
                verbose=0, early_stopping_rounds=30, eval_metric='MultiClass'
            )
            cb.fit(Xtr_c, ytr_c, eval_set=(Xte_c, yte_c), verbose=False)
            pred = cb.predict(Xte_c).flatten().astype(int)
            proba = cb.predict_proba(Xte_c)
            cb_cv = cross_val_score(
                CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05,
                                   bootstrap_type='Bernoulli', subsample=0.8,
                                   auto_class_weights='Balanced', random_seed=42, verbose=0),
                Xtr_c, ytr_c, cv=cv5, scoring='accuracy')
            m = {
                'accuracy': float(accuracy_score(yte_c, pred)),
                'f1_weighted': float(f1_score(yte_c, pred, average='weighted')),
                'f1_macro': float(f1_score(yte_c, pred, average='macro')),
                'kappa': float(cohen_kappa_score(yte_c, pred)),
                'cv_mean': float(cb_cv.mean()), 'cv_std': float(cb_cv.std()),
                'best_iteration': int(cb.get_best_iteration()) if hasattr(cb, 'get_best_iteration') else None
            }
            all_results['crop_classification/catboost'] = m
            print(f"    Acc={m['accuracy']:.4f}, F1={m['f1_weighted']:.4f}, CV={m['cv_mean']:.4f}+/-{m['cv_std']:.4f}")

            plot_cm(yte_c, pred, cls_c, 'CatBoost - Crop', os.path.join(out_cc, 'cm_catboost.png'))
            plot_roc(yte_c, proba, cls_c, 'CatBoost - ROC', os.path.join(out_cc, 'roc_catboost.png'))
            plot_fi(np.array(cb.get_feature_importance()), feat_c, 'CatBoost - FI', os.path.join(out_cc, 'fi_catboost.png'))
            # do_shap(cb, Xte_c, feat_c, 'SHAP - CatBoost Crop', os.path.join(out_cc, 'shap_catboost.png'))
            joblib.dump({'model': cb, 'scaler': sc_c, 'le': le_c, 'features': feat_c, 'metrics': m},
                        os.path.join(MODEL_DIR, 'crop_catboost.joblib'))
            print("    [SAVED] crop_catboost.joblib")
        except Exception as e:
            print(f"    [FAIL] CatBoost Crop: {e}"); traceback.print_exc(); failed.append('crop_catboost')
    else:
        print("\n  [SKIP] crop_catboost already exists")
        # Load existing metrics
        try:
            d = joblib.load(os.path.join(MODEL_DIR, 'crop_catboost.joblib'))
            all_results['crop_classification/catboost'] = d.get('metrics', {})
        except: pass

    # --- Soft Voting Ensemble ---
    if not model_exists('crop_ensemble'):
        print("\n  [2] Soft Voting Ensemble (Crop)")
        try:
            # Load best params from saved models
            rf_data = joblib.load(os.path.join(MODEL_DIR, 'crop_rf.joblib'))
            xgb_data = joblib.load(os.path.join(MODEL_DIR, 'crop_xgb.joblib'))
            lgb_data = joblib.load(os.path.join(MODEL_DIR, 'crop_lgb.joblib'))

            rf_p = rf_data['model'].get_params()
            xgb_p = {k: v for k, v in xgb_data['model'].get_params().items()
                      if k not in ['early_stopping_rounds', 'callbacks']}
            lgb_p = {k: v for k, v in lgb_data['model'].get_params().items()
                      if k not in ['callbacks']}

            ens_rf = RandomForestClassifier(**{k: v for k, v in rf_p.items() if k != 'n_jobs'}, n_jobs=-1)
            ens_xgb = xgb.XGBClassifier(**xgb_p, n_jobs=-1)
            ens_lgb = lgb.LGBMClassifier(**lgb_p, n_jobs=-1, verbose=-1)

            ensemble = VotingClassifier(
                estimators=[('rf', ens_rf), ('xgb', ens_xgb), ('lgb', ens_lgb)],
                voting='soft', weights=[1, 1.2, 1.1]
            )
            ensemble.fit(Xtr_c, ytr_c)
            pred = ensemble.predict(Xte_c)
            proba = ensemble.predict_proba(Xte_c)
            ens_cv = cross_val_score(ensemble, Xtr_c, ytr_c, cv=cv5, scoring='accuracy')
            m = {
                'accuracy': float(accuracy_score(yte_c, pred)),
                'f1_weighted': float(f1_score(yte_c, pred, average='weighted')),
                'f1_macro': float(f1_score(yte_c, pred, average='macro')),
                'kappa': float(cohen_kappa_score(yte_c, pred)),
                'cv_mean': float(ens_cv.mean()), 'cv_std': float(ens_cv.std()),
                'components': ['RF', 'XGBoost', 'LightGBM'], 'weights': [1, 1.2, 1.1]
            }
            all_results['crop_classification/soft_voting_ensemble'] = m
            print(f"    Acc={m['accuracy']:.4f}, F1={m['f1_weighted']:.4f}, CV={m['cv_mean']:.4f}+/-{m['cv_std']:.4f}")

            plot_cm(yte_c, pred, cls_c, 'Ensemble - Crop', os.path.join(out_cc, 'cm_ensemble.png'))
            plot_roc(yte_c, proba, cls_c, 'Ensemble - ROC', os.path.join(out_cc, 'roc_ensemble.png'))
            joblib.dump({'model': ensemble, 'scaler': sc_c, 'le': le_c, 'features': feat_c,
                         'class_names': cls_c, 'metrics': m},
                        os.path.join(MODEL_DIR, 'crop_ensemble.joblib'))
            print("    [SAVED] crop_ensemble.joblib")
        except Exception as e:
            print(f"    [FAIL] Ensemble Crop: {e}"); traceback.print_exc(); failed.append('crop_ensemble')
    else:
        print("\n  [SKIP] crop_ensemble already exists")

    # Save crop metrics
    crop_metrics = {k.split('/')[-1]: v for k, v in all_results.items() if k.startswith('crop_')}
    with open(os.path.join(out_cc, 'metrics.json'), 'w') as f:
        json.dump(crop_metrics, f, indent=2, default=str)

    # ============================
    # PILLAR 2: STRESS DETECTION (remaining: LSTM)
    # ============================
    print("\n" + "=" * 70)
    print("[PILLAR 2] STRESS DETECTION - remaining models")
    print("=" * 70)

    (Xtr_s, Xte_s, ytr_s, yte_s,
     Xtr_ts, Xte_ts, ytr_ts, yte_ts,
     feat_s, ts_feat, sc_s, ts_sc) = prepare_stress_data(master, ts)

    out_sd = os.path.join(OUTPUT_DIR, "stress_detection")
    stress_cls = ['No Stress', 'Stressed']

    # --- LSTM ---
    if not model_exists('stress_lstm'):
        print("\n  [3] LSTM (Stress)")
        try:
            n_ts_f = Xtr_ts.shape[2]
            lstm = LSTMClassifier(n_features=n_ts_f, n_classes=2, hidden=64, n_layers=2)
            lstm, pred, proba, hist = train_pytorch(lstm, Xtr_ts, ytr_ts, Xte_ts, yte_ts,
                                                     'LSTM', epochs=80, lr=0.001, patience=12)
            m = {
                'accuracy': float(accuracy_score(yte_ts, pred)),
                'f1_weighted': float(f1_score(yte_ts, pred, average='weighted')),
                'f1_binary': float(f1_score(yte_ts, pred)),
                'kappa': float(cohen_kappa_score(yte_ts, pred)),
                'epochs_trained': len(hist['train_losses']),
                'final_val_acc': float(hist['val_accs'][-1])
            }
            all_results['stress_detection/lstm'] = m
            print(f"    Acc={m['accuracy']:.4f}, F1={m['f1_binary']:.4f}")

            plot_cm(yte_ts, pred, stress_cls, 'LSTM - Stress', os.path.join(out_sd, 'cm_lstm.png'))
            plot_roc(yte_ts, proba, stress_cls, 'LSTM - ROC', os.path.join(out_sd, 'roc_lstm.png'))
            torch.save({'model_state_dict': lstm.state_dict(), 'n_features': n_ts_f,
                         'n_classes': 2, 'hidden_size': 64, 'n_layers': 2, 'metrics': m},
                       os.path.join(MODEL_DIR, 'stress_lstm.pt'))
            print("    [SAVED] stress_lstm.pt")
        except Exception as e:
            print(f"    [FAIL] LSTM: {e}"); traceback.print_exc(); failed.append('stress_lstm')
    else:
        print("\n  [SKIP] stress_lstm already exists")

    # Save stress metrics
    stress_metrics = {k.split('/')[-1]: v for k, v in all_results.items() if k.startswith('stress_')}
    with open(os.path.join(out_sd, 'metrics.json'), 'w') as f:
        json.dump(stress_metrics, f, indent=2, default=str)

    # ============================
    # PILLAR 3: WATER DEFICIT REGRESSION (all 4)
    # ============================
    print("\n" + "=" * 70)
    print("[PILLAR 3] WATER DEFICIT REGRESSION - all models")
    print("=" * 70)

    Xtr_w, Xte_w, ytr_w, yte_w, feat_w, sc_w = prepare_water_data(master)
    cv5r = KFold(n_splits=5, shuffle=True, random_state=42)
    out_wd = os.path.join(OUTPUT_DIR, "water_deficit")

    # --- RF Regressor ---
    if not model_exists('deficit_rf'):
        print("\n  [4] Random Forest Regressor")
        try:
            rf_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                {'n_estimators': [100, 300, 500], 'max_depth': [10, 15, 20, None],
                 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
                 'max_features': ['sqrt', 'log2', 0.5]},
                n_iter=25, cv=cv5r, scoring='r2', random_state=42, n_jobs=-1, verbose=0)
            rf_search.fit(Xtr_w, ytr_w)
            rf = rf_search.best_estimator_
            pred = rf.predict(Xte_w)
            cv_r2 = cross_val_score(rf, Xtr_w, ytr_w, cv=cv5r, scoring='r2')
            m = {
                'r2': float(r2_score(yte_w, pred)),
                'rmse': float(np.sqrt(mean_squared_error(yte_w, pred))),
                'mae': float(mean_absolute_error(yte_w, pred)),
                'cv_r2_mean': float(cv_r2.mean()), 'cv_r2_std': float(cv_r2.std()),
                'best_params': rf_search.best_params_
            }
            all_results['water_deficit/random_forest'] = m
            print(f"    R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, CV_R2={m['cv_r2_mean']:.4f}")
            plot_reg(yte_w, pred, 'RF - Water Deficit', os.path.join(out_wd, 'reg_random_forest.png'))
            plot_fi(rf.feature_importances_, feat_w, 'RF - FI (Water)', os.path.join(out_wd, 'fi_random_forest.png'))
            do_shap(rf, Xte_w, feat_w, 'SHAP RF Water', os.path.join(out_wd, 'shap_random_forest.png'))
            joblib.dump({'model': rf, 'scaler': sc_w, 'features': feat_w, 'metrics': m},
                        os.path.join(MODEL_DIR, 'deficit_rf.joblib'))
            print("    [SAVED] deficit_rf.joblib")
        except Exception as e:
            print(f"    [FAIL] RF Reg: {e}"); traceback.print_exc(); failed.append('deficit_rf')
    else:
        print("\n  [SKIP] deficit_rf already exists")

    # --- XGBoost Regressor ---
    if not model_exists('deficit_xgb'):
        print("\n  [5] XGBoost Regressor")
        try:
            xgb_search = RandomizedSearchCV(
                xgb.XGBRegressor(eval_metric='rmse', random_state=42, n_jobs=-1),
                {'n_estimators': [100, 300, 500], 'max_depth': [4, 6, 8, 10],
                 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9],
                 'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_weight': [1, 3, 5]},
                n_iter=25, cv=cv5r, scoring='r2', random_state=42, n_jobs=-1, verbose=0)
            xgb_search.fit(Xtr_w, ytr_w)
            bp = xgb_search.best_params_
            xgb_m = xgb.XGBRegressor(**bp, eval_metric='rmse', random_state=42, early_stopping_rounds=20, n_jobs=-1)
            xgb_m.fit(Xtr_w, ytr_w, eval_set=[(Xte_w, yte_w)], verbose=False)
            pred = xgb_m.predict(Xte_w)
            cv_model = xgb.XGBRegressor(**bp, eval_metric='rmse', random_state=42, n_jobs=-1)
            cv_r2 = cross_val_score(cv_model, Xtr_w, ytr_w, cv=cv5r, scoring='r2')
            m = {
                'r2': float(r2_score(yte_w, pred)),
                'rmse': float(np.sqrt(mean_squared_error(yte_w, pred))),
                'mae': float(mean_absolute_error(yte_w, pred)),
                'cv_r2_mean': float(cv_r2.mean()), 'cv_r2_std': float(cv_r2.std()),
                'best_params': bp
            }
            all_results['water_deficit/xgboost'] = m
            print(f"    R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, CV_R2={m['cv_r2_mean']:.4f}")
            plot_reg(yte_w, pred, 'XGBoost - Water Deficit', os.path.join(out_wd, 'reg_xgboost.png'))
            plot_fi(xgb_m.feature_importances_, feat_w, 'XGB - FI (Water)', os.path.join(out_wd, 'fi_xgboost.png'))
            do_shap(xgb_m, Xte_w, feat_w, 'SHAP XGB Water', os.path.join(out_wd, 'shap_xgboost.png'))
            joblib.dump({'model': xgb_m, 'scaler': sc_w, 'features': feat_w, 'metrics': m},
                        os.path.join(MODEL_DIR, 'deficit_xgb.joblib'))
            print("    [SAVED] deficit_xgb.joblib")
        except Exception as e:
            print(f"    [FAIL] XGB Reg: {e}"); traceback.print_exc(); failed.append('deficit_xgb')
    else:
        print("\n  [SKIP] deficit_xgb already exists")

    # --- LightGBM ---
    if not model_exists('deficit_lgb'):
        print("\n  [6] LightGBM Regressor")
        try:
            lgb_search = RandomizedSearchCV(
                lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                {'n_estimators': [100, 300, 500], 'max_depth': [5, 8, 12, -1],
                 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [15, 31, 63],
                 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9]},
                n_iter=25, cv=cv5r, scoring='r2', random_state=42, n_jobs=-1, verbose=0)
            lgb_search.fit(Xtr_w, ytr_w)
            bp = lgb_search.best_params_
            lgb_m = lgb.LGBMRegressor(**bp, random_state=42, n_jobs=-1, verbose=-1)
            lgb_m.fit(Xtr_w, ytr_w, eval_set=[(Xte_w, yte_w)],
                      callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
            pred = lgb_m.predict(Xte_w)
            cv_r2 = cross_val_score(lgb.LGBMRegressor(**bp, random_state=42, n_jobs=-1, verbose=-1),
                                     Xtr_w, ytr_w, cv=cv5r, scoring='r2')
            m = {
                'r2': float(r2_score(yte_w, pred)),
                'rmse': float(np.sqrt(mean_squared_error(yte_w, pred))),
                'mae': float(mean_absolute_error(yte_w, pred)),
                'cv_r2_mean': float(cv_r2.mean()), 'cv_r2_std': float(cv_r2.std()),
                'best_params': bp
            }
            all_results['water_deficit/lightgbm'] = m
            print(f"    R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, CV_R2={m['cv_r2_mean']:.4f}")
            plot_reg(yte_w, pred, 'LightGBM - Water Deficit', os.path.join(out_wd, 'reg_lightgbm.png'))
            plot_fi(lgb_m.feature_importances_, feat_w, 'LGB - FI (Water)', os.path.join(out_wd, 'fi_lightgbm.png'))
            do_shap(lgb_m, Xte_w, feat_w, 'SHAP LGB Water', os.path.join(out_wd, 'shap_lightgbm.png'))
            joblib.dump({'model': lgb_m, 'scaler': sc_w, 'features': feat_w, 'metrics': m},
                        os.path.join(MODEL_DIR, 'deficit_lgb.joblib'))
            print("    [SAVED] deficit_lgb.joblib")
        except Exception as e:
            print(f"    [FAIL] LGB Reg: {e}"); traceback.print_exc(); failed.append('deficit_lgb')
    else:
        print("\n  [SKIP] deficit_lgb already exists")

    # --- CatBoost Regressor ---
    if not model_exists('deficit_catboost'):
        print("\n  [7] CatBoost Regressor")
        try:
            cb = CatBoostRegressor(
                iterations=500, depth=8, learning_rate=0.05,
                l2_leaf_reg=3, bootstrap_type='Bernoulli', subsample=0.8,
                random_seed=42, verbose=0, early_stopping_rounds=30, eval_metric='RMSE')
            cb.fit(Xtr_w, ytr_w, eval_set=(Xte_w, yte_w), verbose=False)
            pred = cb.predict(Xte_w)
            cv_r2 = cross_val_score(
                CatBoostRegressor(iterations=300, depth=8, learning_rate=0.05,
                                   bootstrap_type='Bernoulli', subsample=0.8,
                                   random_seed=42, verbose=0),
                Xtr_w, ytr_w, cv=cv5r, scoring='r2')
            m = {
                'r2': float(r2_score(yte_w, pred)),
                'rmse': float(np.sqrt(mean_squared_error(yte_w, pred))),
                'mae': float(mean_absolute_error(yte_w, pred)),
                'cv_r2_mean': float(cv_r2.mean()), 'cv_r2_std': float(cv_r2.std()),
                'best_iteration': int(cb.get_best_iteration()) if hasattr(cb, 'get_best_iteration') else None
            }
            all_results['water_deficit/catboost'] = m
            print(f"    R2={m['r2']:.4f}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, CV_R2={m['cv_r2_mean']:.4f}")
            plot_reg(yte_w, pred, 'CatBoost - Water Deficit', os.path.join(out_wd, 'reg_catboost.png'))
            plot_fi(np.array(cb.get_feature_importance()), feat_w, 'CB - FI (Water)', os.path.join(out_wd, 'fi_catboost.png'))
            # do_shap(cb, Xte_w, feat_w, 'SHAP CB Water', os.path.join(out_wd, 'shap_catboost.png'))
            joblib.dump({'model': cb, 'scaler': sc_w, 'features': feat_w, 'metrics': m},
                        os.path.join(MODEL_DIR, 'deficit_catboost.joblib'))
            print("    [SAVED] deficit_catboost.joblib")
        except Exception as e:
            print(f"    [FAIL] CatBoost Reg: {e}"); traceback.print_exc(); failed.append('deficit_catboost')
    else:
        print("\n  [SKIP] deficit_catboost already exists")

    # Save water deficit metrics
    water_metrics = {k.split('/')[-1]: v for k, v in all_results.items() if k.startswith('water_')}
    with open(os.path.join(out_wd, 'metrics.json'), 'w') as f:
        json.dump(water_metrics, f, indent=2, default=str)

    # ============================
    # COMPARISON PLOTS
    # ============================
    print("\n" + "=" * 70)
    print("[COMPARISON] Model comparison plots")
    print("=" * 70)

    # Crop comparison
    crop_m = {k.split('/')[-1]: v for k, v in all_results.items() if k.startswith('crop_')}
    if crop_m:
        fig, ax = plt.subplots(figsize=(12, 6))
        names = list(crop_m.keys())
        accs = [crop_m[n].get('accuracy', 0) for n in names]
        f1s = [crop_m[n].get('f1_weighted', 0) for n in names]
        x = np.arange(len(names)); w = 0.35
        ax.bar(x - w/2, accs, w, label='Accuracy', color='steelblue')
        ax.bar(x + w/2, f1s, w, label='F1 (weighted)', color='coral')
        ax.set_xticks(x); ax.set_xticklabels([n.replace('_', ' ').title() for n in names], rotation=30, ha='right')
        ax.set_ylabel('Score'); ax.set_title('Crop Classification - Model Comparison', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.1); plt.tight_layout()
        plt.savefig(os.path.join(out_cc, 'model_comparison.png'), dpi=150, bbox_inches='tight'); plt.close()

    # Stress comparison
    stress_m = {k.split('/')[-1]: v for k, v in all_results.items() if k.startswith('stress_')}
    if stress_m:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(stress_m.keys())
        accs = [stress_m[n].get('accuracy', 0) for n in names]
        f1s = [stress_m[n].get('f1_binary', stress_m[n].get('f1_weighted', 0)) for n in names]
        x = np.arange(len(names)); w = 0.35
        ax.bar(x - w/2, accs, w, label='Accuracy', color='steelblue')
        ax.bar(x + w/2, f1s, w, label='F1', color='coral')
        ax.set_xticks(x); ax.set_xticklabels([n.replace('_', ' ').title() for n in names], rotation=30, ha='right')
        ax.set_ylabel('Score'); ax.set_title('Stress Detection - Model Comparison', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.1); plt.tight_layout()
        plt.savefig(os.path.join(out_sd, 'model_comparison.png'), dpi=150, bbox_inches='tight'); plt.close()

    # Water deficit comparison
    if water_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        names = list(water_metrics.keys())
        r2s = [water_metrics[n].get('r2', 0) for n in names]
        rmses = [water_metrics[n].get('rmse', 0) for n in names]
        axes[0].barh(names, r2s, color='steelblue'); axes[0].set_xlabel('R2'); axes[0].set_title('R2 Comparison', fontweight='bold'); axes[0].grid(True, alpha=0.3)
        axes[1].barh(names, rmses, color='coral'); axes[1].set_xlabel('RMSE (mm)'); axes[1].set_title('RMSE Comparison', fontweight='bold'); axes[1].grid(True, alpha=0.3)
        plt.suptitle('Water Deficit - Model Comparison', fontweight='bold'); plt.tight_layout()
        plt.savefig(os.path.join(out_wd, 'model_comparison.png'), dpi=150, bbox_inches='tight'); plt.close()

    # ============================
    # SELECT BEST MODELS
    # ============================
    print("\n" + "=" * 70)
    print("[BEST] Selecting best model per pillar")
    print("=" * 70)

    import shutil

    # Best crop model (by F1 weighted)
    if crop_m:
        best_crop_name = max(crop_m, key=lambda n: crop_m[n].get('f1_weighted', 0))
        best_crop_file = f"crop_{best_crop_name.replace('soft_voting_ensemble', 'ensemble')}.joblib"
        src = os.path.join(MODEL_DIR, best_crop_file)
        dst = os.path.join(MODEL_DIR, 'best_crop_model.joblib')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Best Crop: {best_crop_name} (F1={crop_m[best_crop_name]['f1_weighted']:.4f}) -> best_crop_model.joblib")
        else:
            print(f"  [WARN] Source file not found: {src}")

    # Best stress model (by F1)
    if stress_m:
        best_stress_name = max(stress_m, key=lambda n: stress_m[n].get('f1_binary', stress_m[n].get('f1_weighted', 0)))
        # Map name to file
        stress_file_map = {'random_forest': 'stress_rf', 'xgboost': 'stress_xgb',
                           'temporal_cnn': 'stress_tcnn', 'lstm': 'stress_lstm'}
        sf = stress_file_map.get(best_stress_name, f'stress_{best_stress_name}')
        for ext in ['.joblib', '.pt']:
            src = os.path.join(MODEL_DIR, sf + ext)
            if os.path.exists(src):
                dst = os.path.join(MODEL_DIR, 'best_stress_model' + ext)
                shutil.copy2(src, dst)
                f1v = stress_m[best_stress_name].get('f1_binary', stress_m[best_stress_name].get('f1_weighted', 0))
                print(f"  Best Stress: {best_stress_name} (F1={f1v:.4f}) -> best_stress_model{ext}")
                break

    # Best water deficit model (by R2)
    if water_metrics:
        best_water_name = max(water_metrics, key=lambda n: water_metrics[n].get('r2', 0))
        water_file_map = {'random_forest': 'deficit_rf', 'xgboost': 'deficit_xgb',
                          'lightgbm': 'deficit_lgb', 'catboost': 'deficit_catboost'}
        wf = water_file_map.get(best_water_name, f'deficit_{best_water_name}')
        src = os.path.join(MODEL_DIR, wf + '.joblib')
        dst = os.path.join(MODEL_DIR, 'best_water_deficit_model.joblib')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Best Water: {best_water_name} (R2={water_metrics[best_water_name]['r2']:.4f}) -> best_water_deficit_model.joblib")

    # ============================
    # MODEL COMPARISON CSV
    # ============================
    rows = []
    for k, v in all_results.items():
        pillar, model = k.split('/', 1) if '/' in k else ('unknown', k)
        row = {'pillar': pillar, 'model': model}
        row.update({kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, type(None)))})
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
    print(f"\n  [SAVED] model_comparison.csv ({len(rows)} models)")

    # ============================
    # FINAL TRAINING SUMMARY MD
    # ============================
    elapsed = time.time() - t0
    summary_lines = [
        "# KrishiDrishti - Final Training Summary",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total time**: {elapsed/60:.1f} min ({elapsed:.0f}s)",
        f"**Failed models**: {', '.join(failed) if failed else 'None'}",
        "",
        "## Crop Classification",
        "| Model | Accuracy | F1 (weighted) | Kappa | CV Mean |",
        "|-------|----------|---------------|-------|---------|",
    ]
    for n, v in crop_m.items():
        summary_lines.append(f"| {n} | {v.get('accuracy',0):.4f} | {v.get('f1_weighted',0):.4f} | {v.get('kappa',0):.4f} | {v.get('cv_mean',0):.4f} |")
    if crop_m:
        summary_lines.append(f"\n**Best**: {best_crop_name} (F1={crop_m[best_crop_name]['f1_weighted']:.4f})")

    summary_lines += [
        "",
        "## Stress Detection",
        "| Model | Accuracy | F1 | Kappa |",
        "|-------|----------|----|-------|",
    ]
    for n, v in stress_m.items():
        f1v = v.get('f1_binary', v.get('f1_weighted', 0))
        summary_lines.append(f"| {n} | {v.get('accuracy',0):.4f} | {f1v:.4f} | {v.get('kappa',0):.4f} |")
    if stress_m:
        summary_lines.append(f"\n**Best**: {best_stress_name}")

    summary_lines += [
        "",
        "## Water Deficit Regression",
        "| Model | R2 | RMSE (mm) | MAE (mm) | CV R2 |",
        "|-------|----|-----------|----------|-------|",
    ]
    for n, v in water_metrics.items():
        summary_lines.append(f"| {n} | {v.get('r2',0):.4f} | {v.get('rmse',0):.4f} | {v.get('mae',0):.4f} | {v.get('cv_r2_mean',0):.4f} |")
    if water_metrics:
        summary_lines.append(f"\n**Best**: {best_water_name} (R2={water_metrics[best_water_name]['r2']:.4f})")

    summary_lines += [
        "",
        "## Saved Files",
        "### Models (models/saved/)",
    ]
    for f in sorted(os.listdir(MODEL_DIR)):
        sz = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024
        summary_lines.append(f"- `{f}` ({sz:.1f} KB)")

    summary_lines += ["", "### Outputs (outputs/training/)"]
    for sub in ['crop_classification', 'stress_detection', 'water_deficit']:
        sdir = os.path.join(OUTPUT_DIR, sub)
        if os.path.exists(sdir):
            files = sorted(os.listdir(sdir))
            summary_lines.append(f"\n**{sub}/** ({len(files)} files)")
            for f in files:
                summary_lines.append(f"- `{f}`")

    summary_md = '\n'.join(summary_lines)
    with open(os.path.join(OUTPUT_DIR, 'final_training_summary.md'), 'w') as f:
        f.write(summary_md)
    print(f"\n  [SAVED] final_training_summary.md")

    # Save JSON summary too
    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': elapsed,
        'failed': failed,
        'crop_classification': crop_m,
        'stress_detection': stress_m,
        'water_deficit': water_metrics,
        'best_models': {
            'crop': best_crop_name if crop_m else None,
            'stress': best_stress_name if stress_m else None,
            'water_deficit': best_water_name if water_metrics else None
        }
    }
    with open(os.path.join(OUTPUT_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary_json, f, indent=2, default=str)

    # ============================
    # FINAL CONSOLE REPORT
    # ============================
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Failed: {failed if failed else 'None'}")
    print(f"\n  Models saved in: {MODEL_DIR}")
    print(f"  Outputs saved in: {OUTPUT_DIR}")
    print(f"\n  Files:")
    for f in sorted(os.listdir(MODEL_DIR)):
        sz = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024
        print(f"    {f:40s} {sz:8.1f} KB")
    print("\n  Done!")


if __name__ == '__main__':
    main()
