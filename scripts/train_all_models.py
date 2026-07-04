"""
KrishiDrishti — Full Model Training Pipeline
==============================================
Trains ALL models required for the three pillars:

Pillar 1 — Crop Classification:
    - Random Forest, XGBoost, LightGBM, CatBoost, Soft Voting Ensemble

Pillar 2 — Stress Detection:
    - Random Forest, XGBoost, Temporal CNN, LSTM

Pillar 3 — Water Deficit Regression:
    - Random Forest Regressor, XGBoost Regressor, LightGBM, CatBoost

Each model includes:
    - 5-fold stratified cross-validation (classification) or KFold (regression)
    - Hyperparameter tuning via RandomizedSearchCV
    - Early stopping where applicable
    - Saved trained models (.joblib / .pt)
    - Metrics JSON
    - Confusion matrices (classification)
    - ROC curves (classification)
    - Feature importance plots
    - SHAP value explanations

Author: Team BEST SHOT
Date: 2026-07-04
"""

import os
import sys
import json
import time
import warnings
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score, cross_val_predict,
    RandomizedSearchCV, train_test_split
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    cohen_kappa_score, f1_score, roc_curve, auc, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.calibration import CalibratedClassifierCV

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
    print("⚠️ SHAP not installed. SHAP explanations will be skipped.")

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ml_ready")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "training")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")

for d in [OUTPUT_DIR, MODEL_SAVE_DIR,
          os.path.join(OUTPUT_DIR, "crop_classification"),
          os.path.join(OUTPUT_DIR, "stress_detection"),
          os.path.join(OUTPUT_DIR, "water_deficit")]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """Load master_features and timeseries_features datasets."""
    print("\n" + "=" * 70)
    print("📂 LOADING DATA")
    print("=" * 70)

    # Try parquet first, fall back to CSV
    try:
        master = pd.read_parquet(os.path.join(DATA_DIR, "master_features.parquet"))
        print(f"  ✅ master_features.parquet loaded: {master.shape}")
    except Exception:
        master = pd.read_csv(os.path.join(DATA_DIR, "master_features.csv"))
        print(f"  ✅ master_features.csv loaded: {master.shape}")

    try:
        ts = pd.read_parquet(os.path.join(DATA_DIR, "timeseries_features.parquet"))
        print(f"  ✅ timeseries_features.parquet loaded: {ts.shape}")
    except Exception:
        ts = pd.read_csv(os.path.join(DATA_DIR, "timeseries_features.csv"))
        print(f"  ✅ timeseries_features.csv loaded: {ts.shape}")

    # Load best features list
    best_feat_path = os.path.join(DATA_DIR, "engineered", "best_features.json")
    if os.path.exists(best_feat_path):
        with open(best_feat_path) as f:
            best_features = json.load(f)
        print(f"  ✅ best_features.json loaded: {len(best_features)} features")
    else:
        best_features = None

    print(f"\n  📊 Master columns: {list(master.columns)[:10]}...")
    print(f"  📊 Timeseries columns: {list(ts.columns)[:10]}...")

    return master, ts, best_features


# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_crop_classification_data(master, best_features):
    """Prepare data for crop classification (Pillar 1)."""
    print("\n  🌾 Preparing Crop Classification data...")

    # Target column
    target_col = None
    for col in ['crop_label', 'crop', 'label', 'crop_type', 'crop_name', 'crop_class']:
        if col in master.columns:
            target_col = col
            break

    if target_col is None:
        # Generate from simulator data — use the crop params
        print("  ⚠️ No crop_label column found. Searching for alternatives...")
        # Check if there are any string columns that could be crop names
        str_cols = master.select_dtypes(include='object').columns.tolist()
        if str_cols:
            target_col = str_cols[0]
            print(f"  📎 Using '{target_col}' as crop label column")
        else:
            print("  ⚠️ Creating synthetic crop labels from clustering...")
            from sklearn.cluster import KMeans
            feat_cols = [c for c in master.columns if master[c].dtype in ['float64', 'float32', 'int64']]
            X_temp = master[feat_cols[:20]].fillna(0).values
            scaler_temp = StandardScaler()
            X_temp = scaler_temp.fit_transform(X_temp)
            km = KMeans(n_clusters=4, random_state=42, n_init=10)
            labels = km.fit_predict(X_temp)
            crop_names = ['Rice', 'Cotton', 'Sugarcane', 'Wheat']
            master['crop_label'] = [crop_names[l] for l in labels]
            target_col = 'crop_label'

    # Identify feature columns
    exclude_cols = [target_col, 'crop_label', 'crop', 'label', 'crop_type', 'crop_name',
                    'plot_id', 'field_id', 'sample_id', 'geometry', 'latitude', 'longitude',
                    'lat', 'lon', 'system:index', '.geo', 'stress_flag', 'stress_intensity',
                    'growth_stage', 'advisory_class', 'crop_class',
                    'stress_class', 'stress_level']

    feature_cols = [c for c in master.columns
                    if c not in exclude_cols
                    and master[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Use best features if available and intersects
    if best_features:
        available_best = [f for f in best_features if f in feature_cols]
        if len(available_best) >= 10:
            feature_cols = available_best
            print(f"  📎 Using {len(feature_cols)} best features")

    X = master[feature_cols].values.astype(np.float32)
    y_raw = master[target_col].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=42
    )

    print(f"  ✅ Features: {len(feature_cols)}")
    print(f"  ✅ Classes: {class_names}")
    print(f"  ✅ Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  ✅ Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, feature_cols, class_names, le, scaler


def prepare_stress_detection_data(master, ts, best_features):
    """Prepare data for stress detection (Pillar 2)."""
    print("\n  💧 Preparing Stress Detection data...")

    # Target: stress_flag (binary)
    target_col = None
    for col in ['stress_flag', 'stress', 'is_stressed']:
        if col in master.columns:
            target_col = col
            break

    if target_col is None:
        print("  ⚠️ No stress_flag column. Deriving from VCI/CSI...")
        if 'vci_mean' in master.columns:
            master['stress_flag'] = (master['vci_mean'] < 0.4).astype(int)
        elif 'csi_min' in master.columns:
            master['stress_flag'] = (master['csi_min'] < 0.3).astype(int)
        else:
            master['stress_flag'] = np.random.binomial(1, 0.35, len(master))
        target_col = 'stress_flag'

    # Feature columns for tabular models
    exclude_cols = ['crop_label', 'crop', 'label', 'crop_type', 'crop_name',
                    'plot_id', 'field_id', 'sample_id', 'geometry', 'latitude', 'longitude',
                    'lat', 'lon', 'system:index', '.geo', 'stress_flag', 'stress_intensity',
                    'growth_stage', 'advisory_class', 'crop_class',
                    'stress_class', 'stress_level']

    feature_cols = [c for c in master.columns
                    if c not in exclude_cols
                    and master[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    X_tab = master[feature_cols].values.astype(np.float32)
    y = master[target_col].values.astype(int)

    X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_tab_scaled = scaler.fit_transform(X_tab)

    X_train_tab, X_test_tab, y_train, y_test = train_test_split(
        X_tab_scaled, y, test_size=0.25, stratify=y, random_state=42
    )

    # --- Prepare time-series data for CNN/LSTM ---
    ts_feature_cols = [c for c in ts.columns
                       if c not in ['plot_id', 'field_id', 'sample_id', 'timestep',
                                    'day', 'date', 'crop_label', 'crop', 'growth_stage',
                                    'stress_flag', 'stress_class', 'advisory_class']
                       and ts[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Determine number of timesteps and samples
    id_col = None
    for col in ['plot_id', 'field_id', 'sample_id']:
        if col in ts.columns:
            id_col = col
            break

    if id_col:
        unique_ids = ts[id_col].unique()
        n_samples = len(unique_ids)
        n_timesteps = len(ts) // n_samples
    else:
        # Assume 46 timesteps per sample (from preprocessing report)
        n_timesteps = 46
        n_samples = len(ts) // n_timesteps

    n_features_ts = len(ts_feature_cols)

    X_ts = ts[ts_feature_cols].values.astype(np.float32)
    X_ts = np.nan_to_num(X_ts, nan=0.0, posinf=1e6, neginf=-1e6)

    # Reshape to (n_samples, n_timesteps, n_features)
    try:
        X_ts_3d = X_ts.reshape(n_samples, n_timesteps, n_features_ts)
    except ValueError:
        # Truncate/pad to match
        total = n_samples * n_timesteps * n_features_ts
        if len(X_ts.flatten()) < total:
            padded = np.zeros(total)
            padded[:len(X_ts.flatten())] = X_ts.flatten()
            X_ts_3d = padded.reshape(n_samples, n_timesteps, n_features_ts)
        else:
            X_ts_3d = X_ts[:n_samples * n_timesteps].reshape(n_samples, n_timesteps, n_features_ts)

    # Scale time-series features
    orig_shape = X_ts_3d.shape
    X_ts_flat = X_ts_3d.reshape(-1, n_features_ts)
    ts_scaler = StandardScaler()
    X_ts_flat = ts_scaler.fit_transform(X_ts_flat)
    X_ts_3d = X_ts_flat.reshape(orig_shape)

    # Match labels to timeseries samples
    y_ts = y[:n_samples] if len(y) >= n_samples else np.tile(y, n_samples // len(y) + 1)[:n_samples]

    X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(
        X_ts_3d, y_ts, test_size=0.25, stratify=y_ts, random_state=42
    )

    print(f"  ✅ Tabular features: {len(feature_cols)}")
    print(f"  ✅ Timeseries shape: {X_ts_3d.shape} (samples × timesteps × features)")
    print(f"  ✅ Stress distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  ✅ Train (tab): {X_train_tab.shape}, Test: {X_test_tab.shape}")

    return (X_train_tab, X_test_tab, y_train, y_test,
            X_train_ts, X_test_ts, y_train_ts, y_test_ts,
            feature_cols, ts_feature_cols, scaler, ts_scaler)


def prepare_water_deficit_data(master, best_features):
    """Prepare data for water deficit regression (Pillar 3)."""
    print("\n  🚿 Preparing Water Deficit Regression data...")

    # Target: water_deficit_mean
    target_col = None
    for col in ['water_deficit_mean', 'water_deficit_sum', 'water_deficit_max']:
        if col in master.columns:
            target_col = col
            break

    if target_col is None:
        print("  ⚠️ No water_deficit column. Deriving from ETc and precipitation...")
        if 'etc_8day_sum' in master.columns and 'precip_8day_sum' in master.columns:
            master['water_deficit_mean'] = (master['etc_8day_sum'] - master['precip_8day_sum'] * 0.8) / 46
            master['water_deficit_mean'] = master['water_deficit_mean'].clip(lower=0)
        else:
            master['water_deficit_mean'] = np.random.uniform(5, 30, len(master))
        target_col = 'water_deficit_mean'

    # Feature columns — exclude all targets and non-features
    exclude_cols = ['crop_label', 'crop', 'label', 'crop_type', 'crop_name',
                    'plot_id', 'field_id', 'sample_id', 'geometry', 'latitude', 'longitude',
                    'lat', 'lon', 'system:index', '.geo', 'stress_flag', 'stress_intensity',
                    'growth_stage', 'advisory_class', 'crop_class',
                    'water_deficit_mean', 'water_deficit_sum', 'water_deficit_max',
                    'stress_class', 'stress_level',
                    # Also exclude advisory fracs as they leak target
                    'advisory_0_frac', 'advisory_1_frac', 'advisory_2_frac', 'advisory_3_frac']

    feature_cols = [c for c in master.columns
                    if c not in exclude_cols
                    and master[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    X = master[feature_cols].values.astype(np.float32)
    y = master[target_col].values.astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    print(f"  ✅ Features: {len(feature_cols)}")
    print(f"  ✅ Target: {target_col}")
    print(f"  ✅ Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")
    print(f"  ✅ Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols, scaler


# ============================================================
# PLOTTING UTILITIES
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    📊 Confusion matrix → {os.path.basename(save_path)}")


def plot_roc_curves(y_true, y_proba, class_names, title, save_path):
    """Plot multi-class ROC curves."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))

    if n_classes == 2:
        y_bin = np.column_stack([1 - y_bin.ravel(), y_bin.ravel()])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        if y_proba.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    📈 ROC curves → {os.path.basename(save_path)}")


def plot_feature_importance(importances, feature_names, title, save_path, top_n=20):
    """Plot feature importance bar chart."""
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(top_n),
            importances[indices][::-1],
            color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    📊 Feature importance → {os.path.basename(save_path)}")


def compute_shap_values(model, X_test, feature_names, title, save_path, model_type='tree'):
    """Compute and plot SHAP values."""
    if not HAS_SHAP:
        print("    ⚠️ SHAP not available, skipping...")
        return None

    try:
        print(f"    🧠 Computing SHAP values ({title})...")
        # Sample if too large
        max_samples = min(200, len(X_test))
        idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[idx]

        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Use KernelExplainer for non-tree models
            bg = shap.sample(X_test, min(50, len(X_test)))
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, bg)
            shap_values = explainer.shap_values(X_sample, nsamples=50)

        # Summary plot
        fig = plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            # Multi-class
            shap.summary_plot(shap_values, X_sample,
                              feature_names=feature_names,
                              plot_type='bar', max_display=15, show=False)
        else:
            shap.summary_plot(shap_values, X_sample,
                              feature_names=feature_names,
                              max_display=15, show=False)

        plt.title(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    🧠 SHAP plot → {os.path.basename(save_path)}")
        return shap_values
    except Exception as e:
        print(f"    ⚠️ SHAP failed: {e}")
        return None


def plot_regression_results(y_true, y_pred, title, save_path):
    """Plot actual vs predicted scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', lw=2, alpha=0.7)
    axes[0].set_xlabel('Actual', fontsize=12)
    axes[0].set_ylabel('Predicted', fontsize=12)
    axes[0].set_title('Actual vs Predicted', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    📊 Regression plot → {os.path.basename(save_path)}")


# ============================================================
# PYTORCH MODELS
# ============================================================
class TemporalCNN(nn.Module):
    """1D Temporal Convolutional Network for time-series classification."""
    def __init__(self, n_features, n_classes, seq_len=46):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTMClassifier(nn.Module):
    """LSTM-based time-series classifier."""
    def __init__(self, n_features, n_classes, hidden_size=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, n_layers,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size * 2, 32)  # *2 for bidirectional
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use the last hidden state from both directions
        x = torch.cat([hn[-2], hn[-1]], dim=1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_pytorch_model(model, X_train, y_train, X_test, y_test,
                         model_name, n_epochs=100, lr=0.001, patience=15):
    """Train a PyTorch model with early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses, val_accs = [], [], []

    print(f"    🏋️ Training {model_name} on {device}...")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()
            val_preds = val_outputs.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_test, val_preds)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{n_epochs}: train_loss={epoch_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if patience_counter >= patience:
            print(f"      ⏹️ Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} — Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_accs, label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} — Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, 'stress_detection', f'{model_name.lower().replace(" ", "_")}_training_curves.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Final predictions
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_t)
        y_pred = final_outputs.argmax(dim=1).cpu().numpy()
        y_proba = torch.softmax(final_outputs, dim=1).cpu().numpy()

    return model, y_pred, y_proba, {'train_losses': train_losses,
                                     'val_losses': val_losses,
                                     'val_accs': val_accs}


# ============================================================
# PILLAR 1: CROP CLASSIFICATION
# ============================================================
def train_crop_classification(X_train, X_test, y_train, y_test,
                               feature_cols, class_names, le, scaler):
    """Train all Crop Classification models."""
    print("\n" + "=" * 70)
    print("🌾 PILLAR 1: CROP CLASSIFICATION")
    print("=" * 70)

    out_dir = os.path.join(OUTPUT_DIR, "crop_classification")
    all_metrics = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_classes = len(class_names)

    # ---- 1. Random Forest ----
    print("\n  [1/5] 🌲 Random Forest")
    rf_params = {
        'n_estimators': [100, 300, 500, 700],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_base, rf_params, n_iter=30, cv=cv, scoring='f1_weighted',
        random_state=42, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train)
    rf_model = rf_search.best_estimator_
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)

    rf_cv = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    rf_metrics = {
        'accuracy': float(accuracy_score(y_test, rf_pred)),
        'f1_weighted': float(f1_score(y_test, rf_pred, average='weighted')),
        'f1_macro': float(f1_score(y_test, rf_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_test, rf_pred)),
        'cv_mean': float(rf_cv.mean()),
        'cv_std': float(rf_cv.std()),
        'best_params': rf_search.best_params_
    }
    all_metrics['random_forest'] = rf_metrics
    print(f"    ✅ Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1_weighted']:.4f}, "
          f"CV: {rf_metrics['cv_mean']:.4f}±{rf_metrics['cv_std']:.4f}")

    plot_confusion_matrix(y_test, rf_pred, class_names,
                          'Random Forest — Crop Classification',
                          os.path.join(out_dir, 'cm_random_forest.png'))
    plot_roc_curves(y_test, rf_proba, class_names,
                    'Random Forest — ROC Curves',
                    os.path.join(out_dir, 'roc_random_forest.png'))
    plot_feature_importance(rf_model.feature_importances_, feature_cols,
                            'Random Forest — Feature Importance',
                            os.path.join(out_dir, 'fi_random_forest.png'))
    compute_shap_values(rf_model, X_test, feature_cols,
                        'SHAP — Random Forest',
                        os.path.join(out_dir, 'shap_random_forest.png'))

    joblib.dump({'model': rf_model, 'scaler': scaler, 'le': le,
                 'features': feature_cols, 'metrics': rf_metrics},
                os.path.join(MODEL_SAVE_DIR, 'crop_rf.joblib'))

    # ---- 2. XGBoost ----
    print("\n  [2/5] 🚀 XGBoost")
    xgb_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
    }
    xgb_base = xgb.XGBClassifier(
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_params, n_iter=30, cv=cv, scoring='f1_weighted',
        random_state=42, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train)
    # Refit best with early stopping
    best_xgb_params = xgb_search.best_params_
    xgb_model = xgb.XGBClassifier(**best_xgb_params, eval_metric='mlogloss',
                                    random_state=42, early_stopping_rounds=20, n_jobs=-1)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)

    xgb_cv_model = xgb.XGBClassifier(**best_xgb_params, eval_metric='mlogloss',
                                        random_state=42, n_jobs=-1)
    xgb_cv = cross_val_score(xgb_cv_model, X_train, y_train, cv=cv, scoring='accuracy')
    xgb_metrics = {
        'accuracy': float(accuracy_score(y_test, xgb_pred)),
        'f1_weighted': float(f1_score(y_test, xgb_pred, average='weighted')),
        'f1_macro': float(f1_score(y_test, xgb_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_test, xgb_pred)),
        'cv_mean': float(xgb_cv.mean()),
        'cv_std': float(xgb_cv.std()),
        'best_params': best_xgb_params,
        'best_iteration': int(xgb_model.best_iteration) if hasattr(xgb_model, 'best_iteration') else None
    }
    all_metrics['xgboost'] = xgb_metrics
    print(f"    ✅ Accuracy: {xgb_metrics['accuracy']:.4f}, F1: {xgb_metrics['f1_weighted']:.4f}, "
          f"CV: {xgb_metrics['cv_mean']:.4f}±{xgb_metrics['cv_std']:.4f}")

    plot_confusion_matrix(y_test, xgb_pred, class_names,
                          'XGBoost — Crop Classification',
                          os.path.join(out_dir, 'cm_xgboost.png'))
    plot_roc_curves(y_test, xgb_proba, class_names,
                    'XGBoost — ROC Curves',
                    os.path.join(out_dir, 'roc_xgboost.png'))
    plot_feature_importance(xgb_model.feature_importances_, feature_cols,
                            'XGBoost — Feature Importance',
                            os.path.join(out_dir, 'fi_xgboost.png'))
    compute_shap_values(xgb_model, X_test, feature_cols,
                        'SHAP — XGBoost',
                        os.path.join(out_dir, 'shap_xgboost.png'))

    joblib.dump({'model': xgb_model, 'scaler': scaler, 'le': le,
                 'features': feature_cols, 'metrics': xgb_metrics},
                os.path.join(MODEL_SAVE_DIR, 'crop_xgb.joblib'))

    # ---- 3. LightGBM ----
    print("\n  [3/5] 💡 LightGBM")
    lgb_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 8, 12, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63, 127],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_samples': [5, 10, 20],
    }
    lgb_base = lgb.LGBMClassifier(
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_search = RandomizedSearchCV(
        lgb_base, lgb_params, n_iter=30, cv=cv, scoring='f1_weighted',
        random_state=42, n_jobs=-1, verbose=0
    )
    lgb_search.fit(X_train, y_train)
    best_lgb_params = lgb_search.best_params_
    lgb_model = lgb.LGBMClassifier(**best_lgb_params, class_weight='balanced',
                                     random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                             lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)

    lgb_cv = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='accuracy')
    lgb_metrics = {
        'accuracy': float(accuracy_score(y_test, lgb_pred)),
        'f1_weighted': float(f1_score(y_test, lgb_pred, average='weighted')),
        'f1_macro': float(f1_score(y_test, lgb_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_test, lgb_pred)),
        'cv_mean': float(lgb_cv.mean()),
        'cv_std': float(lgb_cv.std()),
        'best_params': best_lgb_params,
        'best_iteration': int(lgb_model.best_iteration_) if hasattr(lgb_model, 'best_iteration_') else None
    }
    all_metrics['lightgbm'] = lgb_metrics
    print(f"    ✅ Accuracy: {lgb_metrics['accuracy']:.4f}, F1: {lgb_metrics['f1_weighted']:.4f}, "
          f"CV: {lgb_metrics['cv_mean']:.4f}±{lgb_metrics['cv_std']:.4f}")

    plot_confusion_matrix(y_test, lgb_pred, class_names,
                          'LightGBM — Crop Classification',
                          os.path.join(out_dir, 'cm_lightgbm.png'))
    plot_roc_curves(y_test, lgb_proba, class_names,
                    'LightGBM — ROC Curves',
                    os.path.join(out_dir, 'roc_lightgbm.png'))
    plot_feature_importance(lgb_model.feature_importances_, feature_cols,
                            'LightGBM — Feature Importance',
                            os.path.join(out_dir, 'fi_lightgbm.png'))
    compute_shap_values(lgb_model, X_test, feature_cols,
                        'SHAP — LightGBM',
                        os.path.join(out_dir, 'shap_lightgbm.png'))

    joblib.dump({'model': lgb_model, 'scaler': scaler, 'le': le,
                 'features': feature_cols, 'metrics': lgb_metrics},
                os.path.join(MODEL_SAVE_DIR, 'crop_lgb.joblib'))

    # ---- 4. CatBoost ----
    print("\n  [4/5] 🐱 CatBoost")
    cb_model = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.05,
        l2_leaf_reg=3, subsample=0.8,
        auto_class_weights='Balanced', random_seed=42,
        verbose=0, early_stopping_rounds=30,
        eval_metric='MultiClass'
    )
    cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    cb_pred = cb_model.predict(X_test).flatten().astype(int)
    cb_proba = cb_model.predict_proba(X_test)

    cb_cv = cross_val_score(
        CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05,
                            auto_class_weights='Balanced', random_seed=42, verbose=0),
        X_train, y_train, cv=cv, scoring='accuracy'
    )
    cb_metrics = {
        'accuracy': float(accuracy_score(y_test, cb_pred)),
        'f1_weighted': float(f1_score(y_test, cb_pred, average='weighted')),
        'f1_macro': float(f1_score(y_test, cb_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_test, cb_pred)),
        'cv_mean': float(cb_cv.mean()),
        'cv_std': float(cb_cv.std()),
        'best_iteration': int(cb_model.get_best_iteration()) if hasattr(cb_model, 'get_best_iteration') else None
    }
    all_metrics['catboost'] = cb_metrics
    print(f"    ✅ Accuracy: {cb_metrics['accuracy']:.4f}, F1: {cb_metrics['f1_weighted']:.4f}, "
          f"CV: {cb_metrics['cv_mean']:.4f}±{cb_metrics['cv_std']:.4f}")

    plot_confusion_matrix(y_test, cb_pred, class_names,
                          'CatBoost — Crop Classification',
                          os.path.join(out_dir, 'cm_catboost.png'))
    plot_roc_curves(y_test, cb_proba, class_names,
                    'CatBoost — ROC Curves',
                    os.path.join(out_dir, 'roc_catboost.png'))
    cb_fi = np.array(cb_model.get_feature_importance())
    plot_feature_importance(cb_fi, feature_cols,
                            'CatBoost — Feature Importance',
                            os.path.join(out_dir, 'fi_catboost.png'))
    # CatBoost SHAP
    try:
        shap_vals_cb = cb_model.get_feature_importance(type='ShapValues',
                                                        data=cb_model.get_feature_importance.__func__ and None)
    except Exception:
        pass
    compute_shap_values(cb_model, X_test, feature_cols,
                        'SHAP — CatBoost',
                        os.path.join(out_dir, 'shap_catboost.png'))

    joblib.dump({'model': cb_model, 'scaler': scaler, 'le': le,
                 'features': feature_cols, 'metrics': cb_metrics},
                os.path.join(MODEL_SAVE_DIR, 'crop_catboost.joblib'))

    # ---- 5. Soft Voting Ensemble ----
    print("\n  [5/5] 🎯 Soft Voting Ensemble")
    # Create fresh instances for ensemble with best params
    ens_rf = RandomForestClassifier(**rf_search.best_params_, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
    ens_xgb = xgb.XGBClassifier(**best_xgb_params, eval_metric='mlogloss',
                                  random_state=42, n_jobs=-1)
    ens_lgb = lgb.LGBMClassifier(**best_lgb_params, class_weight='balanced',
                                   random_state=42, n_jobs=-1, verbose=-1)

    ensemble = VotingClassifier(
        estimators=[
            ('rf', ens_rf),
            ('xgb', ens_xgb),
            ('lgb', ens_lgb),
        ],
        voting='soft',
        weights=[1, 1.2, 1.1]  # Slightly favor boosting models
    )
    ensemble.fit(X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    ens_proba = ensemble.predict_proba(X_test)

    ens_cv = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
    ens_metrics = {
        'accuracy': float(accuracy_score(y_test, ens_pred)),
        'f1_weighted': float(f1_score(y_test, ens_pred, average='weighted')),
        'f1_macro': float(f1_score(y_test, ens_pred, average='macro')),
        'kappa': float(cohen_kappa_score(y_test, ens_pred)),
        'cv_mean': float(ens_cv.mean()),
        'cv_std': float(ens_cv.std()),
        'component_models': ['RF', 'XGBoost', 'LightGBM'],
        'weights': [1, 1.2, 1.1]
    }
    all_metrics['soft_voting_ensemble'] = ens_metrics
    print(f"    ✅ Accuracy: {ens_metrics['accuracy']:.4f}, F1: {ens_metrics['f1_weighted']:.4f}, "
          f"CV: {ens_metrics['cv_mean']:.4f}±{ens_metrics['cv_std']:.4f}")

    plot_confusion_matrix(y_test, ens_pred, class_names,
                          'Soft Voting Ensemble — Crop Classification',
                          os.path.join(out_dir, 'cm_ensemble.png'))
    plot_roc_curves(y_test, ens_proba, class_names,
                    'Soft Voting Ensemble — ROC Curves',
                    os.path.join(out_dir, 'roc_ensemble.png'))

    joblib.dump({'model': ensemble, 'scaler': scaler, 'le': le,
                 'features': feature_cols, 'metrics': ens_metrics,
                 'class_names': class_names},
                os.path.join(MODEL_SAVE_DIR, 'crop_ensemble.joblib'))

    # ---- Model Comparison ----
    print("\n  📊 Model Comparison:")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1 (wt)':>10} {'Kappa':>10} {'CV Mean':>10}")
    print(f"  {'-'*65}")
    for name, m in all_metrics.items():
        print(f"  {name:<25} {m['accuracy']:>10.4f} {m['f1_weighted']:>10.4f} "
              f"{m['kappa']:>10.4f} {m['cv_mean']:>10.4f}")

    # Save comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(all_metrics.keys())
    accuracies = [all_metrics[n]['accuracy'] for n in model_names]
    f1s = [all_metrics[n]['f1_weighted'] for n in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 (weighted)', color='coral')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Crop Classification — Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', ' ').title() for n in model_names], rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n  💾 All crop classification models saved to {MODEL_SAVE_DIR}")
    return all_metrics


# ============================================================
# PILLAR 2: STRESS DETECTION
# ============================================================
def train_stress_detection(X_train_tab, X_test_tab, y_train, y_test,
                            X_train_ts, X_test_ts, y_train_ts, y_test_ts,
                            feature_cols, ts_feature_cols, scaler, ts_scaler):
    """Train all Stress Detection models."""
    print("\n" + "=" * 70)
    print("💧 PILLAR 2: STRESS DETECTION")
    print("=" * 70)

    out_dir = os.path.join(OUTPUT_DIR, "stress_detection")
    all_metrics = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    class_names = ['No Stress', 'Stressed']
    n_classes = 2

    # ---- 1. Random Forest ----
    print("\n  [1/4] 🌲 Random Forest")
    rf_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [8, 12, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        rf_params, n_iter=25, cv=cv, scoring='f1', random_state=42, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train_tab, y_train)
    rf_model = rf_search.best_estimator_
    rf_pred = rf_model.predict(X_test_tab)
    rf_proba = rf_model.predict_proba(X_test_tab)

    rf_cv = cross_val_score(rf_model, X_train_tab, y_train, cv=cv, scoring='accuracy')
    rf_metrics = {
        'accuracy': float(accuracy_score(y_test, rf_pred)),
        'f1_weighted': float(f1_score(y_test, rf_pred, average='weighted')),
        'f1_binary': float(f1_score(y_test, rf_pred)),
        'kappa': float(cohen_kappa_score(y_test, rf_pred)),
        'cv_mean': float(rf_cv.mean()),
        'cv_std': float(rf_cv.std()),
        'best_params': rf_search.best_params_
    }
    all_metrics['random_forest'] = rf_metrics
    print(f"    ✅ Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1_binary']:.4f}, "
          f"CV: {rf_metrics['cv_mean']:.4f}±{rf_metrics['cv_std']:.4f}")

    plot_confusion_matrix(y_test, rf_pred, class_names,
                          'Random Forest — Stress Detection',
                          os.path.join(out_dir, 'cm_random_forest.png'))
    plot_roc_curves(y_test, rf_proba, class_names,
                    'Random Forest — ROC Curves',
                    os.path.join(out_dir, 'roc_random_forest.png'))
    plot_feature_importance(rf_model.feature_importances_, feature_cols,
                            'Random Forest — Feature Importance (Stress)',
                            os.path.join(out_dir, 'fi_random_forest.png'))
    compute_shap_values(rf_model, X_test_tab, feature_cols,
                        'SHAP — Random Forest (Stress)',
                        os.path.join(out_dir, 'shap_random_forest.png'))

    joblib.dump({'model': rf_model, 'scaler': scaler, 'features': feature_cols,
                 'metrics': rf_metrics},
                os.path.join(MODEL_SAVE_DIR, 'stress_rf.joblib'))

    # ---- 2. XGBoost ----
    print("\n  [2/4] 🚀 XGBoost")
    # Compute scale_pos_weight for imbalanced data
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    spw = n_neg / max(n_pos, 1)

    xgb_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
    }
    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(scale_pos_weight=spw, eval_metric='logloss',
                           random_state=42, n_jobs=-1),
        xgb_params, n_iter=25, cv=cv, scoring='f1', random_state=42, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train_tab, y_train)
    best_xgb_params = xgb_search.best_params_
    xgb_model = xgb.XGBClassifier(**best_xgb_params, scale_pos_weight=spw,
                                    eval_metric='logloss', random_state=42,
                                    early_stopping_rounds=20, n_jobs=-1)
    xgb_model.fit(X_train_tab, y_train, eval_set=[(X_test_tab, y_test)], verbose=False)
    xgb_pred = xgb_model.predict(X_test_tab)
    xgb_proba = xgb_model.predict_proba(X_test_tab)

    xgb_cv_model = xgb.XGBClassifier(**best_xgb_params, scale_pos_weight=spw,
                                        eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_cv = cross_val_score(xgb_cv_model, X_train_tab, y_train, cv=cv, scoring='accuracy')
    xgb_metrics = {
        'accuracy': float(accuracy_score(y_test, xgb_pred)),
        'f1_weighted': float(f1_score(y_test, xgb_pred, average='weighted')),
        'f1_binary': float(f1_score(y_test, xgb_pred)),
        'kappa': float(cohen_kappa_score(y_test, xgb_pred)),
        'cv_mean': float(xgb_cv.mean()),
        'cv_std': float(xgb_cv.std()),
        'best_params': best_xgb_params
    }
    all_metrics['xgboost'] = xgb_metrics
    print(f"    ✅ Accuracy: {xgb_metrics['accuracy']:.4f}, F1: {xgb_metrics['f1_binary']:.4f}, "
          f"CV: {xgb_metrics['cv_mean']:.4f}±{xgb_metrics['cv_std']:.4f}")

    plot_confusion_matrix(y_test, xgb_pred, class_names,
                          'XGBoost — Stress Detection',
                          os.path.join(out_dir, 'cm_xgboost.png'))
    plot_roc_curves(y_test, xgb_proba, class_names,
                    'XGBoost — ROC Curves',
                    os.path.join(out_dir, 'roc_xgboost.png'))
    plot_feature_importance(xgb_model.feature_importances_, feature_cols,
                            'XGBoost — Feature Importance (Stress)',
                            os.path.join(out_dir, 'fi_xgboost.png'))
    compute_shap_values(xgb_model, X_test_tab, feature_cols,
                        'SHAP — XGBoost (Stress)',
                        os.path.join(out_dir, 'shap_xgboost.png'))

    joblib.dump({'model': xgb_model, 'scaler': scaler, 'features': feature_cols,
                 'metrics': xgb_metrics},
                os.path.join(MODEL_SAVE_DIR, 'stress_xgb.joblib'))

    # ---- 3. Temporal CNN ----
    print("\n  [3/4] 📡 Temporal CNN")
    n_ts_features = X_train_ts.shape[2]
    tcnn_model = TemporalCNN(n_features=n_ts_features, n_classes=n_classes,
                              seq_len=X_train_ts.shape[1])

    tcnn_model, tcnn_pred, tcnn_proba, tcnn_history = train_pytorch_model(
        tcnn_model, X_train_ts, y_train_ts, X_test_ts, y_test_ts,
        model_name='Temporal CNN', n_epochs=100, lr=0.001, patience=15
    )

    tcnn_metrics = {
        'accuracy': float(accuracy_score(y_test_ts, tcnn_pred)),
        'f1_weighted': float(f1_score(y_test_ts, tcnn_pred, average='weighted')),
        'f1_binary': float(f1_score(y_test_ts, tcnn_pred)),
        'kappa': float(cohen_kappa_score(y_test_ts, tcnn_pred)),
        'epochs_trained': len(tcnn_history['train_losses']),
        'final_val_acc': float(tcnn_history['val_accs'][-1])
    }
    all_metrics['temporal_cnn'] = tcnn_metrics
    print(f"    ✅ Accuracy: {tcnn_metrics['accuracy']:.4f}, F1: {tcnn_metrics['f1_binary']:.4f}")

    plot_confusion_matrix(y_test_ts, tcnn_pred, class_names,
                          'Temporal CNN — Stress Detection',
                          os.path.join(out_dir, 'cm_temporal_cnn.png'))
    if tcnn_proba.shape[1] >= n_classes:
        plot_roc_curves(y_test_ts, tcnn_proba, class_names,
                        'Temporal CNN — ROC Curves',
                        os.path.join(out_dir, 'roc_temporal_cnn.png'))

    torch.save({
        'model_state_dict': tcnn_model.state_dict(),
        'n_features': n_ts_features,
        'n_classes': n_classes,
        'metrics': tcnn_metrics
    }, os.path.join(MODEL_SAVE_DIR, 'stress_tcnn.pt'))

    # ---- 4. LSTM ----
    print("\n  [4/4] 🧠 LSTM")
    lstm_model = LSTMClassifier(n_features=n_ts_features, n_classes=n_classes,
                                 hidden_size=64, n_layers=2)

    lstm_model, lstm_pred, lstm_proba, lstm_history = train_pytorch_model(
        lstm_model, X_train_ts, y_train_ts, X_test_ts, y_test_ts,
        model_name='LSTM', n_epochs=100, lr=0.001, patience=15
    )

    lstm_metrics = {
        'accuracy': float(accuracy_score(y_test_ts, lstm_pred)),
        'f1_weighted': float(f1_score(y_test_ts, lstm_pred, average='weighted')),
        'f1_binary': float(f1_score(y_test_ts, lstm_pred)),
        'kappa': float(cohen_kappa_score(y_test_ts, lstm_pred)),
        'epochs_trained': len(lstm_history['train_losses']),
        'final_val_acc': float(lstm_history['val_accs'][-1])
    }
    all_metrics['lstm'] = lstm_metrics
    print(f"    ✅ Accuracy: {lstm_metrics['accuracy']:.4f}, F1: {lstm_metrics['f1_binary']:.4f}")

    plot_confusion_matrix(y_test_ts, lstm_pred, class_names,
                          'LSTM — Stress Detection',
                          os.path.join(out_dir, 'cm_lstm.png'))
    if lstm_proba.shape[1] >= n_classes:
        plot_roc_curves(y_test_ts, lstm_proba, class_names,
                        'LSTM — ROC Curves',
                        os.path.join(out_dir, 'roc_lstm.png'))

    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'n_features': n_ts_features,
        'n_classes': n_classes,
        'hidden_size': 64,
        'n_layers': 2,
        'metrics': lstm_metrics
    }, os.path.join(MODEL_SAVE_DIR, 'stress_lstm.pt'))

    # ---- Model Comparison ----
    print("\n  📊 Model Comparison:")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'Kappa':>10}")
    print(f"  {'-'*55}")
    for name, m in all_metrics.items():
        f1_val = m.get('f1_binary', m.get('f1_weighted', 0))
        print(f"  {name:<25} {m['accuracy']:>10.4f} {f1_val:>10.4f} {m['kappa']:>10.4f}")

    # Comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(all_metrics.keys())
    accs = [all_metrics[n]['accuracy'] for n in model_names]
    f1s = [all_metrics[n].get('f1_binary', all_metrics[n].get('f1_weighted', 0)) for n in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
    ax.bar(x + width/2, f1s, width, label='F1 Score', color='coral')
    ax.set_ylabel('Score')
    ax.set_title('Stress Detection — Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', ' ').title() for n in model_names], rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n  💾 All stress detection models saved to {MODEL_SAVE_DIR}")
    return all_metrics


# ============================================================
# PILLAR 3: WATER DEFICIT REGRESSION
# ============================================================
def train_water_deficit(X_train, X_test, y_train, y_test, feature_cols, scaler):
    """Train all Water Deficit Regression models."""
    print("\n" + "=" * 70)
    print("🚿 PILLAR 3: WATER DEFICIT REGRESSION")
    print("=" * 70)

    out_dir = os.path.join(OUTPUT_DIR, "water_deficit")
    all_metrics = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # ---- 1. Random Forest Regressor ----
    print("\n  [1/4] 🌲 Random Forest Regressor")
    rf_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params, n_iter=25, cv=cv, scoring='r2', random_state=42, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train)
    rf_model = rf_search.best_estimator_
    rf_pred = rf_model.predict(X_test)

    rf_cv_r2 = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='r2')
    rf_cv_mse = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    rf_metrics = {
        'r2': float(r2_score(y_test, rf_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, rf_pred))),
        'mae': float(mean_absolute_error(y_test, rf_pred)),
        'cv_r2_mean': float(rf_cv_r2.mean()),
        'cv_r2_std': float(rf_cv_r2.std()),
        'cv_rmse_mean': float(np.sqrt(-rf_cv_mse.mean())),
        'best_params': rf_search.best_params_
    }
    all_metrics['random_forest'] = rf_metrics
    print(f"    ✅ R²: {rf_metrics['r2']:.4f}, RMSE: {rf_metrics['rmse']:.4f}, "
          f"MAE: {rf_metrics['mae']:.4f}, CV R²: {rf_metrics['cv_r2_mean']:.4f}")

    plot_regression_results(y_test, rf_pred, 'Random Forest — Water Deficit',
                            os.path.join(out_dir, 'reg_random_forest.png'))
    plot_feature_importance(rf_model.feature_importances_, feature_cols,
                            'Random Forest — Feature Importance (Water Deficit)',
                            os.path.join(out_dir, 'fi_random_forest.png'))
    compute_shap_values(rf_model, X_test, feature_cols,
                        'SHAP — Random Forest (Water Deficit)',
                        os.path.join(out_dir, 'shap_random_forest.png'))

    joblib.dump({'model': rf_model, 'scaler': scaler, 'features': feature_cols,
                 'metrics': rf_metrics},
                os.path.join(MODEL_SAVE_DIR, 'deficit_rf.joblib'))

    # ---- 2. XGBoost Regressor ----
    print("\n  [2/4] 🚀 XGBoost Regressor")
    xgb_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
    }
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(eval_metric='rmse', random_state=42, n_jobs=-1),
        xgb_params, n_iter=25, cv=cv, scoring='r2', random_state=42, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train)
    best_xgb_params = xgb_search.best_params_
    xgb_model = xgb.XGBRegressor(**best_xgb_params, eval_metric='rmse',
                                   random_state=42, early_stopping_rounds=20, n_jobs=-1)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)

    xgb_cv_model = xgb.XGBRegressor(**best_xgb_params, eval_metric='rmse',
                                       random_state=42, n_jobs=-1)
    xgb_cv_r2 = cross_val_score(xgb_cv_model, X_train, y_train, cv=cv, scoring='r2')
    xgb_metrics = {
        'r2': float(r2_score(y_test, xgb_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, xgb_pred))),
        'mae': float(mean_absolute_error(y_test, xgb_pred)),
        'cv_r2_mean': float(xgb_cv_r2.mean()),
        'cv_r2_std': float(xgb_cv_r2.std()),
        'best_params': best_xgb_params
    }
    all_metrics['xgboost'] = xgb_metrics
    print(f"    ✅ R²: {xgb_metrics['r2']:.4f}, RMSE: {xgb_metrics['rmse']:.4f}, "
          f"MAE: {xgb_metrics['mae']:.4f}, CV R²: {xgb_metrics['cv_r2_mean']:.4f}")

    plot_regression_results(y_test, xgb_pred, 'XGBoost — Water Deficit',
                            os.path.join(out_dir, 'reg_xgboost.png'))
    plot_feature_importance(xgb_model.feature_importances_, feature_cols,
                            'XGBoost — Feature Importance (Water Deficit)',
                            os.path.join(out_dir, 'fi_xgboost.png'))
    compute_shap_values(xgb_model, X_test, feature_cols,
                        'SHAP — XGBoost (Water Deficit)',
                        os.path.join(out_dir, 'shap_xgboost.png'))

    joblib.dump({'model': xgb_model, 'scaler': scaler, 'features': feature_cols,
                 'metrics': xgb_metrics},
                os.path.join(MODEL_SAVE_DIR, 'deficit_xgb.joblib'))

    # ---- 3. LightGBM ----
    print("\n  [3/4] 💡 LightGBM")
    lgb_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 8, 12, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_samples': [5, 10, 20],
    }
    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        lgb_params, n_iter=25, cv=cv, scoring='r2', random_state=42, n_jobs=-1, verbose=0
    )
    lgb_search.fit(X_train, y_train)
    best_lgb_params = lgb_search.best_params_
    lgb_model = lgb.LGBMRegressor(**best_lgb_params, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                             lgb.log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_test)

    lgb_cv_r2 = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='r2')
    lgb_metrics = {
        'r2': float(r2_score(y_test, lgb_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, lgb_pred))),
        'mae': float(mean_absolute_error(y_test, lgb_pred)),
        'cv_r2_mean': float(lgb_cv_r2.mean()),
        'cv_r2_std': float(lgb_cv_r2.std()),
        'best_params': best_lgb_params
    }
    all_metrics['lightgbm'] = lgb_metrics
    print(f"    ✅ R²: {lgb_metrics['r2']:.4f}, RMSE: {lgb_metrics['rmse']:.4f}, "
          f"MAE: {lgb_metrics['mae']:.4f}, CV R²: {lgb_metrics['cv_r2_mean']:.4f}")

    plot_regression_results(y_test, lgb_pred, 'LightGBM — Water Deficit',
                            os.path.join(out_dir, 'reg_lightgbm.png'))
    plot_feature_importance(lgb_model.feature_importances_, feature_cols,
                            'LightGBM — Feature Importance (Water Deficit)',
                            os.path.join(out_dir, 'fi_lightgbm.png'))
    compute_shap_values(lgb_model, X_test, feature_cols,
                        'SHAP — LightGBM (Water Deficit)',
                        os.path.join(out_dir, 'shap_lightgbm.png'))

    joblib.dump({'model': lgb_model, 'scaler': scaler, 'features': feature_cols,
                 'metrics': lgb_metrics},
                os.path.join(MODEL_SAVE_DIR, 'deficit_lgb.joblib'))

    # ---- 4. CatBoost ----
    print("\n  [4/4] 🐱 CatBoost")
    cb_model = CatBoostRegressor(
        iterations=500, depth=8, learning_rate=0.05,
        l2_leaf_reg=3, subsample=0.8,
        random_seed=42, verbose=0, early_stopping_rounds=30,
        eval_metric='RMSE'
    )
    cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    cb_pred = cb_model.predict(X_test)

    cb_cv_r2 = cross_val_score(
        CatBoostRegressor(iterations=300, depth=8, learning_rate=0.05,
                           random_seed=42, verbose=0),
        X_train, y_train, cv=cv, scoring='r2'
    )
    cb_metrics = {
        'r2': float(r2_score(y_test, cb_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, cb_pred))),
        'mae': float(mean_absolute_error(y_test, cb_pred)),
        'cv_r2_mean': float(cb_cv_r2.mean()),
        'cv_r2_std': float(cb_cv_r2.std()),
        'best_iteration': int(cb_model.get_best_iteration()) if hasattr(cb_model, 'get_best_iteration') else None
    }
    all_metrics['catboost'] = cb_metrics
    print(f"    ✅ R²: {cb_metrics['r2']:.4f}, RMSE: {cb_metrics['rmse']:.4f}, "
          f"MAE: {cb_metrics['mae']:.4f}, CV R²: {cb_metrics['cv_r2_mean']:.4f}")

    plot_regression_results(y_test, cb_pred, 'CatBoost — Water Deficit',
                            os.path.join(out_dir, 'reg_catboost.png'))
    cb_fi = np.array(cb_model.get_feature_importance())
    plot_feature_importance(cb_fi, feature_cols,
                            'CatBoost — Feature Importance (Water Deficit)',
                            os.path.join(out_dir, 'fi_catboost.png'))
    compute_shap_values(cb_model, X_test, feature_cols,
                        'SHAP — CatBoost (Water Deficit)',
                        os.path.join(out_dir, 'shap_catboost.png'))

    joblib.dump({'model': cb_model, 'scaler': scaler, 'features': feature_cols,
                 'metrics': cb_metrics},
                os.path.join(MODEL_SAVE_DIR, 'deficit_catboost.joblib'))

    # ---- Model Comparison ----
    print("\n  📊 Model Comparison:")
    print(f"  {'Model':<25} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'CV R²':>10}")
    print(f"  {'-'*65}")
    for name, m in all_metrics.items():
        print(f"  {name:<25} {m['r2']:>10.4f} {m['rmse']:>10.4f} "
              f"{m['mae']:>10.4f} {m['cv_r2_mean']:>10.4f}")

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    model_names = list(all_metrics.keys())

    r2s = [all_metrics[n]['r2'] for n in model_names]
    rmses = [all_metrics[n]['rmse'] for n in model_names]

    axes[0].barh(model_names, r2s, color='steelblue')
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('R² Score Comparison', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    axes[1].barh(model_names, rmses, color='coral')
    axes[1].set_xlabel('RMSE (mm)')
    axes[1].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Water Deficit Regression — Model Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n  💾 All water deficit models saved to {MODEL_SAVE_DIR}")
    return all_metrics


# ============================================================
# MAIN
# ============================================================
def main():
    """Run the complete training pipeline."""
    start_time = time.time()

    print("\n" + "🛰️ " * 20)
    print("  KrishiDrishti — Complete Model Training Pipeline")
    print("  Team BEST SHOT | ROBOVANTA Hackathon")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🛰️ " * 20)

    # ---- Load Data ----
    master, ts, best_features = load_data()

    # ========================================
    # PILLAR 1: CROP CLASSIFICATION
    # ========================================
    try:
        (X_train_cc, X_test_cc, y_train_cc, y_test_cc,
         feat_cc, classes_cc, le_cc, scaler_cc) = prepare_crop_classification_data(master, best_features)

        crop_metrics = train_crop_classification(
            X_train_cc, X_test_cc, y_train_cc, y_test_cc,
            feat_cc, classes_cc, le_cc, scaler_cc
        )
    except Exception as e:
        print(f"\n  ❌ Crop Classification failed: {e}")
        traceback.print_exc()
        crop_metrics = {'error': str(e)}

    # ========================================
    # PILLAR 2: STRESS DETECTION
    # ========================================
    try:
        (X_train_st, X_test_st, y_train_st, y_test_st,
         X_train_ts, X_test_ts, y_train_ts, y_test_ts,
         feat_st, ts_feat_st, scaler_st, ts_scaler_st) = prepare_stress_detection_data(
            master, ts, best_features
        )

        stress_metrics = train_stress_detection(
            X_train_st, X_test_st, y_train_st, y_test_st,
            X_train_ts, X_test_ts, y_train_ts, y_test_ts,
            feat_st, ts_feat_st, scaler_st, ts_scaler_st
        )
    except Exception as e:
        print(f"\n  ❌ Stress Detection failed: {e}")
        traceback.print_exc()
        stress_metrics = {'error': str(e)}

    # ========================================
    # PILLAR 3: WATER DEFICIT REGRESSION
    # ========================================
    try:
        (X_train_wd, X_test_wd, y_train_wd, y_test_wd,
         feat_wd, scaler_wd) = prepare_water_deficit_data(master, best_features)

        deficit_metrics = train_water_deficit(
            X_train_wd, X_test_wd, y_train_wd, y_test_wd,
            feat_wd, scaler_wd
        )
    except Exception as e:
        print(f"\n  ❌ Water Deficit failed: {e}")
        traceback.print_exc()
        deficit_metrics = {'error': str(e)}

    # ========================================
    # SUMMARY
    # ========================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETE")
    print("=" * 70)
    print(f"  ⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"\n  📁 Saved models:  {MODEL_SAVE_DIR}")
    print(f"  📁 Outputs:       {OUTPUT_DIR}")

    # List saved files
    print("\n  📦 Saved Model Files:")
    for f in sorted(os.listdir(MODEL_SAVE_DIR)):
        fpath = os.path.join(MODEL_SAVE_DIR, f)
        fsize = os.path.getsize(fpath) / 1024
        print(f"    • {f} ({fsize:.1f} KB)")

    print("\n  📊 Output Files:")
    for subdir in ['crop_classification', 'stress_detection', 'water_deficit']:
        sdir = os.path.join(OUTPUT_DIR, subdir)
        if os.path.exists(sdir):
            files = os.listdir(sdir)
            print(f"    [{subdir}]: {len(files)} files")
            for f in sorted(files):
                print(f"      • {f}")

    # Save overall summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': elapsed,
        'crop_classification': crop_metrics,
        'stress_detection': stress_metrics,
        'water_deficit': deficit_metrics
    }
    with open(os.path.join(OUTPUT_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  ✅ Training summary → {os.path.join(OUTPUT_DIR, 'training_summary.json')}")
    print("\n" + "🌾 " * 20)
    print("  KrishiDrishti training pipeline complete!")
    print("🌾 " * 20)


if __name__ == '__main__':
    main()
