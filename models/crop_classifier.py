"""
Crop Type Classifier — RF + XGBoost Ensemble
=============================================
Module for training and evaluating crop type classification using
multi-temporal satellite features. Uses Random Forest + XGBoost 
ensemble with soft voting for robust predictions.

Features used:
    - Multi-temporal vegetation indices (NDVI, EVI, NDWI, NDMI, SAVI)
    - SAR backscatter features (VV, VH, VH/VV ratio, RVI)
    - Phenological metrics (SOS, Peak, EOS, Season Length)
    - GLCM texture features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    cohen_kappa_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import os
import json

# Optional: SHAP for explainability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP not installed. Install with `pip install shap` for explainability.")

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CLASSIFICATION_PARAMS, CROP_CLASSES, MODEL_DIR, OUTPUT_DIR


class CropClassifier:
    """
    Ensemble crop type classifier using Random Forest + XGBoost.
    
    Workflow:
        1. Load multi-temporal feature data (from GEE export)
        2. Train RF and XGBoost independently
        3. Combine via soft voting ensemble
        4. Validate with confusion matrix, Kappa, and per-class F1
        5. Generate SHAP explainability plots
    """
    
    def __init__(self, config=None):
        """
        Initialize the classifier with hyperparameters.
        
        Args:
            config: dict — Override default params from config.py
        """
        self.config = config or CLASSIFICATION_PARAMS
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize models
        self.rf_model = RandomForestClassifier(**self.config['random_forest'])
        self.xgb_model = xgb.XGBClassifier(**self.config['xgboost'])
        
        # Ensemble (soft voting = uses predicted probabilities)
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('xgb', self.xgb_model)
            ],
            voting='soft',
            weights=[
                self.config['ensemble_weights']['rf'],
                self.config['ensemble_weights']['xgb']
            ]
        )
        
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        self.results = {}
    
    def prepare_data(self, df, label_column='label', drop_columns=None):
        """
        Prepare training data from a DataFrame.
        
        Args:
            df: pd.DataFrame — Feature data with labels
            label_column: str — Column name for crop labels
            drop_columns: list — Columns to exclude from features
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Drop non-feature columns
        exclude = [label_column, 'geometry', 'system:index', '.geo', 'longitude', 'latitude']
        if drop_columns:
            exclude.extend(drop_columns)
        
        feature_cols = [c for c in df.columns if c not in exclude]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[label_column].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.class_names = list(self.label_encoder.classes_)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=self.config['test_size'],
            stratify=y_encoded,
            random_state=42
        )
        
        print(f"✅ Data prepared:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Classes: {self.class_names}")
        print(f"   Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the ensemble classifier.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Optional test data for immediate evaluation
        """
        print("\n" + "=" * 60)
        print("Training Crop Type Classifier")
        print("=" * 60)
        
        # ---- Train Random Forest ----
        print("\n[1/3] Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        rf_train_acc = self.rf_model.score(X_train, y_train)
        print(f"   RF Training Accuracy: {rf_train_acc:.4f}")
        
        if X_test is not None:
            rf_test_acc = self.rf_model.score(X_test, y_test)
            print(f"   RF Test Accuracy: {rf_test_acc:.4f}")
        
        # ---- Train XGBoost ----
        print("\n[2/3] Training XGBoost...")
        self.xgb_model.fit(X_train, y_train)
        xgb_train_acc = self.xgb_model.score(X_train, y_train)
        print(f"   XGB Training Accuracy: {xgb_train_acc:.4f}")
        
        if X_test is not None:
            xgb_test_acc = self.xgb_model.score(X_test, y_test)
            print(f"   XGB Test Accuracy: {xgb_test_acc:.4f}")
        
        # ---- Train Ensemble ----
        print("\n[3/3] Training Ensemble (Soft Voting)...")
        self.ensemble.fit(X_train, y_train)
        ens_train_acc = self.ensemble.score(X_train, y_train)
        print(f"   Ensemble Training Accuracy: {ens_train_acc:.4f}")
        
        if X_test is not None:
            ens_test_acc = self.ensemble.score(X_test, y_test)
            print(f"   Ensemble Test Accuracy: {ens_test_acc:.4f}")
        
        self.is_trained = True
        
        # ---- Cross-Validation ----
        print("\n[CV] Running 5-fold cross-validation on ensemble...")
        cv_scores = cross_val_score(
            self.ensemble, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42),
            scoring='accuracy'
        )
        print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.results['cv_scores'] = cv_scores.tolist()
        
        # Evaluate on test set
        if X_test is not None:
            self.evaluate(X_test, y_test)
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive evaluation with confusion matrix, Kappa, and per-class metrics.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        y_pred = self.ensemble.predict(X_test)
        
        # Overall Accuracy
        oa = accuracy_score(y_test, y_pred)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # F1 Scores
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Classification Report
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"\n🎯 Overall Accuracy (OA): {oa:.4f} ({oa*100:.1f}%)")
        print(f"📊 Cohen's Kappa:         {kappa:.4f}")
        print(f"📈 F1 Score (macro):       {f1_macro:.4f}")
        print(f"📈 F1 Score (weighted):    {f1_weighted:.4f}")
        
        target_met = "✅ TARGET MET" if oa >= self.config['accuracy_target'] else "❌ Below target"
        print(f"\n{target_met} (Target: {self.config['accuracy_target']*100}%)")
        
        print(f"\n{'='*60}")
        print("Per-Class Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Store results
        self.results.update({
            'overall_accuracy': oa,
            'kappa': kappa,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        })
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)
        
        return self.results
    
    def _plot_confusion_matrix(self, y_test, y_pred):
        """Generate and save a beautiful confusion matrix plot."""
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names, yticklabels=self.class_names,
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Normalized
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=self.class_names, yticklabels=self.class_names,
            ax=axes[1]
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.suptitle('KrishiDrishti — Crop Classification Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance from Random Forest."""
        if not self.is_trained or self.feature_names is None:
            raise ValueError("Model not trained or feature names not set!")
        
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
        bars = ax.barh(
            range(top_n),
            importances[indices][::-1],
            color=colors
        )
        
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([self.feature_names[i] for i in indices[::-1]])
        ax.set_xlabel('Feature Importance (Gini)')
        ax.set_title(f'Top {top_n} Features — Random Forest', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Feature importance saved to: {save_path}")
    
    def explain_with_shap(self, X_test, max_samples=500):
        """
        Generate SHAP explanations for model predictions.
        This is a KEY differentiator — Explainable AI.
        """
        if not HAS_SHAP:
            print("⚠️ SHAP not available. Skipping explainability.")
            return
        
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        print("\n🧠 Generating SHAP explanations...")
        
        # Use a subset for speed
        if len(X_test) > max_samples:
            idx = np.random.choice(len(X_test), max_samples, replace=False)
            X_sample = X_test[idx]
        else:
            X_sample = X_test
        
        # SHAP for XGBoost (faster than RF)
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=15,
            show=False
        )
        plt.title('SHAP Feature Importance — XGBoost', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, 'shap_importance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"🧠 SHAP plot saved to: {save_path}")
    
    def predict(self, X):
        """Predict crop types for new data."""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.ensemble.predict(X_scaled)
        labels = self.label_encoder.inverse_transform(predictions)
        
        return labels
    
    def predict_proba(self, X):
        """Get prediction probabilities (confidence scores)."""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict_proba(X_scaled)
    
    def save_model(self, name='crop_classifier'):
        """Save trained model to disk."""
        save_path = os.path.join(MODEL_DIR, f'{name}.joblib')
        model_data = {
            'ensemble': self.ensemble,
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'results': self.results,
            'config': self.config
        }
        joblib.dump(model_data, save_path)
        print(f"💾 Model saved to: {save_path}")
    
    def load_model(self, name='crop_classifier'):
        """Load a trained model from disk."""
        load_path = os.path.join(MODEL_DIR, f'{name}.joblib')
        model_data = joblib.load(load_path)
        
        self.ensemble = model_data['ensemble']
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data['xgb_model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        self.results = model_data['results']
        self.is_trained = True
        
        print(f"📂 Model loaded from: {load_path}")
        print(f"   Classes: {self.class_names}")
        print(f"   OA: {self.results.get('overall_accuracy', 'N/A')}")


# ============================================================
# DEMO / USAGE EXAMPLE
# ============================================================
def demo_with_synthetic_data():
    """
    Demonstrate the classifier with synthetic data.
    Replace with actual GEE-exported data in the hackathon.
    """
    print("🌾 KrishiDrishti — Crop Classifier Demo (Synthetic Data)")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 2000
    n_features = 40  # ~15 dates × ~3 indices per composite
    n_classes = 5
    
    # Generate synthetic multi-temporal features
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Add class-specific patterns (simulate real spectral differences)
    for cls in range(n_classes):
        mask = y == cls
        X[mask] += np.random.randn(n_features) * 0.5
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    crop_names = ['Rice', 'Cotton', 'Maize', 'Soybean', 'Sugarcane']
    labels = [crop_names[i] for i in y]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = labels
    
    # Initialize and train
    classifier = CropClassifier()
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    classifier.train(X_train, y_train, X_test, y_test)
    
    # Feature importance
    classifier.plot_feature_importance()
    
    # SHAP (if available)
    classifier.explain_with_shap(X_test)
    
    # Save model
    classifier.save_model()
    
    return classifier


if __name__ == '__main__':
    classifier = demo_with_synthetic_data()
