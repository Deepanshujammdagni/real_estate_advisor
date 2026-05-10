"""
Real Estate Investment Advisor
Step 3: Model Training
- Classification: Good Investment (XGBoost + Random Forest)
- Regression: Price After 5 Years (XGBoost + Random Forest)
- MLflow experiment tracking
- Model evaluation & feature importance
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not installed — using RandomForest only")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not installed — skipping experiment tracking")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR  = "models"
REPORT_DIR = "model_reports"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CLASSIFICATION_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Parking_Space", "Floor_No", "Total_Floors",
    "School_Density_Score", "Hospital_Proximity_Score",
    "Appreciation_Rate", "Amenity_Count", "Transport_Score",
    "Property_Type_enc", "City_enc", "State_enc",
    "Furnished_Status_enc", "Owner_Type_enc",
    "Availability_Status_enc", "Security_enc",
]

REGRESSION_FEATURES = CLASSIFICATION_FEATURES.copy()
REGRESSION_FEATURES.remove("Price_in_Lakhs")  # avoid target leakage in regression

TARGET_CLASS = "Good_Investment"
TARGET_REG   = "Price_After_5_Years"


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def _available_features(df, feature_list):
    """Return only features that exist AND are numeric (or can be coerced to numeric)."""
    result = []
    for f in feature_list:
        if f not in df.columns:
            continue
        # Accept if already numeric dtype
        if pd.api.types.is_numeric_dtype(df[f]):
            result.append(f)
        else:
            # Try coercion — if most values survive, include it
            coerced = pd.to_numeric(df[f], errors="coerce")
            if coerced.notna().mean() >= 0.5:
                result.append(f)
    return result


def _safe_numeric(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Force-cast all feature columns to float, coercing strings to NaN then 0."""
    X = df[features].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna(0).astype(float)


def _save_fig(fig, name):
    path = os.path.join(REPORT_DIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, classes, title, name):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="#0f1117")
    ax.set_title(title, color="white")
    ax.set_xlabel("Predicted", color="#a0a8c0")
    ax.set_ylabel("Actual",    color="#a0a8c0")
    ax.tick_params(colors="#a0a8c0")
    return _save_fig(fig, name)


def plot_feature_importance(model, feature_names, title, name, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.barh(imp.index, imp.values, color="#6c63ff", alpha=0.85)
    ax.set_title(title, color="white", fontsize=12)
    ax.set_xlabel("Feature Importance", color="#a0a8c0")
    ax.tick_params(colors="#a0a8c0")
    return _save_fig(fig, name)


def plot_regression_actual_vs_predicted(y_true, y_pred, title, name):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#6c63ff")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect Fit")
    ax.set_xlabel("Actual (₹ Lakhs)", color="#a0a8c0")
    ax.set_ylabel("Predicted (₹ Lakhs)", color="#a0a8c0")
    ax.set_title(title, color="white")
    ax.tick_params(colors="#a0a8c0")
    ax.legend()
    return _save_fig(fig, name)


# ─────────────────────────────────────────────
# CLASSIFICATION TRAINING
# ─────────────────────────────────────────────
def train_classification(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("  CLASSIFICATION — Good Investment Prediction")
    print("=" * 55)

    feats = _available_features(df, CLASSIFICATION_FEATURES)
    print(f"  ✅ Using {len(feats)} numeric features: {feats}")
    X = _safe_numeric(df, feats)
    y = df[TARGET_CLASS].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Random Forest ─────────────────────────────────
    with _mlflow_run("classification_random_forest"):
        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_train_sc, y_train)
        y_pred_rf = rf.predict(X_test_sc)
        y_prob_rf = rf.predict_proba(X_test_sc)[:, 1]

        acc  = accuracy_score(y_test, y_pred_rf)
        auc  = roc_auc_score(y_test, y_prob_rf)
        report = classification_report(y_test, y_pred_rf)
        print(f"\n[RandomForest Classifier]\nAccuracy: {acc:.4f} | AUC: {auc:.4f}")
        print(report)

        _mlflow_log({"accuracy": acc, "auc": auc}, {"n_estimators": 200, "max_depth": 10})
        cm_path = plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf),
                                        ["Not Good", "Good"], "RF Confusion Matrix", "rf_clf_cm")
        fi_path = plot_feature_importance(rf, feats,
                                          "RF — Feature Importance (Classification)", "rf_clf_fi")
        results["RandomForest"] = {"model": rf, "scaler": scaler, "accuracy": acc, "auc": auc,
                                   "features": feats, "report": report}

    # ── XGBoost ───────────────────────────────────────
    if XGB_AVAILABLE:
        with _mlflow_run("classification_xgboost"):
            xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 use_label_encoder=False, eval_metric="logloss",
                                 random_state=42, n_jobs=-1)
            xgb.fit(X_train_sc, y_train)
            y_pred_xgb = xgb.predict(X_test_sc)
            y_prob_xgb = xgb.predict_proba(X_test_sc)[:, 1]

            acc_x = accuracy_score(y_test, y_pred_xgb)
            auc_x = roc_auc_score(y_test, y_prob_xgb)
            report_x = classification_report(y_test, y_pred_xgb)
            print(f"\n[XGBoost Classifier]\nAccuracy: {acc_x:.4f} | AUC: {auc_x:.4f}")
            print(report_x)

            _mlflow_log({"accuracy": acc_x, "auc": auc_x}, {"n_estimators": 300, "max_depth": 6})
            plot_confusion_matrix(confusion_matrix(y_test, y_pred_xgb),
                                  ["Not Good", "Good"], "XGB Confusion Matrix", "xgb_clf_cm")
            plot_feature_importance(xgb, feats,
                                    "XGB — Feature Importance (Classification)", "xgb_clf_fi")
            results["XGBoost"] = {"model": xgb, "scaler": scaler, "accuracy": acc_x, "auc": auc_x,
                                   "features": feats, "report": report_x}

    # ── Pick best & save ──────────────────────────────
    best_key = max(results, key=lambda k: results[k]["auc"])
    best = results[best_key]
    print(f"\n🏆 Best Classification Model: {best_key} (AUC={best['auc']:.4f})")

    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump({"model": best["model"], "scaler": scaler,
                     "features": feats, "model_name": best_key}, f)

    print(f"💾 Classifier saved → {MODEL_DIR}/classifier.pkl")
    return results


# ─────────────────────────────────────────────
# REGRESSION TRAINING
# ─────────────────────────────────────────────
def train_regression(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("  REGRESSION — Price After 5 Years Prediction")
    print("=" * 55)

    feats = _available_features(df, REGRESSION_FEATURES)
    print(f"  ✅ Using {len(feats)} numeric features: {feats}")
    X = _safe_numeric(df, feats)
    y = df[TARGET_REG]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    def _reg_metrics(y_true, y_pred, label):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        print(f"\n[{label}]\nRMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
        return rmse, mae, r2

    # ── Random Forest ─────────────────────────────────
    with _mlflow_run("regression_random_forest"):
        rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                   random_state=42, n_jobs=-1)
        rf.fit(X_train_sc, y_train)
        y_pred_rf = rf.predict(X_test_sc)
        rmse, mae, r2 = _reg_metrics(y_test, y_pred_rf, "RandomForest Regressor")
        _mlflow_log({"rmse": rmse, "mae": mae, "r2": r2}, {"n_estimators": 200})
        plot_feature_importance(rf, feats, "RF — Feature Importance (Regression)", "rf_reg_fi")
        plot_regression_actual_vs_predicted(y_test, y_pred_rf,
                                            "RF — Actual vs Predicted", "rf_reg_scatter")
        results["RandomForest"] = {"model": rf, "rmse": rmse, "mae": mae, "r2": r2, "features": feats}

    # ── XGBoost ───────────────────────────────────────
    if XGB_AVAILABLE:
        with _mlflow_run("regression_xgboost"):
            xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                               random_state=42, n_jobs=-1)
            xgb.fit(X_train_sc, y_train)
            y_pred_xgb = xgb.predict(X_test_sc)
            rmse_x, mae_x, r2_x = _reg_metrics(y_test, y_pred_xgb, "XGBoost Regressor")
            _mlflow_log({"rmse": rmse_x, "mae": mae_x, "r2": r2_x}, {"n_estimators": 300})
            plot_feature_importance(xgb, feats, "XGB — Feature Importance (Regression)", "xgb_reg_fi")
            plot_regression_actual_vs_predicted(y_test, y_pred_xgb,
                                                "XGB — Actual vs Predicted", "xgb_reg_scatter")
            results["XGBoost"] = {"model": xgb, "rmse": rmse_x, "mae": mae_x, "r2": r2_x, "features": feats}

    # ── Pick best & save ──────────────────────────────
    best_key = min(results, key=lambda k: results[k]["rmse"])
    best = results[best_key]
    print(f"\n🏆 Best Regression Model: {best_key} (RMSE={best['rmse']:.2f})")

    with open(os.path.join(MODEL_DIR, "regressor.pkl"), "wb") as f:
        pickle.dump({"model": best["model"], "scaler": scaler,
                     "features": feats, "model_name": best_key}, f)

    print(f"💾 Regressor saved → {MODEL_DIR}/regressor.pkl")
    return results


# ─────────────────────────────────────────────
# MLFLOW HELPERS
# ─────────────────────────────────────────────
from contextlib import contextmanager

@contextmanager
def _mlflow_run(run_name):
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("RealEstate_Advisor")
        with mlflow.start_run(run_name=run_name):
            yield
    else:
        yield


def _mlflow_log(metrics: dict, params: dict):
    if not MLFLOW_AVAILABLE:
        return
    for k, v in params.items():
        mlflow.log_param(k, v)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed_housing.csv"

    print("Loading processed data …")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")

    clf_results = train_classification(df)
    reg_results = train_regression(df)

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55)
    print(f"Models saved in → ./{MODEL_DIR}/")
    print(f"Reports saved in → ./{REPORT_DIR}/")