"""
classification.py
-----------------
Cluster-based churn (Will_Return) classification pipeline.

Steps per clustering algorithm:
  1. Merge cluster labels into the prediction DataFrame.
  2. For each cluster segment, train/test split.
  3. (Optional) Apply SMOTE+Tomek to balance the training set.
  4. Train Logistic Regression, Random Forest, Gradient Boosting, XGBoost.
  5. Collect per-segment metrics (Accuracy, Precision, Recall, F1, ROC-AUC).

Also provides:
  - Global baseline (no clustering)
  - Confusion matrix visualisation for the best pipeline
  - Comparison barplots across all approaches
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    print("xgboost not installed – XGBoost will be skipped.")


# ── Model registry ────────────────────────────────────────────────────────────

def get_classification_models(random_state: int = 42) -> dict:
    """Return a fresh dictionary of unfitted classification models."""
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=random_state, solver="liblinear"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }
    if _XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, learning_rate=0.1,
            random_state=random_state, eval_metric="logloss"
        )
    return models


# ── SMOTE+Tomek helper ────────────────────────────────────────────────────────

def _smote_tomek_resample(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE+Tomek to balance the training set.
    k_neighbors is adjusted automatically for small minority classes.
    Returns (X_resampled, y_resampled).
    Falls back to original data if minority class is too small.
    """
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn not installed – run: pip install imbalanced-learn")

    min_minority = y_train.value_counts().min()
    if min_minority > 1:
        k_nb = min(5, min_minority - 1)
        smote  = SMOTE(k_neighbors=k_nb, random_state=random_state)
        st     = SMOTETomek(smote=smote, random_state=random_state)
        return st.fit_resample(X_train, y_train)
    # fallback – not enough minority samples
    return X_train, y_train


# ── Single-segment evaluation ─────────────────────────────────────────────────

def _evaluate_segment(X_train, X_test, y_train, y_test,
                       models: dict,
                       use_smote: bool = False,
                       random_state: int = 42) -> list:
    """
    Train each model and return a list of metric dicts.
    """
    if use_smote:
        X_train, y_train = _smote_tomek_resample(X_train, y_train, random_state)

    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = (roc_auc_score(y_test, y_prob)
               if len(np.unique(y_test)) == 2 else np.nan)
        results.append({
            "Classification_Algorithm": model_name,
            "Accuracy" : accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall"   : recall_score(y_test, y_pred, zero_division=0),
            "F1_Score" : f1_score(y_test, y_pred, zero_division=0),
            "ROC_AUC"  : roc,
        })
    return results


# ── Cluster-based classification ──────────────────────────────────────────────

ALGO_CLUSTER_COLUMNS = {
    "K-Means"            : "KMeans_Cluster",
    "Fuzzy C-Means"      : "Fuzzy_Cluster",
    "Spectral Clustering": "Spectral_Cluster",
}

FEATURES_CLF = ["Recency", "Frequency", "Monetary_Past"]


def run_cluster_classification(rfmt_cleaned: pd.DataFrame,
                                predict_df: pd.DataFrame,
                                algo_cluster_cols: dict = None,
                                use_smote: bool = False,
                                test_size: float = 0.2,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Run cluster-based churn classification for multiple clustering algorithms.

    Parameters
    ----------
    rfmt_cleaned : pd.DataFrame
        RFMT DataFrame with cluster label columns.
    predict_df : pd.DataFrame
        Customer-level DataFrame with features + 'Will_Return' target.
        Must contain: 'Customer ID', 'Recency', 'Frequency', 'Monetary_Past', 'Will_Return'.
    algo_cluster_cols : dict, optional
        Mapping {algo_name: cluster_col_name}.  Defaults to ALGO_CLUSTER_COLUMNS.
    use_smote : bool
        Whether to apply SMOTE+Tomek to balance each segment's training set.
    test_size : float
    random_state : int

    Returns
    -------
    pd.DataFrame
        Per-segment evaluation metrics.
    """
    if algo_cluster_cols is None:
        algo_cluster_cols = ALGO_CLUSTER_COLUMNS

    all_results = []
    models = get_classification_models(random_state)

    for algo_name, cluster_col in algo_cluster_cols.items():
        if cluster_col not in rfmt_cleaned.columns:
            print(f"[SKIP] Column '{cluster_col}' not found.")
            continue
        print(f"\n── {algo_name} ({'SMOTE' if use_smote else 'no SMOTE'}) ──")

        temp_df = predict_df.merge(
            rfmt_cleaned[["Customer ID", cluster_col]],
            on="Customer ID", how="left"
        )
        temp_df["Segment"] = temp_df[cluster_col].astype(str)

        for seg in sorted(temp_df["Segment"].unique()):
            seg_df = temp_df[temp_df["Segment"] == seg]
            X = seg_df[FEATURES_CLF]
            y = seg_df["Will_Return"]

            # Single-class segment – record and skip
            if len(y.unique()) < 2:
                all_results.append({
                    "Clustering_Algorithm"  : algo_name,
                    "Segment_ID"            : seg,
                    "Classification_Algorithm": "N/A (Single Class)",
                    "Accuracy" : 1.0 if len(y) > 0 else 0,
                    "Precision": np.nan, "Recall": np.nan,
                    "F1_Score" : np.nan, "ROC_AUC": np.nan,
                })
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=random_state, stratify=y
            )
            seg_results = _evaluate_segment(
                X_train, X_test, y_train, y_test,
                models, use_smote, random_state
            )
            for r in seg_results:
                r.update({"Clustering_Algorithm": algo_name, "Segment_ID": seg})
            all_results.extend(seg_results)

    return pd.DataFrame(all_results)


# ── Global baseline (no clustering) ──────────────────────────────────────────

def run_global_classification(predict_df: pd.DataFrame,
                               use_smote: bool = False,
                               test_size: float = 0.2,
                               random_state: int = 42) -> pd.DataFrame:
    """
    Train models on the full predict_df without any clustering.

    Parameters
    ----------
    predict_df : pd.DataFrame
    use_smote : bool
    test_size : float
    random_state : int

    Returns
    -------
    pd.DataFrame  with per-model evaluation metrics.
    """
    X = predict_df[FEATURES_CLF]
    y = predict_df["Will_Return"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    models  = get_classification_models(random_state)
    results = _evaluate_segment(
        X_train, X_test, y_train, y_test,
        models, use_smote, random_state
    )
    return pd.DataFrame(results)


# ── Confusion matrix for best pipeline ───────────────────────────────────────

def plot_best_pipeline_cm(rfmt_cleaned: pd.DataFrame,
                           predict_df: pd.DataFrame,
                           cluster_col: str = "KMeans_Cluster",
                           random_state: int = 42) -> None:
    """
    Aggregate confusion matrix for K-Means × XGBoost + SMOTE+Tomek
    across all segments.

    Parameters
    ----------
    rfmt_cleaned : pd.DataFrame
    predict_df : pd.DataFrame
    cluster_col : str
    random_state : int
    """
    if not _XGB_AVAILABLE:
        print("XGBoost not available – skipping confusion matrix.")
        return

    temp_df = predict_df.merge(
        rfmt_cleaned[["Customer ID", cluster_col]],
        on="Customer ID", how="left"
    )
    temp_df["Segment"] = temp_df[cluster_col].astype(str)

    all_y_test, all_y_pred = [], []

    for seg in temp_df["Segment"].dropna().unique():
        seg_df = temp_df[temp_df["Segment"] == seg]
        X = seg_df[FEATURES_CLF]
        y = seg_df["Will_Return"]
        if len(y.unique()) < 2:
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        X_res, y_res = _smote_tomek_resample(X_train, y_train, random_state)
        model = XGBClassifier(
            n_estimators=100, learning_rate=0.1,
            random_state=random_state, eval_metric="logloss"
        )
        model.fit(X_res, y_res)
        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(model.predict(X_test).tolist())

    cm = confusion_matrix(all_y_test, all_y_pred)
    labels = ["Not Return (0)", "Will Return (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title(
        "Confusion Matrix\nK-Means × XGBoost + SMOTE+Tomek",
        fontsize=12, fontweight="bold"
    )

    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
        ax=axes[1], colorbar=False, cmap="Blues", values_format=".2%"
    )
    axes[1].set_title(
        "Normalised Confusion Matrix\nK-Means × XGBoost + SMOTE+Tomek",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

    print("\n=== Classification Report ===")
    print(classification_report(all_y_test, all_y_pred, target_names=labels))


# ── Comparison visualisation ──────────────────────────────────────────────────

def plot_approach_comparison(df_no_cluster_orig: pd.DataFrame,
                              df_no_cluster_balanced: pd.DataFrame,
                              df_cluster_balanced: pd.DataFrame) -> None:
    """
    Bar charts comparing F1-Score and ROC-AUC across all approaches.

    Parameters
    ----------
    df_no_cluster_orig : pd.DataFrame  – global, no SMOTE
    df_no_cluster_balanced : pd.DataFrame – global + SMOTE
    df_cluster_balanced : pd.DataFrame  – cluster-based + SMOTE
        Must have columns: 'Clustering_Algorithm', 'Classification_Algorithm',
        'F1_Score', 'ROC_AUC'
    """
    def _prep(df, approach_label, clf_col="Classification_Algorithm"):
        d = df[[clf_col, "F1_Score", "ROC_AUC"]].copy()
        d = d.rename(columns={clf_col: "Model"})
        d["Approach"] = approach_label
        return d

    # Aggregate cluster-based to (Clustering × Classifier) average
    agg = (df_cluster_balanced
           .groupby(["Clustering_Algorithm", "Classification_Algorithm"])
           [["F1_Score", "ROC_AUC"]].mean()
           .reset_index())
    agg["Approach"] = agg["Clustering_Algorithm"] + " + Balanced"
    agg = agg.rename(columns={"Classification_Algorithm": "Model"})

    df_all = pd.concat([
        _prep(df_no_cluster_orig,      "No Clustering (Original)"),
        _prep(df_no_cluster_balanced,  "No Clustering (Balanced)"),
        agg[["Model", "Approach", "F1_Score", "ROC_AUC"]],
    ], ignore_index=True)

    for metric, title in [
        ("F1_Score", "F1-Score Comparison"),
        ("ROC_AUC",  "ROC-AUC Comparison"),
    ]:
        plt.figure(figsize=(18, 8))
        sns.barplot(
            data=df_all.sort_values(metric, ascending=False),
            x="Model", y=metric, hue="Approach",
            palette="viridis" if metric == "F1_Score" else "plasma",
        )
        plt.title(f"{title}: Original vs Balanced Baseline vs Balanced Clustering",
                  fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Classification Model", fontsize=12)
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.legend(title="Approach", bbox_to_anchor=(1.05, 1),
                   loc="upper left", fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
