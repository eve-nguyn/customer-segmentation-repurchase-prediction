"""
main.py
-------
End-to-end pipeline runner for the Hybrid Customer Segmentation and
Churn Prediction Framework (UCI Online Retail II).

Usage
-----
Edit the DATA_PATHS section below, then run:
    python src/main.py
"""

import pandas as pd

# ── User configuration ────────────────────────────────────────────────────────

DATA_PATHS = {
    "path_09_10": "data/online_retail_II_2009_2010.csv",
    "path_10_11": "data/online_retail_II_2010_2011.csv",
    "sep"       : ";",
}

N_CLUSTERS    = 4      # K / c for K-Means, Fuzzy C-Means, Spectral
RANDOM_STATE  = 42
FORECAST_DAYS = 30     # Tail days used as forecast window for churn label
USE_SMOTE     = True   # Apply SMOTE+Tomek inside cluster-based classification

# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load & clean ───────────────────────────────────────────────────────
    from data_loader import load_and_clean
    print("=" * 60)
    print("STEP 1 — DATA LOADING & CLEANING")
    print("=" * 60)
    df = load_and_clean(
        DATA_PATHS["path_09_10"],
        DATA_PATHS["path_10_11"],
        sep=DATA_PATHS["sep"],
    )

    # ── 2. Outlier treatment for raw Quantity / Price ────────────────────────
    from outlier_utils import cap_columns
    print("\n" + "=" * 60)
    print("STEP 2 — OUTLIER TREATMENT (Qty & Price)")
    print("=" * 60)
    df = cap_columns(df, ["Quantity", "Price"])

    # ── 3. RFMT feature engineering ──────────────────────────────────────────
    from feature_engineering import (
        compute_rfmt, clean_rfmt, temporal_split,
        build_churn_target, plot_rfmt_distributions,
        plot_rfmt_boxplots, print_rfmt_stats,
    )
    print("\n" + "=" * 60)
    print("STEP 3 — RFMT FEATURE ENGINEERING")
    print("=" * 60)

    # Parse InvoiceDate before temporal split
    import pandas as pd
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d.%m.%Y %H:%M")

    obs_df, forecast_df, _ = temporal_split(df, FORECAST_DAYS)

    rfmt_full = compute_rfmt(df)
    rfmt_cleaned = clean_rfmt(rfmt_full)

    # EDA
    plot_rfmt_distributions(rfmt_cleaned, title_suffix="After Outlier Treatment")
    plot_rfmt_boxplots(rfmt_cleaned)
    print_rfmt_stats(rfmt_cleaned)

    # Build churn prediction target
    predict_df = build_churn_target(obs_df, forecast_df)

    # ── 4. Clustering ─────────────────────────────────────────────────────────
    from clustering import run_all_clustering, scale_rfmt, select_k_elbow
    print("\n" + "=" * 60)
    print("STEP 4 — CLUSTERING")
    print("=" * 60)

    rfmt_scaled, _ = scale_rfmt(rfmt_cleaned)
    optimal_k = select_k_elbow(rfmt_scaled)
    k = optimal_k if optimal_k else N_CLUSTERS
    print(f"Using k = {k}")

    rfmt_clustered = run_all_clustering(rfmt_cleaned, n_clusters=k,
                                         random_state=RANDOM_STATE)

    # ── 5. Classification (global baseline + cluster-based) ───────────────────
    from classification import (
        run_global_classification,
        run_cluster_classification,
        plot_approach_comparison,
        plot_best_pipeline_cm,
    )
    print("\n" + "=" * 60)
    print("STEP 5 — CLASSIFICATION")
    print("=" * 60)

    df_global_orig     = run_global_classification(predict_df, use_smote=False,
                                                    random_state=RANDOM_STATE)
    df_global_balanced = run_global_classification(predict_df, use_smote=True,
                                                    random_state=RANDOM_STATE)
    df_cluster_balanced = run_cluster_classification(
        rfmt_clustered, predict_df, use_smote=True, random_state=RANDOM_STATE
    )

    print("\n─── Global baseline (no SMOTE) ───")
    print(df_global_orig.sort_values("F1_Score", ascending=False).to_string(index=False))

    print("\n─── Global baseline + SMOTE ───")
    print(df_global_balanced.sort_values("F1_Score", ascending=False).to_string(index=False))

    print("\n─── Cluster-based + SMOTE (avg per combo) ───")
    agg = (df_cluster_balanced
           .groupby(["Clustering_Algorithm", "Classification_Algorithm"])
           [["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]]
           .mean()
           .round(4))
    print(agg.sort_values("F1_Score", ascending=False).to_string())

    plot_approach_comparison(df_global_orig, df_global_balanced, df_cluster_balanced)
    plot_best_pipeline_cm(rfmt_clustered, predict_df, random_state=RANDOM_STATE)

    # ── 6. Cluster profiling ──────────────────────────────────────────────────
    from cluster_profiling import (
        build_centroid_summary,
        plot_rfmt_by_cluster,
        plot_size_vs_returnrate,
        plot_size_vs_cumrevenue,
    )
    print("\n" + "=" * 60)
    print("STEP 6 — CLUSTER PROFILING")
    print("=" * 60)

    centroid = build_centroid_summary(rfmt_clustered, predict_df)
    print(centroid.to_string())

    plot_rfmt_by_cluster(centroid)
    plot_size_vs_returnrate(centroid)
    plot_size_vs_cumrevenue(rfmt_clustered, df, centroid)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
