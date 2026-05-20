"""
cluster_profiling.py
--------------------
Visualise and interpret cluster profiles after K-Means clustering.

Charts produced:
  1. Per-cluster RFMT bar charts (4 metrics × clusters)
  2. Cluster size vs. return rate (dual-axis)
  3. Cluster size vs. cumulative revenue contribution (dual-axis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Default labels / colours for 5-cluster solution ──────────────────────────

DEFAULT_COLORS = {
    0: "#E74C3C",
    1: "#C0392B",
    2: "#27AE60",
    3: "#F39C12",
    4: "#2980B9",
}

DEFAULT_LABELS = {
    0: "At-Risk",
    1: "Churned",
    2: "Potential Loyalists",
    3: "VIPs",
    4: "Occasional Buyers",
}

# High → low return rate
DEFAULT_ORDER = [3, 2, 4, 0, 1]


# ── Centroid summary ──────────────────────────────────────────────────────────

def build_centroid_summary(rfmt_cleaned: pd.DataFrame,
                            predict_df: pd.DataFrame,
                            cluster_col: str = "KMeans_Cluster",
                            features: list = None) -> pd.DataFrame:
    """
    Compute mean RFMT values, cluster size, share, and return rate
    per cluster.

    Parameters
    ----------
    rfmt_cleaned : pd.DataFrame
        Must contain cluster_col and RFMT columns.
    predict_df : pd.DataFrame
        Must contain 'Customer ID', 'Will_Return', and cluster_col after merge.
    cluster_col : str
    features : list[str]

    Returns
    -------
    pd.DataFrame  indexed by cluster id.
    """
    if features is None:
        features = ["Recency", "Frequency", "Monetary", "T"]

    centroid = rfmt_cleaned.groupby(cluster_col)[features].mean().round(1)
    sizes    = rfmt_cleaned[cluster_col].value_counts().sort_index()
    centroid["N"]        = sizes
    centroid["PctTotal"] = (sizes / len(rfmt_cleaned) * 100).round(1)

    # Return rate from predict_df
    rr = (
        predict_df
        .merge(rfmt_cleaned[["Customer ID", cluster_col]], on="Customer ID", how="left")
        .groupby(cluster_col)["Will_Return"]
        .mean()
        .mul(100)
        .round(1)
    )
    centroid["ReturnRate"] = rr
    return centroid


# ── Chart 1: RFMT bar charts ──────────────────────────────────────────────────

def plot_rfmt_by_cluster(centroid: pd.DataFrame,
                          cluster_order: list = None,
                          cluster_colors: dict = None,
                          cluster_labels: dict = None,
                          features: list = None) -> None:
    """
    2×2 bar charts showing the mean value of each RFMT feature per cluster.

    Parameters
    ----------
    centroid : pd.DataFrame   from build_centroid_summary()
    cluster_order : list      cluster ids sorted high→low by return rate
    cluster_colors : dict     {cluster_id: hex_color}
    cluster_labels : dict     {cluster_id: "Label"}
    features : list[str]
    """
    if cluster_order  is None: cluster_order  = list(centroid.index)
    if cluster_colors is None: cluster_colors = DEFAULT_COLORS
    if cluster_labels is None: cluster_labels = DEFAULT_LABELS
    if features       is None: features       = ["Recency", "Frequency", "Monetary", "T"]

    titles = [
        "Recency (days since last purchase)",
        "Frequency (avg invoices per customer)",
        "Monetary (avg spend, £)",
        "Inter-purchase Interval T (days)",
    ]
    units = ["days", "invoices", "£", "days"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Cluster Profiles by RFMT Dimensions",
                 fontsize=15, fontweight="bold")

    for ax, feat, title, unit in zip(axes.flatten(), features, titles, units):
        vals   = [centroid.loc[c, feat]  for c in cluster_order]
        colors = [cluster_colors.get(c, "#888") for c in cluster_order]
        names  = [f"C{c}\n{cluster_labels.get(c, c)}" for c in cluster_order]

        bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.6)
        for bar, val in zip(bars, vals):
            fmt = f"£{val:,.0f}" if feat == "Monetary" else f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                fmt, ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


# ── Chart 2: cluster size vs return rate ─────────────────────────────────────

def plot_size_vs_returnrate(centroid: pd.DataFrame,
                             cluster_order: list = None,
                             cluster_colors: dict = None,
                             cluster_labels: dict = None) -> None:
    """
    Dual-axis chart: cluster size (bars) + return rate (line).

    Parameters
    ----------
    centroid : pd.DataFrame   from build_centroid_summary()
    """
    if cluster_order  is None: cluster_order  = list(centroid.index)
    if cluster_colors is None: cluster_colors = DEFAULT_COLORS
    if cluster_labels is None: cluster_labels = DEFAULT_LABELS

    names   = [f"C{c}: {cluster_labels.get(c, c)}" for c in cluster_order]
    sizes   = [int(centroid.loc[c, "N"])          for c in cluster_order]
    rrates  = [centroid.loc[c, "ReturnRate"]       for c in cluster_order]
    colors  = [cluster_colors.get(c, "#888")       for c in cluster_order]
    x       = np.arange(len(cluster_order))
    total   = sum(sizes)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(x, sizes, color=colors, alpha=0.85,
                   edgecolor="white", width=0.5, label="Cluster size (N)")

    for bar, n in zip(bars, sizes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{n:,}\n({n/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color="#333"
        )

    ax1.set_ylabel("Number of customers", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, rotation=30, ha="right")
    ax1.set_ylim(0, max(sizes) * 1.22)

    ax2 = ax1.twinx()
    ax2.plot(x, rrates, "o--", color="navy", linewidth=2,
             markersize=9, label="Return rate (%)")
    for xi, rr in zip(x, rrates):
        ax2.annotate(f"{rr:.1f}%", xy=(xi, rr), xytext=(0, 10),
                     textcoords="offset points", ha="center",
                     fontsize=10, fontweight="bold", color="navy")
    ax2.set_ylabel("Return Rate (%)", fontsize=11, color="navy")
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="y", labelcolor="navy")

    lines1, lbs1 = ax1.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbs1 + lbs2, loc="upper right", fontsize=9)
    ax1.set_title("Cluster Size vs. Return Rate",
                  fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


# ── Chart 3: cluster size vs cumulative revenue ───────────────────────────────

def plot_size_vs_cumrevenue(rfmt_cleaned: pd.DataFrame,
                             transactions_df: pd.DataFrame,
                             centroid: pd.DataFrame,
                             cluster_order: list = None,
                             cluster_colors: dict = None,
                             cluster_labels: dict = None,
                             cluster_col: str = "KMeans_Cluster") -> None:
    """
    Dual-axis chart: cluster size (bars) + cumulative revenue % (line).

    Parameters
    ----------
    rfmt_cleaned : pd.DataFrame
    transactions_df : pd.DataFrame
        Full cleaned transaction DataFrame (must have 'Quantity', 'Price',
        'Customer ID').
    centroid : pd.DataFrame   from build_centroid_summary()
    cluster_order : list
    cluster_colors : dict
    cluster_labels : dict
    cluster_col : str
    """
    if cluster_order  is None: cluster_order  = list(centroid.index)
    if cluster_colors is None: cluster_colors = DEFAULT_COLORS
    if cluster_labels is None: cluster_labels = DEFAULT_LABELS

    # Revenue per cluster
    trans = transactions_df.copy()
    trans["Revenue"] = trans["Quantity"] * trans["Price"]
    rev_cluster = (
        trans
        .merge(rfmt_cleaned[["Customer ID", cluster_col]], on="Customer ID", how="left")
        .groupby(cluster_col)["Revenue"]
        .sum()
    )
    rev_pct     = (rev_cluster / rev_cluster.sum() * 100).round(1)
    rev_seq     = [rev_pct.loc[c] for c in cluster_order]
    cum_rev_pct = np.cumsum(rev_seq)

    names  = [f"C{c}: {cluster_labels.get(c, c)}" for c in cluster_order]
    sizes  = [int(centroid.loc[c, "N"]) for c in cluster_order]
    colors = [cluster_colors.get(c, "#888") for c in cluster_order]
    x      = np.arange(len(cluster_order))
    total  = sum(sizes)

    fig, ax1 = plt.subplots(figsize=(13, 6))
    bars = ax1.bar(x, sizes, color=colors, alpha=0.85,
                   edgecolor="white", width=0.5, label="Cluster size (N)")

    for bar, n in zip(bars, sizes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{n:,}\n({n/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color="#333"
        )
    ax1.set_ylabel("Number of customers", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.tick_params(axis="x", rotation=15)
    ax1.set_ylim(0, max(sizes) * 1.18)

    ax2 = ax1.twinx()
    ax2.plot(x, cum_rev_pct, "D-", color="darkgreen", linewidth=2.5,
             markersize=9, label="Cumulative revenue (%)")
    for xi, cv, rv in zip(x, cum_rev_pct, rev_seq):
        ax2.annotate(f"{cv:.1f}%", xy=(xi, cv), xytext=(0, 10),
                     textcoords="offset points", ha="center",
                     fontsize=10, fontweight="bold", color="darkgreen")
        ax2.annotate(f"(+{rv:.1f}%)", xy=(xi, cv), xytext=(0, -14),
                     textcoords="offset points", ha="center",
                     fontsize=8, fontweight="bold", color="navy")
    ax2.axhline(100, color="darkgreen", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Cumulative Revenue Contribution (%)",
                   fontsize=11, color="darkgreen")
    ax2.set_ylim(0, 120)
    ax2.tick_params(axis="y", labelcolor="darkgreen")

    lines1, lbs1 = ax1.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbs1 + lbs2, loc="upper right", fontsize=9)
    ax1.set_title("Cluster Size vs. Cumulative Revenue Contribution",
                  fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
