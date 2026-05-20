"""
feature_engineering.py
-----------------------
Compute RFMT (Recency, Frequency, Monetary, Inter-purchase Interval T)
features from the cleaned Online Retail II dataset.

Also includes:
  - Temporal train/forecast split
  - RFMT distribution EDA (histograms, boxplots, correlation, skewness)
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from outlier_utils import cap_columns


# ── RFMT computation ──────────────────────────────────────────────────────────

def _calculate_t(row: pd.Series) -> float:
    """Average inter-purchase interval in days."""
    if row["Frequency"] > 1:
        return (row["Tn"] - row["T1"]).days / (row["Frequency"] - 1)
    return float(row["Recency"])


def compute_rfmt(df: pd.DataFrame,
                 date_format: str = "%d.%m.%Y %H:%M") -> pd.DataFrame:
    """
    Compute RFMT features at customer level.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame with columns:
        'Customer ID', 'InvoiceDate', 'Invoice', 'Quantity', 'Price'.
    date_format : str
        strptime format for InvoiceDate.

    Returns
    -------
    pd.DataFrame
        Customer-level RFMT DataFrame with columns:
        ['Customer ID', 'Recency', 'Frequency', 'Monetary', 'T']
    """
    df = df.copy()

    # Parse dates and compute TotalPrice
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format=date_format)
    df["TotalPrice"]  = df["Quantity"] * df["Price"]

    analysis_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfmt = df.groupby("Customer ID").agg(
        Recency    = ("InvoiceDate", lambda d: (analysis_date - d.max()).days),
        T1         = ("InvoiceDate", "min"),
        Tn         = ("InvoiceDate", "max"),
        Frequency  = ("Invoice",     "nunique"),
        Monetary   = ("TotalPrice",  "sum"),
    ).reset_index()

    rfmt["T"] = rfmt.apply(_calculate_t, axis=1)
    rfmt = rfmt.drop(columns=["T1", "Tn"])

    print(f"RFMT computed for {len(rfmt):,} customers.")
    return rfmt


def clean_rfmt(rfmt: pd.DataFrame,
               columns: list = None) -> pd.DataFrame:
    """
    Apply percentile-based capping to RFMT variables.

    Parameters
    ----------
    rfmt : pd.DataFrame
    columns : list[str], optional
        Columns to cap (default: ['Recency', 'Frequency', 'Monetary', 'T']).

    Returns
    -------
    pd.DataFrame (copy)
    """
    if columns is None:
        columns = ["Recency", "Frequency", "Monetary", "T"]
    return cap_columns(rfmt, columns)


# ── Temporal split ────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame,
                   forecast_window_days: int = 30):
    """
    Split the transaction DataFrame into an observation period and
    a forecast period.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions with a parsed 'InvoiceDate' column.
    forecast_window_days : int
        Number of tail days reserved for the forecast window.

    Returns
    -------
    (obs_df, forecast_df, cutoff_date) : tuple
    """
    max_date    = df["InvoiceDate"].max()
    cutoff_date = max_date - dt.timedelta(days=forecast_window_days)

    obs_df      = df[df["InvoiceDate"] <  cutoff_date]
    forecast_df = df[df["InvoiceDate"] >= cutoff_date]

    print(f"Max date     : {max_date.date()}")
    print(f"Cut-off date : {cutoff_date.date()}")
    print(f"Obs rows     : {len(obs_df):,}  |  Forecast rows : {len(forecast_df):,}")
    return obs_df, forecast_df, cutoff_date


def build_churn_target(obs_df: pd.DataFrame,
                       forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the customer-level prediction DataFrame used in classification.

    For each customer in obs_df:
      - Monetary_Past  : total spend in observation period
      - Will_Return    : 1 if customer has at least one transaction in forecast window

    Parameters
    ----------
    obs_df : pd.DataFrame
    forecast_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with columns:
        ['Customer ID', 'Recency', 'Frequency', 'Monetary_Past', 'Will_Return']
    """
    # RFMT from observation period only
    rfmt_obs = compute_rfmt(obs_df)
    rfmt_obs = rfmt_obs.rename(columns={"Monetary": "Monetary_Past"})

    # Customers who bought in forecast window
    returning = set(forecast_df["Customer ID"].unique())
    rfmt_obs["Will_Return"] = rfmt_obs["Customer ID"].isin(returning).astype(int)

    print(f"Predict DF shape : {rfmt_obs.shape}")
    print(f"Will_Return distribution:\n{rfmt_obs['Will_Return'].value_counts()}")
    return rfmt_obs


# ── EDA ───────────────────────────────────────────────────────────────────────

RFMT_LABELS = {
    "Recency"   : "Recency (days)",
    "Frequency" : "Frequency (invoices)",
    "Monetary"  : "Monetary (£)",
    "T"         : "T – Inter-purchase (days)",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_rfmt_distributions(rfmt: pd.DataFrame,
                             features: list = None,
                             title_suffix: str = "") -> None:
    """
    Plot histogram + KDE for each RFMT feature with mean/median lines
    and skewness annotation.

    Parameters
    ----------
    rfmt : pd.DataFrame
    features : list[str]
    title_suffix : str
        Appended to the figure suptitle for context.
    """
    if features is None:
        features = ["Recency", "Frequency", "Monetary", "T"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"RFMT Variable Distribution {title_suffix}",
        fontsize=14, fontweight="bold"
    )

    for ax, feat, clr in zip(axes.flatten(), features, COLORS):
        data = rfmt[feat].dropna()
        ax.hist(data, bins=50, color=clr, alpha=0.65, edgecolor="white",
                density=True, label="Histogram")
        data.plot.kde(ax=ax, color="black", linewidth=1.8, label="KDE")
        ax.axvline(data.mean(),   color="red",  linestyle="--", linewidth=1.2,
                   label=f"Mean={data.mean():.1f}")
        ax.axvline(data.median(), color="navy", linestyle=":",  linewidth=1.2,
                   label=f"Median={data.median():.1f}")
        ax.set_title(RFMT_LABELS.get(feat, feat), fontsize=11)
        ax.set_xlabel(RFMT_LABELS.get(feat, feat), fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        skew = data.skew()
        ax.text(0.97, 0.95, f"Skewness = {skew:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color="dimgray",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6))
        ax.legend(fontsize=7.5)

    plt.tight_layout()
    plt.show()


def plot_rfmt_boxplots(rfmt: pd.DataFrame, features: list = None) -> None:
    """
    2-row grid: histograms (top) + boxplots (bottom) for RFMT features.

    Parameters
    ----------
    rfmt : pd.DataFrame
    features : list[str]
    """
    if features is None:
        features = ["Recency", "Frequency", "Monetary", "T"]

    labels = [RFMT_LABELS.get(f, f) for f in features]

    fig, axes = plt.subplots(2, len(features), figsize=(5 * len(features), 10))
    fig.suptitle("RFMT Variable Distribution After Capping — Before Standardisation",
                 fontsize=14, fontweight="bold")

    for i, (feat, label) in enumerate(zip(features, labels)):
        data = rfmt[feat].dropna()

        # Histogram
        axes[0, i].hist(data, bins=40, color="steelblue",
                         edgecolor="white", alpha=0.85)
        axes[0, i].axvline(data.median(), color="red",    linestyle="--",
                            linewidth=1.5, label=f"Median={data.median():.1f}")
        axes[0, i].axvline(data.mean(),   color="orange", linestyle="--",
                            linewidth=1.5, label=f"Mean={data.mean():.1f}")
        axes[0, i].set_title(f"Histogram: {label}", fontsize=10)
        axes[0, i].set_xlabel(label, fontsize=9)
        axes[0, i].set_ylabel("Number of Customers", fontsize=9)
        axes[0, i].legend(fontsize=8)

        # Boxplot
        axes[1, i].boxplot(data, vert=True, patch_artist=True,
                            boxprops=dict(facecolor="lightblue", color="navy"),
                            medianprops=dict(color="red", linewidth=2))
        axes[1, i].set_title(f"Boxplot: {label}", fontsize=10)
        axes[1, i].set_ylabel(label, fontsize=9)

    plt.tight_layout()
    plt.show()


def print_rfmt_stats(rfmt: pd.DataFrame, features: list = None) -> None:
    """
    Print descriptive statistics + skewness & kurtosis for RFMT features.

    Parameters
    ----------
    rfmt : pd.DataFrame
    features : list[str]
    """
    if features is None:
        features = ["Recency", "Frequency", "Monetary", "T"]

    desc = rfmt[features].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    desc.loc["skewness"] = rfmt[features].skew()
    desc.loc["kurtosis"] = rfmt[features].kurt()
    print("\n" + "=" * 60)
    print("RFMT DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(desc.round(2).to_string())

    print("\n=== Skewness & Kurtosis ===")
    for feat in features:
        sk = stats.skew(rfmt[feat].dropna())
        ku = stats.kurtosis(rfmt[feat].dropna())
        print(f"  {feat:12s}: skewness={sk:6.3f}, kurtosis={ku:7.3f}")
