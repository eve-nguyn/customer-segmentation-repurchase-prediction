"""
outlier_utils.py
----------------
Outlier detection (IQR) and treatment (percentile-based capping)
for the Online Retail II pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Threshold helpers ─────────────────────────────────────────────────────────

def outlier_thresholds(df: pd.DataFrame, variable: str):
    """
    Compute upper and lower capping bounds using the 1st–99th percentile IQR.

    Parameters
    ----------
    df : pd.DataFrame
    variable : str

    Returns
    -------
    (up_limit, low_limit) : tuple[float, float]
    """
    q1 = df[variable].quantile(0.01)
    q3 = df[variable].quantile(0.99)
    iqr = q3 - q1
    up_limit  = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return up_limit, low_limit


def replace_with_threshold(df: pd.DataFrame, variable: str) -> None:
    """
    Cap outliers in-place using outlier_thresholds().
    Converts int64 columns to float64 before capping.

    Parameters
    ----------
    df : pd.DataFrame  (modified in-place)
    variable : str
    """
    up_limit, low_limit = outlier_thresholds(df, variable)
    if df[variable].dtype == "int64":
        df[variable] = df[variable].astype(float)
    df.loc[df[variable] < low_limit, variable] = low_limit
    df.loc[df[variable] > up_limit,  variable] = up_limit


def cap_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply replace_with_threshold to a list of columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]

    Returns
    -------
    pd.DataFrame  (copy with capped values)
    """
    df = df.copy()
    for col in columns:
        replace_with_threshold(df, col)
    return df


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_iqr_outliers(df: pd.DataFrame, variable: str) -> None:
    """
    Plot boxplot and scatter plot highlighting IQR outliers for a variable.
    Uses the standard IQR rule (Q1−1.5·IQR, Q3+1.5·IQR).

    Parameters
    ----------
    df : pd.DataFrame
    variable : str
    """
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    outliers = df[(df[variable] < lower_bound) | (df[variable] > upper_bound)]
    print(f"Outliers in '{variable}': {len(outliers):,}  "
          f"(lower={lower_bound:.2f}, upper={upper_bound:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Boxplot
    sns.boxplot(x=df[variable], ax=axes[0])
    axes[0].set_title(f"Boxplot – {variable}")
    axes[0].set_xlabel(variable)

    # Scatter
    axes[1].scatter(df.index, df[variable], alpha=0.4, s=8, label="Normal")
    axes[1].scatter(outliers.index, outliers[variable],
                    color="red", alpha=0.6, s=12, label="Outliers")
    axes[1].axhline(upper_bound, color="green", linestyle="--", label="Upper bound")
    axes[1].axhline(lower_bound, color="blue",  linestyle="--", label="Lower bound")
    axes[1].set_title(f"Scatter – {variable} (IQR outliers highlighted)")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel(variable)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_boxplots_after_capping(df: pd.DataFrame, columns: list) -> None:
    """
    Side-by-side boxplots for a list of columns (post-capping check).

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
    """
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Post-capping: {col}")
        ax.set_xlabel(col)
    plt.suptitle("Distribution After Outlier Treatment", fontweight="bold")
    plt.tight_layout()
    plt.show()
