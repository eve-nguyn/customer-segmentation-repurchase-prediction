"""
clustering.py
-------------
Clustering pipeline for the RFMT customer segmentation step.

Supported algorithms:
  - K-Means      (sklearn)
  - Fuzzy C-Means (skfuzzy)
  - Spectral Clustering (sklearn)
  - MeanShift    (sklearn)

Utilities:
  - StandardScaler normalisation
  - Elbow / Silhouette selection for K-Means
  - Cluster label assignment back to the RFMT DataFrame
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# ── Standardisation ───────────────────────────────────────────────────────────

RFMT_FEATURES = ["Recency", "Frequency", "Monetary", "T"]


def scale_rfmt(rfmt: pd.DataFrame,
               features: list = None) -> tuple:
    """
    Fit a StandardScaler on rfmt[features] and return the scaled array
    together with the fitted scaler.

    Parameters
    ----------
    rfmt : pd.DataFrame
    features : list[str]

    Returns
    -------
    (rfmt_scaled, scaler) : (np.ndarray, StandardScaler)
    """
    if features is None:
        features = RFMT_FEATURES
    scaler = StandardScaler()
    rfmt_scaled = scaler.fit_transform(rfmt[features])
    return rfmt_scaled, scaler


# ── K-Means ───────────────────────────────────────────────────────────────────

def select_k_elbow(rfmt_scaled: np.ndarray,
                   k_range: range = range(2, 11),
                   random_state: int = 42) -> int:
    """
    Use the distortion elbow method to suggest an optimal K.
    Requires yellowbrick (pip install yellowbrick).

    Parameters
    ----------
    rfmt_scaled : np.ndarray
    k_range : range
    random_state : int

    Returns
    -------
    optimal_k : int  (None if yellowbrick is unavailable)
    """
    try:
        from yellowbrick.cluster import KElbowVisualizer
        model = KMeans(random_state=random_state, n_init=10)
        viz   = KElbowVisualizer(model, k=k_range, metric="distortion", timings=False)
        viz.fit(rfmt_scaled)
        viz.show()
        return viz.elbow_value_
    except ImportError:
        print("yellowbrick not installed – computing distortion manually.")
        distortions = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            km.fit(rfmt_scaled)
            distortions.append(km.inertia_)
        plt.figure(figsize=(8, 4))
        plt.plot(list(k_range), distortions, "o-")
        plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow Curve")
        plt.tight_layout(); plt.show()
        return None


def run_kmeans(rfmt: pd.DataFrame,
               rfmt_scaled: np.ndarray,
               n_clusters: int = 4,
               random_state: int = 42) -> pd.DataFrame:
    """
    Fit K-Means and attach 'KMeans_Cluster' labels to rfmt.

    Parameters
    ----------
    rfmt : pd.DataFrame
    rfmt_scaled : np.ndarray
    n_clusters : int
    random_state : int

    Returns
    -------
    pd.DataFrame  (copy with 'KMeans_Cluster' column added)
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(rfmt_scaled)
    sil = silhouette_score(rfmt_scaled, labels)

    rfmt = rfmt.copy()
    rfmt["KMeans_Cluster"] = labels

    print(f"K-Means  |  k={n_clusters}  |  Silhouette={sil:.4f}")
    return rfmt


# ── Fuzzy C-Means ─────────────────────────────────────────────────────────────

def run_fuzzy_cmeans(rfmt: pd.DataFrame,
                     rfmt_scaled: np.ndarray,
                     n_clusters: int = 4,
                     m: float = 2.0,
                     error: float = 0.005,
                     maxiter: int = 1000,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Fit Fuzzy C-Means (skfuzzy) and attach 'Fuzzy_Cluster' labels to rfmt.

    Parameters
    ----------
    rfmt : pd.DataFrame
    rfmt_scaled : np.ndarray
    n_clusters : int
    m : float       Fuzzification exponent (default 2)
    error : float   Convergence threshold
    maxiter : int   Maximum iterations
    random_state : int

    Returns
    -------
    pd.DataFrame  (copy with 'Fuzzy_Cluster' column added)
    """
    try:
        import skfuzzy as fuzz
    except ImportError:
        raise ImportError("skfuzzy not installed – run: pip install scikit-fuzzy")

    np.random.seed(random_state)
    data = rfmt_scaled.T          # skfuzzy expects (features × samples)

    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        data, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None
    )
    labels = np.argmax(u, axis=0)
    sil    = silhouette_score(rfmt_scaled, labels)

    rfmt = rfmt.copy()
    rfmt["Fuzzy_Cluster"] = labels

    print(f"Fuzzy C-Means  |  c={n_clusters}  |  FPC={fpc:.4f}  |  Silhouette={sil:.4f}")
    return rfmt


# ── Spectral Clustering ───────────────────────────────────────────────────────

def run_spectral(rfmt: pd.DataFrame,
                 rfmt_scaled: np.ndarray,
                 n_clusters: int = 4,
                 random_state: int = 42) -> pd.DataFrame:
    """
    Fit Spectral Clustering and attach 'Spectral_Cluster' labels to rfmt.

    Parameters
    ----------
    rfmt : pd.DataFrame
    rfmt_scaled : np.ndarray
    n_clusters : int
    random_state : int

    Returns
    -------
    pd.DataFrame  (copy with 'Spectral_Cluster' column added)
    """
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        random_state=random_state,
        n_jobs=-1,
    )
    labels = sc.fit_predict(rfmt_scaled)
    sil    = silhouette_score(rfmt_scaled, labels)

    rfmt = rfmt.copy()
    rfmt["Spectral_Cluster"] = labels

    print(f"Spectral  |  k={n_clusters}  |  Silhouette={sil:.4f}")
    return rfmt


# ── Mean Shift ────────────────────────────────────────────────────────────────

def run_meanshift(rfmt: pd.DataFrame,
                  rfmt_scaled: np.ndarray,
                  quantile: float = 0.2,
                  n_samples: int = 500,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Fit MeanShift clustering and attach 'MeanShift_Cluster' labels to rfmt.
    Bandwidth is estimated automatically.

    Parameters
    ----------
    rfmt : pd.DataFrame
    rfmt_scaled : np.ndarray
    quantile : float  Passed to estimate_bandwidth
    n_samples : int   Passed to estimate_bandwidth
    random_state : int

    Returns
    -------
    pd.DataFrame  (copy with 'MeanShift_Cluster' column added)
    """
    np.random.seed(random_state)
    bw = estimate_bandwidth(rfmt_scaled, quantile=quantile, n_samples=n_samples)
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    labels = ms.fit_predict(rfmt_scaled)

    n_clusters = len(np.unique(labels))
    sil = silhouette_score(rfmt_scaled, labels) if n_clusters > 1 else float("nan")

    rfmt = rfmt.copy()
    rfmt["MeanShift_Cluster"] = labels

    print(f"MeanShift  |  k_detected={n_clusters}  |  bandwidth={bw:.4f}  |  Silhouette={sil:.4f}")
    return rfmt


# ── Run all algorithms ────────────────────────────────────────────────────────

def run_all_clustering(rfmt: pd.DataFrame,
                       n_clusters: int = 4,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Convenience function: scale rfmt features and run K-Means, Fuzzy C-Means,
    Spectral, and MeanShift in sequence, returning a single DataFrame with
    all four cluster label columns.

    Parameters
    ----------
    rfmt : pd.DataFrame
    n_clusters : int   Used for K-Means, Fuzzy, Spectral (MeanShift auto-detects)
    random_state : int

    Returns
    -------
    pd.DataFrame
    """
    rfmt_scaled, _ = scale_rfmt(rfmt)

    rfmt = run_kmeans(rfmt, rfmt_scaled, n_clusters, random_state)
    rfmt = run_fuzzy_cmeans(rfmt, rfmt_scaled, n_clusters, random_state=random_state)
    rfmt = run_spectral(rfmt, rfmt_scaled, n_clusters, random_state)
    rfmt = run_meanshift(rfmt, rfmt_scaled, random_state=random_state)

    return rfmt
