"""
data_loader.py
--------------
Load and preprocess the UCI Online Retail II dataset.
Handles merging of two CSV files, missing value removal,
duplicate removal, cancelled invoice filtering, and
Quantity/Price cleaning.
"""

import pandas as pd
import numpy as np


# ── Noise stock codes to exclude ──────────────────────────────────────────────
NOISE_CODES = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK"}


def load_data(path_09_10: str, path_10_11: str, sep: str = ";") -> pd.DataFrame:
    """
    Load and concatenate the two Online Retail II CSV files.

    Parameters
    ----------
    path_09_10 : str
        Path to the 2009–2010 CSV file.
    path_10_11 : str
        Path to the 2010–2011 CSV file.
    sep : str
        CSV delimiter (default ';').

    Returns
    -------
    pd.DataFrame
        Concatenated raw DataFrame.
    """
    df_2009_2010 = pd.read_csv(path_09_10, sep=sep)
    df_2010_2011 = pd.read_csv(path_10_11, sep=sep)
    df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)
    print(f"Loaded {len(df):,} rows from 2 files.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
      1. Drop rows with missing Customer ID.
      2. Remove duplicate rows.
      3. Remove cancelled invoices (Invoice starts with 'C').
      4. Remove noise StockCodes.
      5. Remove rows where Quantity <= 0 or Price <= 0.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    print(f"Original shape : {df.shape}")

    # 1. Drop missing Customer ID
    df = df.dropna(subset=["Customer ID"])
    print(f"After drop null Customer ID : {df.shape}")

    # 2. Remove duplicates
    df = df.drop_duplicates()
    print(f"After drop duplicates       : {df.shape}")

    # 3. Remove cancelled invoices
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    print(f"After remove cancelled      : {df.shape}")

    # 4. Remove noise stock codes
    df = df[~df["StockCode"].astype(str).str.upper().isin(NOISE_CODES)]
    print(f"After remove noise codes    : {df.shape}")

    # 5. Remove non-positive Quantity / Price
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    print(f"After remove neg Qty/Price  : {df.shape}")

    return df.reset_index(drop=True)


def load_and_clean(path_09_10: str, path_10_11: str, sep: str = ";") -> pd.DataFrame:
    """
    Convenience wrapper: load then clean in one call.
    """
    df = load_data(path_09_10, path_10_11, sep)
    df = clean_data(df)
    return df
