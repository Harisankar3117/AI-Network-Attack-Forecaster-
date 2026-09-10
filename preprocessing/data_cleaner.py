import pandas as pd
import numpy as np


def clean_dataset(df):

    print("Starting data cleaning...")

    original_rows = len(df)

    # 1. Clean column names
    df.columns = df.columns.str.strip()

    # 2. Remove repeated header rows
    if "Label" in df.columns:
        df = df[df["Label"].astype(str).str.strip() != "Label"]

    # 3. Remove duplicate rows
    before_duplicates = len(df)
    df = df.drop_duplicates()
    removed_duplicates = before_duplicates - len(df)

    # 4. Replace infinity values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 5. Count missing values
    missing_before = df.isna().sum().sum()

    # 6. Remove rows containing missing values
    df = df.dropna()

    missing_after = df.isna().sum().sum()

    print("\nData cleaning completed!")
    print("--------------------------------")
    print("Original rows       :", original_rows)
    print("Duplicate rows removed:", removed_duplicates)
    print("Missing values found:", missing_before)
    print("Remaining rows      :", len(df))
    print("Rows removed total  :", original_rows - len(df))
    print("--------------------------------")

    return df