import os
import glob
import pandas as pd
import numpy as np

from preprocessing.data_cleaner import clean_dataset


# ============================================================
# PATH CONFIGURATION
# ============================================================

RAW_DIR = "data/raw/CIC-IDS2018"
PROCESSED_DIR = "data/processed/CIC-IDS2018"

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# FILES TO PROCESS
# ============================================================

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
]


# ============================================================
# EXPECTED DATES
# ============================================================

EXPECTED_DATES = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": "2018-02-14",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": "2018-02-16",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv": "2018-02-22",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv": "2018-03-01",
}


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(filename):

    input_path = os.path.join(RAW_DIR, filename)

    output_filename = filename.replace(
        ".csv",
        "_cleaned.csv"
    )

    output_path = os.path.join(
        PROCESSED_DIR,
        output_filename
    )

    print("\n" + "=" * 80)
    print("PROCESSING FILE")
    print("=" * 80)
    print("Input :", input_path)
    print("Output:", output_path)

    if not os.path.exists(input_path):
        print("\nERROR: File not found!")
        print(input_path)
        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading CSV...")

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    print("Rows loaded   :", len(df))
    print("Columns loaded:", len(df.columns))

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    df = clean_dataset(df)

    if df.empty:
        print("ERROR: Dataset became empty after cleaning.")
        return

    # --------------------------------------------------------
    # Timestamp parsing
    # --------------------------------------------------------

    print("\nParsing timestamps...")

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"].astype(str).str.strip(),
        errors="coerce",
        dayfirst=True,
        format="mixed"
    )

    invalid_timestamp_count = df["Timestamp"].isna().sum()

    print(
        "Invalid timestamps:",
        invalid_timestamp_count
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["Timestamp"]
    ).copy()

    # --------------------------------------------------------
    # IMPORTANT TIMESTAMP FIX
    # --------------------------------------------------------

    expected_date = pd.Timestamp(
        EXPECTED_DATES[filename]
    ).date()

    before_date_filter = len(df)

    df = df[
        df["Timestamp"].dt.date == expected_date
    ].copy()

    removed_wrong_dates = (
        before_date_filter - len(df)
    )

    print(
        "Wrong-date rows removed:",
        removed_wrong_dates
    )

    print(
        "Expected date:",
        expected_date
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    print("\nConverting numeric columns...")

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Remove remaining NaN
    # --------------------------------------------------------

    before_nan = len(df)

    df = df.dropna()

    removed_nan = (
        before_nan - len(df)
    )

    print(
        "NaN rows removed:",
        removed_nan
    )

    # --------------------------------------------------------
    # Create binary attack label
    # --------------------------------------------------------

    if "Label" not in df.columns:

        print(
            "ERROR: Label column not found!"
        )

        return

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # Benign = 0
    # Everything else = 1

    df["Attack"] = (
        df["Label"]
        .str.lower()
        .ne("benign")
        .astype(int)
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates()

    removed_duplicates = (
        before_duplicates - len(df)
    )

    print(
        "Duplicate rows removed:",
        removed_duplicates
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        by="Timestamp"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Final timestamp verification
    # --------------------------------------------------------

    print("\nTimestamp verification:")

    print(
        "First timestamp:",
        df["Timestamp"].min()
    )

    print(
        "Last timestamp :",
        df["Timestamp"].max()
    )

    before_2018 = (
        df["Timestamp"].dt.year < 2018
    ).sum()

    print(
        "Rows before 2018:",
        before_2018
    )

    wrong_date_rows = (
        df["Timestamp"].dt.date != expected_date
    ).sum()

    print(
        "Wrong-date rows:",
        wrong_date_rows
    )

    # --------------------------------------------------------
    # Label statistics
    # --------------------------------------------------------

    benign_count = (
        df["Attack"] == 0
    ).sum()

    attack_count = (
        df["Attack"] == 1
    ).sum()

    print("\nLabel statistics:")
    print(
        "Benign:",
        benign_count
    )

    print(
        "Attack:",
        attack_count
    )

    print(
        "Attack ratio:",
        attack_count / len(df)
    )

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    print("\nSaving cleaned dataset...")

    df.to_csv(
        output_path,
        index=False
    )

    print("\nSUCCESS!")
    print("----------------------------------------")
    print("Final rows   :", len(df))
    print("Final columns:", len(df.columns))
    print("Saved to     :", output_path)
    print("----------------------------------------")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("CIC-IDS2018 DATA PREPARATION")
    print("=" * 80)

    print("\nRaw directory:")
    print(RAW_DIR)

    print("\nProcessed directory:")
    print(PROCESSED_DIR)

    for filename in FILES:

        process_file(filename)

    print("\n")
    print("=" * 80)
    print("ALL DATASETS PROCESSED")
    print("=" * 80)


if __name__ == "__main__":
    main()