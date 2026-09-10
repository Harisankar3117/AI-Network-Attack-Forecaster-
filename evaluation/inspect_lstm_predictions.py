import pandas as pd
import numpy as np


PREDICTION_FILE = "data/processed/CIC-IDS2018/lstm_attack_predictions.csv"


print("=" * 70)
print("LSTM ATTACK PREDICTION INSPECTION")
print("=" * 70)

print("\nLoading prediction file...")
df = pd.read_csv(PREDICTION_FILE)

print("----------------------------------------")
print("Rows    :", len(df))
print("Columns :", len(df.columns))
print("----------------------------------------")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 10 predictions:")
print(df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

print(df.describe(include="all").transpose().to_string())


# Find probability column
probability_columns = [
    col for col in df.columns
    if "prob" in col.lower()
]

if probability_columns:
    probability_col = probability_columns[0]

    print("\n" + "=" * 70)
    print("ATTACK PROBABILITY ANALYSIS")
    print("=" * 70)

    probabilities = pd.to_numeric(
        df[probability_col],
        errors="coerce"
    ).dropna()

    print("----------------------------------------")
    print("Probability column :", probability_col)
    print("Minimum            :", probabilities.min())
    print("Maximum            :", probabilities.max())
    print("Mean               :", probabilities.mean())
    print("Median             :", probabilities.median())
    print("Std                :", probabilities.std())
    print("----------------------------------------")

    print("\nProbability percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"{p:>3}% percentile : {np.percentile(probabilities, p):.4f}")


# Find actual attack label
label_columns = [
    col for col in df.columns
    if col.lower() in ["attack", "actual_attack", "y_true"]
]

if label_columns:
    label_col = label_columns[0]

    print("\n" + "=" * 70)
    print("ACTUAL ATTACK DISTRIBUTION")
    print("=" * 70)

    print(df[label_col].value_counts().to_string())


# Probability bins
if probability_columns:
    print("\n" + "=" * 70)
    print("RISK DISTRIBUTION")
    print("=" * 70)

    bins = [-np.inf, 0.10, 0.25, 0.30, 0.50, 0.70, np.inf]
    labels = [
        "Very Low (<0.10)",
        "Low (0.10-0.25)",
        "Medium-Low (0.25-0.30)",
        "Medium (0.30-0.50)",
        "High (0.50-0.70)",
        "Very High (>=0.70)"
    ]

    risk_distribution = pd.cut(
        probabilities,
        bins=bins,
        labels=labels
    )

    print(risk_distribution.value_counts().sort_index().to_string())


# Attack vs benign probability comparison
if probability_columns and label_columns:

    print("\n" + "=" * 70)
    print("ATTACK vs BENIGN PROBABILITY")
    print("=" * 70)

    comparison = df.groupby(label_col)[probability_col].agg(
        ["count", "mean", "median", "min", "max"]
    )

    print(comparison.to_string())


print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)