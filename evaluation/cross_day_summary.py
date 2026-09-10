import pandas as pd
import os

# ============================================================
# CROSS-DAY EVALUATION SUMMARY
# ============================================================

results = [
    {
        "Test Day": "Wednesday 14-02-2018",
        "Attack Type": "Brute Force",
        "Evaluation Type": "Validation",
        "Accuracy": 0.8840,
        "Precision": 0.2935,
        "Recall": 0.5470,
        "F1 Score": 0.3820,
        "FPR": 0.1728,
        "ROC-AUC": 0.7732,
    },
    {
        "Test Day": "Friday 16-02-2018",
        "Attack Type": "DoS",
        "Evaluation Type": "Unseen Day",
        "Accuracy": 0.5172,
        "Precision": 0.9922,
        "Recall": 0.0429,
        "F1 Score": 0.0823,
        "FPR": 0.0003,
        "ROC-AUC": 0.1885,
    },
    {
        "Test Day": "Thursday 01-03-2018",
        "Attack Type": "Infiltration",
        "Evaluation Type": "Unseen Day",
        "Accuracy": 0.7205,
        "Precision": 0.5066,
        "Recall": 0.2795,
        "F1 Score": 0.3602,
        "FPR": 0.1067,
        "ROC-AUC": 0.5854,
    },
]

# Create DataFrame
df = pd.DataFrame(results)

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("CROSS-DAY LSTM EVALUATION SUMMARY")
print("=" * 80)

print(df.to_string(index=False))

# ============================================================
# AVERAGE
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "FPR",
    "ROC-AUC",
]

print("\n" + "-" * 80)
print("AVERAGE PERFORMANCE")
print("-" * 80)

for metric in metric_columns:
    print(
        f"{metric:12s}: "
        f"{df[metric].mean():.4f}"
    )

# ============================================================
# SAVE
# ============================================================

output_dir = "data/processed/CIC-IDS2018"

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "cross_day_lstm_summary.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("SUMMARY SAVED")
print("=" * 80)

print(output_file)