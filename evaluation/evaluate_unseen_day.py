import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from features.network_state import create_network_states
from features.windowing import create_sequences
from models.lstm_world_model import LSTMWorldModel


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = (
    "data/processed/CIC-IDS2018/"
    "Friday-16-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
)

MODEL_PATH = "models_weights/lstm_world_model.pth"
SCALER_PATH = "models_weights/network_state_scaler.pkl"

SEQUENCE_LENGTH = 10
THRESHOLD = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD SCALER
# ============================================================

import joblib


print("\n" + "=" * 60)
print("LSTM UNSEEN-DAY EVALUATION — FRIDAY")
print("=" * 60)

print("Device:", DEVICE)

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

print("\nFOUND:")
print(TEST_FILE)
print(MODEL_PATH)
print(SCALER_PATH)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Friday dataset...")

df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("\nDataset loaded")
print("--------------------------------")
print("Rows    :", len(df))
print("Columns :", len(df.columns))
print("--------------------------------")


# ============================================================
# TIMESTAMP
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"].astype(str).str.strip(),
    errors="coerce",
    dayfirst=True,
    format="mixed"
)

df = df.dropna(subset=["Timestamp"])

df = df.sort_values("Timestamp").reset_index(drop=True)

print("\nTimestamp range:")
print("First:", df["Timestamp"].min())
print("Last :", df["Timestamp"].max())


# ============================================================
# VERIFY DATE
# ============================================================

expected_date = "2018-02-16"

actual_dates = df["Timestamp"].dt.strftime("%Y-%m-%d").unique()

print("\nDates found:", actual_dates)

if not all(date == expected_date for date in actual_dates):
    raise ValueError(
        f"Unexpected dates found. Expected only {expected_date}"
    )


# ============================================================
# LABEL PREPARATION
# ============================================================

if "Label" not in df.columns:
    raise ValueError("Label column not found.")

df["Label"] = df["Label"].astype(str).str.strip()

df["Attack"] = (
    df["Label"]
    .str.lower()
    .ne("benign")
    .astype(int)
)

print("\nAttack distribution")
print("--------------------------------")
print("Benign :", int((df["Attack"] == 0).sum()))
print("Attack :", int((df["Attack"] == 1).sum()))
print(
    "Attack rate:",
    df["Attack"].mean()
)
print("--------------------------------")


# ============================================================
# CREATE NETWORK STATES
# ============================================================

print("\nCreating network states...")

state_df = df.copy()

states = create_network_states(state_df)

print("\nState shape:", states.shape)


# ============================================================
# ALIGN LABELS
# ============================================================

# create_network_states removes invalid rows internally.
# Recreate the same valid-state filtering for labels.

from features.network_state import (
    STATE_FEATURES,
    NON_NEGATIVE_FEATURES,
)

aligned_df = df[STATE_FEATURES].copy()

for column in STATE_FEATURES:
    aligned_df[column] = pd.to_numeric(
        aligned_df[column],
        errors="coerce"
    )

aligned_df = aligned_df.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_mask = ~aligned_df.isna().any(axis=1)

valid_mask &= ~(
    aligned_df[NON_NEGATIVE_FEATURES] < 0
).any(axis=1)

labels = df.loc[valid_mask, "Attack"].to_numpy(
    dtype=np.int64
)

timestamps = df.loc[valid_mask, "Timestamp"].to_numpy()

print("\nState/label alignment")
print("--------------------------------")
print("Valid samples:", len(labels))
print("State samples:", len(states))
print("Labels       :", len(labels))
print("--------------------------------")

if len(states) != len(labels):
    raise ValueError(
        "State and label lengths do not match."
    )


# ============================================================
# LOAD WEDNESDAY SCALER
# ============================================================

print("\nLoading Wednesday-trained scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")
print("IMPORTANT: scaler is NOT refitted on Friday.")


# ============================================================
# SCALE STATES
# ============================================================

scaled_states = scaler.transform(states)

scaled_states = scaled_states.astype(
    np.float32
)

print("\nScaled state shape:")
print(scaled_states.shape)


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================

print("\nCreating temporal sequences...")

X, y = create_sequences(
    scaled_states,
    sequence_length=SEQUENCE_LENGTH
)

# Labels corresponding to each next-state prediction
y_attack = labels[SEQUENCE_LENGTH:]

sequence_timestamps = timestamps[SEQUENCE_LENGTH:]

print("\nSequence shapes")
print("--------------------------------")
print("X:", X.shape)
print("y:", y_attack.shape)
print("timestamps:", sequence_timestamps.shape)
print("--------------------------------")

if len(X) != len(y_attack):
    raise ValueError(
        "Sequence and label lengths do not match."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Wednesday-trained LSTM...")

model = LSTMWorldModel(
    input_size=X.shape[2],
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
).to(DEVICE)

all_probabilities = []

with torch.no_grad():

    batch_size = 512

    for start in range(
        0,
        len(X_tensor),
        batch_size
    ):

        batch = X_tensor[
            start:start + batch_size
        ]

        _, logits = model(batch)

        probabilities = torch.sigmoid(
            logits
        )

        all_probabilities.append(
            probabilities
            .cpu()
            .numpy()
            .reshape(-1)
        )

probabilities = np.concatenate(
    all_probabilities
)


# ============================================================
# CLASSIFICATION
# ============================================================

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_attack,
    predictions
)

precision = precision_score(
    y_attack,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_attack,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_attack,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_attack,
    predictions
)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)

if len(np.unique(y_attack)) == 2:
    roc_auc = roc_auc_score(
        y_attack,
        probabilities
    )
else:
    roc_auc = float("nan")


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("UNSEEN-DAY RESULTS")
print("=" * 60)

print("Test Dataset : February 16, 2018")
print("Attack Type  : DoS")
print("Threshold    :", THRESHOLD)

print("--------------------------------")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"FPR          : {fpr:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")
print("--------------------------------")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ============================================================
# PROBABILITY STATISTICS
# ============================================================

print("\nProbability statistics")
print("--------------------------------")

print(
    "Minimum :",
    float(probabilities.min())
)

print(
    "Maximum :",
    float(probabilities.max())
)

print(
    "Mean    :",
    float(probabilities.mean())
)

print(
    "Median  :",
    float(np.median(probabilities))
)


# ============================================================
# PROBABILITY BY ACTUAL CLASS
# ============================================================

benign_probs = probabilities[
    y_attack == 0
]

attack_probs = probabilities[
    y_attack == 1
]

print("\nProbability by actual class")
print("--------------------------------")

print(
    "Benign mean:",
    float(benign_probs.mean())
)

print(
    "Attack mean:",
    float(attack_probs.mean())
)


# ============================================================
# RISK LEVEL
# ============================================================

risk_levels = np.where(
    probabilities < 0.30,
    "Low",
    np.where(
        probabilities < 0.70,
        "Medium",
        "High"
    )
)

print("\nRisk distribution")
print("--------------------------------")

print(
    "Low < 0.30:",
    int((risk_levels == "Low").sum())
)

print(
    "Medium 0.30-0.70:",
    int((risk_levels == "Medium").sum())
)

print(
    "High >= 0.70:",
    int((risk_levels == "High").sum())
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output_file = (
    "data/processed/CIC-IDS2018/"
    "lstm_unseen_friday_predictions.csv"
)

prediction_df = pd.DataFrame({
    "Timestamp": sequence_timestamps,
    "Actual_Attack": y_attack,
    "Attack_Probability": probabilities,
    "Predicted_Attack": predictions,
    "Risk_Level": risk_levels
})

prediction_df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 60)
print("FRIDAY UNSEEN-DAY EVALUATION COMPLETE")
print("=" * 60)