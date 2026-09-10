import os
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from features.network_state import (
    STATE_FEATURES,
    create_network_states
)

from features.windowing import create_sequences

from models.lstm_world_model import LSTMWorldModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = (
    "data/processed/CIC-IDS2018/"
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
)

MODEL_PATH = (
    "models_weights/lstm_world_model.pth"
)

SCALER_PATH = (
    "models_weights/network_state_scaler.pkl"
)

THRESHOLD_PATH = (
    "models_weights/attack_threshold.pkl"
)

SEQUENCE_LENGTH = 10

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_probability,
    threshold
):
    """
    Calculate classification metrics
    for a given attack probability threshold.
    """

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate
    if (tn + fp) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    title,
    metrics,
    y_true,
    y_pred
):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Decision Threshold : "
        f"{metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['fpr']:.4f}"
    )

    print(
        f"ROC-AUC            : "
        f"{metrics['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix")
    print("-" * 40)

    print(
        "              Predicted"
    )

    print(
        "              Benign  Attack"
    )

    print(
        f"Actual Benign "
        f"{metrics['tn']:8d} "
        f"{metrics['fp']:8d}"
    )

    print(
        f"Actual Attack "
        f"{metrics['fn']:8d} "
        f"{metrics['tp']:8d}"
    )

    print("\nClassification Report")
    print("-" * 40)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Benign",
                "Attack"
            ],
            zero_division=0
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI NETWORK ATTACK FORECASTER")
    print("UPDATED LSTM WORLD MODEL EVALUATION")
    print("=" * 70)

    print(
        "\nDevice:",
        DEVICE
    )

    # ========================================================
    # CHECK FILES
    # ========================================================

    print("\nChecking required files...")
    print("-" * 40)

    required_files = [
        DATA_FILE,
        MODEL_PATH,
        SCALER_PATH
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file not found: "
                f"{file_path}"
            )

        print(
            "Found:",
            file_path
        )

    print("-" * 40)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading CIC-IDS2018 dataset...")
    print("-" * 40)

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    print(
        "Rows loaded   :",
        len(df)
    )

    print(
        "Columns loaded:",
        len(df.columns)
    )

    print("-" * 40)

    # ========================================================
    # CLEAN COLUMN NAMES
    # ========================================================

    df.columns = df.columns.str.strip()

    # ========================================================
    # TIMESTAMP
    # ========================================================

    print("\nPreparing timestamp...")
    print("-" * 40)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
        .astype(str)
        .str.strip(),
        errors="coerce",
        dayfirst=True,
        format="mixed"
    )

    valid_timestamp = (
        df["Timestamp"].notna()
    )

    print(
        "Valid timestamps  :",
        valid_timestamp.sum()
    )

    print(
        "Invalid timestamps:",
        (~valid_timestamp).sum()
    )

    df = df.loc[
        valid_timestamp
    ].copy()

    print(
        "Rows after timestamp parsing:",
        len(df)
    )

    print("-" * 40)

    # ========================================================
    # ATTACK LABEL
    # ========================================================

    print("\nCreating binary attack labels...")
    print("-" * 40)

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    df["Attack"] = (
        df["Label"]
        .str.lower()
        .ne("benign")
        .astype(np.float32)
    )

    print(
        "Benign samples:",
        int(
            (df["Attack"] == 0).sum()
        )
    )

    print(
        "Attack samples:",
        int(
            (df["Attack"] == 1).sum()
        )
    )

    print(
        "Total samples :",
        len(df)
    )

    print(
        "Attack ratio  :",
        df["Attack"].mean()
    )

    print("-" * 40)

    # ========================================================
    # VALIDATE NETWORK FEATURES
    # ========================================================

    print("\nValidating network features...")
    print("-" * 40)

    missing_features = [
        feature
        for feature in STATE_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing network features: "
            f"{missing_features}"
        )

    for feature in STATE_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_features = (
        df[STATE_FEATURES]
        .notna()
        .all(axis=1)
    )

    print(
        "Valid rows  :",
        valid_features.sum()
    )

    print(
        "Invalid rows:",
        (~valid_features).sum()
    )

    df = df.loc[
        valid_features
    ].copy()

    print("-" * 40)

    # ========================================================
    # SORT BY TIME
    # ========================================================

    print("\nSorting traffic chronologically...")
    print("-" * 40)

    df = df.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    print(
        "First timestamp:",
        df["Timestamp"].iloc[0]
    )

    print(
        "Last timestamp :",
        df["Timestamp"].iloc[-1]
    )

    print("-" * 40)

    # ========================================================
    # CREATE NETWORK STATES
    # ========================================================

    print("\nCreating network states...")

    network_states = create_network_states(
        df
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Recreate the exact valid-state mask used by
    # create_network_states()
    # --------------------------------------------------------

    state_df = df[
        STATE_FEATURES
    ].copy()

    state_df = state_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    state_valid_mask = (
        state_df.notna().all(axis=1)
        &
        (state_df >= 0).all(axis=1)
    )

    labels = df.loc[
        state_valid_mask,
        "Attack"
    ].to_numpy(
        dtype=np.float32
    )

    timestamps = df.loc[
        state_valid_mask,
        "Timestamp"
    ].to_numpy()

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(network_states) != len(labels):

        raise ValueError(
            "Network state count and label "
            "count do not match.\n"
            f"States: {len(network_states)}\n"
            f"Labels: {len(labels)}"
        )

    print(
        "\nState shape :",
        network_states.shape
    )

    print(
        "Labels shape:",
        labels.shape
    )

    # ========================================================
    # TEMPORAL SPLIT
    # ========================================================

    print(
        "\nCreating temporal "
        "train/validation split..."
    )

    print("-" * 40)

    split_index = int(
        len(network_states) * 0.80
    )

    train_states = (
        network_states[:split_index]
    )

    val_states = (
        network_states[split_index:]
    )

    train_labels = (
        labels[:split_index]
    )

    val_labels = (
        labels[split_index:]
    )

    val_timestamps = (
        timestamps[split_index:]
    )

    print(
        "Training samples  :",
        len(train_states)
    )

    print(
        "Validation samples:",
        len(val_states)
    )

    print(
        "Training attack rate:",
        train_labels.mean()
    )

    print(
        "Validation attack rate:",
        val_labels.mean()
    )

    print("-" * 40)

    # ========================================================
    # LOAD SCALER
    # ========================================================

    print("\nLoading trained scaler...")
    print("-" * 40)

    scaler = joblib.load(
        SCALER_PATH
    )

    # ========================================================
    # SCALE DATA
    # ========================================================

    print("\nScaling validation data...")
    print("-" * 40)

    train_states_scaled = (
        scaler.transform(
            train_states
        ).astype(np.float32)
    )

    val_states_scaled = (
        scaler.transform(
            val_states
        ).astype(np.float32)
    )

    print(
        "Train scaled shape:",
        train_states_scaled.shape
    )

    print(
        "Validation scaled shape:",
        val_states_scaled.shape
    )

    print("-" * 40)

    # ========================================================
    # CREATE TEMPORAL SEQUENCES
    # ========================================================

    print("\nCreating validation sequences...")

    X_val, _ = create_sequences(
        val_states_scaled,
        sequence_length=SEQUENCE_LENGTH
    )

    # --------------------------------------------------------
    # Attack label corresponds to the NEXT state.
    # --------------------------------------------------------

    y_val_attack = (
        val_labels[
            SEQUENCE_LENGTH:
        ]
    )

    val_timestamps = (
        val_timestamps[
            SEQUENCE_LENGTH:
        ]
    )

    print(
        "\nValidation sequence shape:",
        X_val.shape
    )

    print(
        "Validation attack labels:",
        y_val_attack.shape
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\nLoading LSTM World Model...")
    print("-" * 40)

    model = LSTMWorldModel(
        input_size=len(STATE_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print(
        "Model loaded successfully!"
    )

    print("-" * 40)

    # ========================================================
    # GENERATE ATTACK PROBABILITIES
    # ========================================================

    print(
        "\nGenerating attack probabilities..."
    )

    X_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    probabilities = []

    # --------------------------------------------------------
    # Batch inference
    # --------------------------------------------------------

    batch_size = 512

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(X_tensor)
            )

            X_batch = X_tensor[
                start:end
            ].to(DEVICE)

            # Latest model returns:
            # next_state, attack_logits

            _, attack_logits = model(
                X_batch
            )

            # Convert logits → probability

            attack_probability = (
                torch.sigmoid(
                    attack_logits
                )
            )

            probabilities.extend(
                attack_probability
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            )

    y_probability = np.asarray(
        probabilities,
        dtype=np.float32
    )

    print(
        "Predictions generated:",
        len(y_probability)
    )

    # ========================================================
    # ROC-AUC
    # ========================================================

    if len(
        np.unique(y_val_attack)
    ) < 2:

        raise ValueError(
            "Validation data contains "
            "only one class. ROC-AUC "
            "cannot be calculated."
        )

    roc_auc = roc_auc_score(
        y_val_attack,
        y_probability
    )

    print(
        "\nROC-AUC:",
        f"{roc_auc:.4f}"
    )

    # ========================================================
    # DEFAULT THRESHOLD
    # ========================================================

    default_threshold = 0.50

    default_metrics = calculate_metrics(
        y_val_attack,
        y_probability,
        default_threshold
    )

    default_metrics[
        "roc_auc"
    ] = roc_auc

    y_default_pred = (
        y_probability >=
        default_threshold
    ).astype(int)

    print_metrics(
        "LSTM WORLD MODEL - THRESHOLD 0.50",
        default_metrics,
        y_val_attack,
        y_default_pred
    )

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LSTM THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = np.arange(
        0.05,
        0.51,
        0.05
    )

    threshold_results = []

    print(
        "\nThreshold | Precision | Recall | F1 | FPR"
    )

    print("-" * 70)

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_val_attack,
            y_probability,
            float(threshold)
        )

        threshold_results.append(
            metrics
        )

        print(
            f"{threshold:9.2f} | "
            f"{metrics['precision']:9.4f} | "
            f"{metrics['recall']:6.4f} | "
            f"{metrics['f1']:6.4f} | "
            f"{metrics['fpr']:6.4f}"
        )

    # ========================================================
    # BEST F1 THRESHOLD
    # ========================================================

    best_result = max(
        threshold_results,
        key=lambda x: x["f1"]
    )

    best_threshold = (
        best_result["threshold"]
    )

    best_metrics = best_result.copy()

    best_metrics[
        "roc_auc"
    ] = roc_auc

    # ========================================================
    # SAVE THRESHOLD
    # ========================================================

    joblib.dump(
        best_threshold,
        THRESHOLD_PATH
    )

    print("\n")
    print(
        "Best LSTM Threshold:",
        f"{best_threshold:.2f}"
    )

    print(
        "Best LSTM F1:",
        f"{best_metrics['f1']:.4f}"
    )

    print(
        "Threshold saved:",
        THRESHOLD_PATH
    )

    # ========================================================
    # BEST THRESHOLD EVALUATION
    # ========================================================

    y_best_pred = (
        y_probability >=
        best_threshold
    ).astype(int)

    print_metrics(
        "BEST LSTM WORLD MODEL EVALUATION",
        best_metrics,
        y_val_attack,
        y_best_pred
    )

    # ========================================================
    # ATTACK TIMELINE INFORMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ATTACK RISK TIMELINE SUMMARY")
    print("=" * 70)

    print(
        "\nMinimum attack probability:",
        f"{y_probability.min():.4f}"
    )

    print(
        "Maximum attack probability:",
        f"{y_probability.max():.4f}"
    )

    print(
        "Mean attack probability:",
        f"{y_probability.mean():.4f}"
    )

    # --------------------------------------------------------
    # Count high-risk predictions
    # --------------------------------------------------------

    high_risk_count = (
        y_probability >= 0.70
    ).sum()

    medium_risk_count = (
        (y_probability >= 0.30)
        &
        (y_probability < 0.70)
    ).sum()

    low_risk_count = (
        y_probability < 0.30
    ).sum()

    print(
        "\nRisk Distribution"
    )

    print("-" * 40)

    print(
        "Low Risk    (< 0.30):",
        int(low_risk_count)
    )

    print(
        "Medium Risk (0.30-0.70):",
        int(medium_risk_count)
    )

    print(
        "High Risk   (>= 0.70):",
        int(high_risk_count)
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_file = (
        "data/processed/CIC-IDS2018/"
        "lstm_attack_predictions.csv"
    )

    prediction_df = pd.DataFrame(
        {
            "Timestamp": val_timestamps,
            "Actual_Attack": y_val_attack,
            "Attack_Probability": y_probability,
            "Predicted_Attack": y_best_pred
        }
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    print("\n")
    print(
        "Prediction file saved:"
    )

    print(
        prediction_file
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL LSTM EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"\nBest Threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy       : "
        f"{best_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision      : "
        f"{best_metrics['precision']:.4f}"
    )

    print(
        f"Recall         : "
        f"{best_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score       : "
        f"{best_metrics['f1']:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{best_metrics['fpr']:.4f}"
    )

    print(
        f"ROC-AUC        : "
        f"{roc_auc:.4f}"
    )

    print("\n" + "=" * 70)
    print("LSTM EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()