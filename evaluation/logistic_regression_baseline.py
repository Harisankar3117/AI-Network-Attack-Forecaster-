import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from features.network_state import (
    STATE_FEATURES,
    NON_NEGATIVE_FEATURES
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = (
    "data/processed/CIC-IDS2018/"
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
)

SCALER_PATH = (
    "models_weights/"
    "logistic_regression_scaler.pkl"
)

MODEL_PATH = (
    "models_weights/"
    "logistic_regression_baseline.pkl"
)

TRAIN_RATIO = 0.80

# Same threshold selected during LSTM analysis
LSTM_THRESHOLD = 0.25


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading CIC-IDS2018 dataset...")
    print("--------------------------------")

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

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

    print("--------------------------------")

    return df


# ============================================================
# PREPARE TIMESTAMP
# ============================================================

def prepare_timestamp(df):

    print("\nPreparing timestamp...")

    if "Timestamp" not in df.columns:

        raise ValueError(
            "Timestamp column not found."
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
        .astype(str)
        .str.strip(),
        errors="coerce",
        dayfirst=True,
        format="mixed"
    )

    invalid = df[
        "Timestamp"
    ].isna().sum()

    print("--------------------------------")

    print(
        "Valid timestamps  :",
        len(df) - invalid
    )

    print(
        "Invalid timestamps:",
        invalid
    )

    print("--------------------------------")

    df = df.dropna(
        subset=["Timestamp"]
    ).copy()

    df = df.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    print(
        "Rows after timestamp parsing:",
        len(df)
    )

    return df


# ============================================================
# CREATE ATTACK LABEL
# ============================================================

def create_attack_labels(df):

    print("\nCreating binary attack labels...")

    if "Label" not in df.columns:

        raise ValueError(
            "Label column not found."
        )

    labels = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Benign = 0
    # Attack = 1

    df["Attack"] = (
        labels != "benign"
    ).astype(np.int32)

    benign_count = int(
        (df["Attack"] == 0).sum()
    )

    attack_count = int(
        (df["Attack"] == 1).sum()
    )

    print("--------------------------------")

    print(
        "Benign samples:",
        benign_count
    )

    print(
        "Attack samples:",
        attack_count
    )

    print(
        "Total samples :",
        len(df)
    )

    print(
        "Attack ratio  :",
        df["Attack"].mean()
    )

    print("--------------------------------")

    return df


# ============================================================
# VALIDATE NETWORK FEATURES
# ============================================================

def validate_features(df):

    print("\nValidating network features...")

    missing_features = [
        feature
        for feature in STATE_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing network features:\n"
            + str(missing_features)
        )

    state_df = df[
        STATE_FEATURES
    ].copy()

    # Convert all features to numeric

    for column in STATE_FEATURES:

        state_df[column] = pd.to_numeric(
            state_df[column],
            errors="coerce"
        )

    # Replace infinity

    state_df = state_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Start with all rows valid

    valid_mask = pd.Series(
        True,
        index=df.index
    )

    # Remove NaN rows

    valid_mask &= (
        state_df.notna().all(axis=1)
    )

    # Remove invalid negative values

    for column in NON_NEGATIVE_FEATURES:

        valid_mask &= (
            state_df[column] >= 0
        )

    valid_count = int(
        valid_mask.sum()
    )

    invalid_count = int(
        (~valid_mask).sum()
    )

    print("--------------------------------")

    print(
        "Valid rows  :",
        valid_count
    )

    print(
        "Invalid rows:",
        invalid_count
    )

    print("--------------------------------")

    if valid_count == 0:

        raise ValueError(
            "No valid network state rows found."
        )

    df = df.loc[
        valid_mask
    ].copy()

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# CREATE NETWORK STATES
# ============================================================

def create_states(df):

    print("\nCreating network states...")

    state_df = df[
        STATE_FEATURES
    ].copy()

    for column in STATE_FEATURES:

        state_df[column] = pd.to_numeric(
            state_df[column],
            errors="coerce"
        )

    state_df = state_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if state_df.isna().any().any():

        raise ValueError(
            "NaN values found in network states."
        )

    states = state_df.to_numpy(
        dtype=np.float32
    )

    print("--------------------------------")

    print(
        "State shape:",
        states.shape
    )

    print("--------------------------------")

    return states


# ============================================================
# TEMPORAL TRAIN / VALIDATION SPLIT
# ============================================================

def temporal_split(
    states,
    labels
):

    print("\nCreating temporal train/validation split...")

    split_index = int(
        len(states) * TRAIN_RATIO
    )

    train_states = states[
        :split_index
    ]

    val_states = states[
        split_index:
    ]

    train_labels = labels[
        :split_index
    ]

    val_labels = labels[
        split_index:
    ]

    print("--------------------------------")

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

    print("--------------------------------")

    return (
        train_states,
        val_states,
        train_labels,
        val_labels
    )


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(
    train_states,
    val_states
):

    print("\nScaling network features...")

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_states
    )

    val_scaled = scaler.transform(
        val_states
    )

    train_scaled = train_scaled.astype(
        np.float32
    )

    val_scaled = val_scaled.astype(
        np.float32
    )

    print("--------------------------------")

    print(
        "Train scaled shape:",
        train_scaled.shape
    )

    print(
        "Validation scaled shape:",
        val_scaled.shape
    )

    print("--------------------------------")

    return (
        train_scaled,
        val_scaled,
        scaler
    )


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

def train_model(
    train_states,
    train_labels
):

    print("\nTraining Logistic Regression...")
    print("--------------------------------")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )

    model.fit(
        train_states,
        train_labels
    )

    print(
        "Logistic Regression training completed!"
    )

    print("--------------------------------")

    return model


# ============================================================
# FALSE POSITIVE RATE
# ============================================================

def calculate_fpr(
    actual,
    predicted
):

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    denominator = tn + fp

    if denominator == 0:

        return 0.0

    return fp / denominator


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    val_states,
    val_labels,
    threshold=0.50
):

    print("\nGenerating predictions...")

    probabilities = model.predict_proba(
        val_states
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        val_labels,
        predictions
    )

    precision = precision_score(
        val_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        val_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        val_labels,
        predictions,
        zero_division=0
    )

    fpr = calculate_fpr(
        val_labels,
        predictions
    )

    try:

        roc_auc = roc_auc_score(
            val_labels,
            probabilities
        )

    except ValueError:

        roc_auc = float("nan")

    cm = confusion_matrix(
        val_labels,
        predictions,
        labels=[0, 1]
    )

    print("\n")
    print("=" * 70)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    print(
        f"\nDecision Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Precision          : {precision:.4f}"
    )

    print(
        f"Recall             : {recall:.4f}"
    )

    print(
        f"F1 Score           : {f1:.4f}"
    )

    print(
        f"False Positive Rate: {fpr:.4f}"
    )

    print(
        f"ROC-AUC            : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix")
    print("--------------------------------")

    print(
        "              Predicted"
    )

    print(
        "              Benign  Attack"
    )

    print(
        f"Actual Benign  {cm[0,0]:7d}  {cm[0,1]:7d}"
    )

    print(
        f"Actual Attack  {cm[1,0]:7d}  {cm[1,1]:7d}"
    )

    print("--------------------------------")

    print("\nClassification Report")
    print("--------------------------------")

    print(
        classification_report(
            val_labels,
            predictions,
            target_names=[
                "Benign",
                "Attack"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc
    }


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    model,
    val_states,
    val_labels
):

    print("\n")
    print("=" * 70)
    print("LOGISTIC REGRESSION THRESHOLD ANALYSIS")
    print("=" * 70)

    probabilities = model.predict_proba(
        val_states
    )[:, 1]

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50
    ]

    best_threshold = 0.50
    best_f1 = -1.0

    print(
        "\nThreshold | Precision | Recall | F1 | FPR"
    )

    print(
        "-" * 58
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            val_labels,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            val_labels,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            val_labels,
            predictions,
            zero_division=0
        )

        fpr = calculate_fpr(
            val_labels,
            predictions
        )

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f} | "
            f"{fpr:6.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    print(
        "-" * 58
    )

    print(
        f"\nBest Logistic Threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best Logistic F1        : "
        f"{best_f1:.4f}"
    )

    return best_threshold


# ============================================================
# SAVE MODEL AND SCALER
# ============================================================

def save_artifacts(
    model,
    scaler,
    threshold
):

    os.makedirs(
        "models_weights",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    threshold_path = (
        "models_weights/"
        "logistic_regression_threshold.pkl"
    )

    joblib.dump(
        threshold,
        threshold_path
    )

    print("\nArtifacts saved!")
    print("--------------------------------")

    print(
        "Model     :",
        MODEL_PATH
    )

    print(
        "Scaler    :",
        SCALER_PATH
    )

    print(
        "Threshold :",
        threshold_path
    )

    print("--------------------------------")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI NETWORK ATTACK FORECASTER")
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. TIMESTAMP
    # --------------------------------------------------------

    df = prepare_timestamp(
        df
    )

    # --------------------------------------------------------
    # 3. ATTACK LABEL
    # --------------------------------------------------------

    df = create_attack_labels(
        df
    )

    # --------------------------------------------------------
    # 4. VALIDATE FEATURES
    # --------------------------------------------------------

    df = validate_features(
        df
    )

    # --------------------------------------------------------
    # 5. CREATE NETWORK STATES
    # --------------------------------------------------------

    states = create_states(
        df
    )

    # --------------------------------------------------------
    # 6. LABELS
    # --------------------------------------------------------

    labels = df[
        "Attack"
    ].to_numpy(
        dtype=np.int32
    )

    # --------------------------------------------------------
    # 7. TEMPORAL SPLIT
    # --------------------------------------------------------

    (
        train_states,
        val_states,
        train_labels,
        val_labels
    ) = temporal_split(
        states,
        labels
    )

    # --------------------------------------------------------
    # 8. SCALE
    # --------------------------------------------------------

    (
        train_scaled,
        val_scaled,
        scaler
    ) = scale_features(
        train_states,
        val_states
    )

    # --------------------------------------------------------
    # 9. TRAIN MODEL
    # --------------------------------------------------------

    model = train_model(
        train_scaled,
        train_labels
    )

    # --------------------------------------------------------
    # 10. STANDARD EVALUATION
    # --------------------------------------------------------

    results_50 = evaluate_model(
        model,
        val_scaled,
        val_labels,
        threshold=0.50
    )

    # --------------------------------------------------------
    # 11. THRESHOLD ANALYSIS
    # --------------------------------------------------------

    best_threshold = threshold_analysis(
        model,
        val_scaled,
        val_labels
    )

    # --------------------------------------------------------
    # 12. BEST THRESHOLD EVALUATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BEST LOGISTIC REGRESSION EVALUATION")
    print("=" * 70)

    best_results = evaluate_model(
        model,
        val_scaled,
        val_labels,
        threshold=best_threshold
    )

    # --------------------------------------------------------
    # 13. SAVE
    # --------------------------------------------------------

    save_artifacts(
        model,
        scaler,
        best_threshold
    )

    # --------------------------------------------------------
    # 14. LSTM COMPARISON
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("LSTM vs LOGISTIC REGRESSION")
    print("=" * 70)

    print(
        "\nLSTM (Threshold 0.25)"
    )

    print(
        "Accuracy  : 0.7966"
    )

    print(
        "Precision : 0.2902"
    )

    print(
        "Recall    : 0.5211"
    )

    print(
        "F1 Score  : 0.3728"
    )

    print(
        "FPR       : 0.1673"
    )

    print(
        "ROC-AUC   : 0.7646"
    )

    print(
        "\nLogistic Regression (Best Threshold)"
    )

    print(
        f"Threshold : {best_threshold:.2f}"
    )

    print(
        f"Accuracy  : {best_results['accuracy']:.4f}"
    )

    print(
        f"Precision : {best_results['precision']:.4f}"
    )

    print(
        f"Recall    : {best_results['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_results['f1']:.4f}"
    )

    print(
        f"FPR       : {best_results['fpr']:.4f}"
    )

    print(
        f"ROC-AUC   : {best_results['roc_auc']:.4f}"
    )

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION BASELINE COMPLETED!")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()