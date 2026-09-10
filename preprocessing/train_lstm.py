import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from features.network_state import (
    STATE_FEATURES,
    create_network_states
)

from features.windowing import create_sequences

from models.lstm_world_model import (
    LSTMWorldModel
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = (
    "data/processed/CIC-IDS2018/"
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
)

MODEL_DIR = "models_weights"

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "network_state_scaler.pkl"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_world_model.pth"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# HYPERPARAMETERS
# ============================================================

SEQUENCE_LENGTH = 10

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.2

BATCH_SIZE = 256

EPOCHS = 15

LEARNING_RATE = 0.0005

STATE_LOSS_WEIGHT = 0.20

ATTACK_LOSS_WEIGHT = 1.0

GRADIENT_CLIP = 1.0

PATIENCE = 4

TRAIN_RATIO = 0.80

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print(
    "Rows loaded:",
    len(df)
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
)


# ============================================================
# TIMESTAMP
# ============================================================

print("\nParsing timestamps...")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
    .astype(str)
    .str.strip(),
    errors="coerce",
    dayfirst=True,
    format="mixed"
)

invalid_timestamps = (
    df["Timestamp"].isna().sum()
)

print(
    "Invalid timestamps:",
    invalid_timestamps
)

df = df.dropna(
    subset=["Timestamp"]
).copy()


# ============================================================
# TIMESTAMP RANGE CHECK
# ============================================================

df = df[
    df["Timestamp"].dt.date
    == pd.Timestamp("2018-02-14").date()
].copy()


print(
    "Rows after timestamp validation:",
    len(df)
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "Timestamp"
).reset_index(
    drop=True
)


print(
    "First timestamp:",
    df["Timestamp"].min()
)

print(
    "Last timestamp:",
    df["Timestamp"].max()
)


# ============================================================
# CREATE BINARY ATTACK LABEL
# ============================================================

if "Label" not in df.columns:

    raise ValueError(
        "Label column not found."
    )


df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)


df["Attack"] = (
    df["Label"]
    .str.lower()
    .ne("benign")
    .astype(int)
)


print("\nClass distribution:")

print(
    "Benign:",
    (df["Attack"] == 0).sum()
)

print(
    "Attack:",
    (df["Attack"] == 1).sum()
)


# ============================================================
# NETWORK STATE DATAFRAME
# ============================================================

print("\nPreparing network state features...")

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


# ============================================================
# VALID STATE MASK
# ============================================================

valid_state_mask = (
    state_df.notna().all(axis=1)
    &
    (state_df >= 0).all(axis=1)
)


removed_state_rows = (
    (~valid_state_mask).sum()
)


print(
    "Invalid state rows removed:",
    removed_state_rows
)


# Keep labels aligned with states
state_df = state_df.loc[
    valid_state_mask
].copy()

labels = df.loc[
    valid_state_mask,
    "Attack"
].to_numpy(
    dtype=np.float32
)

timestamps = df.loc[
    valid_state_mask,
    "Timestamp"
].reset_index(
    drop=True
)


# ============================================================
# CREATE NETWORK STATES
# ============================================================

network_states = create_network_states(
    state_df
)


labels = labels[
    :len(network_states)
]


print(
    "\nNetwork state shape:",
    network_states.shape
)

print(
    "Label shape:",
    labels.shape
)


# ============================================================
# TEMPORAL TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(
    len(network_states)
    * TRAIN_RATIO
)


train_states = network_states[
    :split_index
]

val_states = network_states[
    split_index:
]


train_labels = labels[
    :split_index
]

val_labels = labels[
    split_index:
]


print("\nTemporal split:")

print(
    "Training samples:",
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


# ============================================================
# SCALE NETWORK STATES
# ============================================================

print("\nScaling network states...")

scaler = StandardScaler()

train_states_scaled = scaler.fit_transform(
    train_states
).astype(
    np.float32
)

val_states_scaled = scaler.transform(
    val_states
).astype(
    np.float32
)


# ============================================================
# SAVE SCALER
# ============================================================

with open(
    SCALER_PATH,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print(
    "Scaler saved:",
    SCALER_PATH
)


# ============================================================
# TEMPORAL WINDOWING
# ============================================================

print("\nCreating training sequences...")

X_train, y_train_state = create_sequences(
    train_states_scaled,
    sequence_length=SEQUENCE_LENGTH
)


print("\nCreating validation sequences...")

X_val, y_val_state = create_sequences(
    val_states_scaled,
    sequence_length=SEQUENCE_LENGTH
)


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# For sequence [0:10], target label belongs to state index 10.
# Therefore labels must start from SEQUENCE_LENGTH.
# ------------------------------------------------------------

y_train_attack = train_labels[
    SEQUENCE_LENGTH:
]

y_val_attack = val_labels[
    SEQUENCE_LENGTH:
]


# ============================================================
# SAFETY CHECK
# ============================================================

print("\nSequence shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train_state:",
    y_train_state.shape
)

print(
    "y_train_attack:",
    y_train_attack.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "y_val_state:",
    y_val_state.shape
)

print(
    "y_val_attack:",
    y_val_attack.shape
)


if len(y_train_attack) != len(X_train):

    raise ValueError(
        "Training label alignment error."
    )


if len(y_val_attack) != len(X_val):

    raise ValueError(
        "Validation label alignment error."
    )


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train_state),
    torch.tensor(
        y_train_attack
    ).unsqueeze(1)
)


val_dataset = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(y_val_state),
    torch.tensor(
        y_val_attack
    ).unsqueeze(1)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = LSTMWorldModel(
    input_size=len(STATE_FEATURES),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(
    DEVICE
)


print("\nModel created.")

print(model)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

state_criterion = nn.SmoothL1Loss()

attack_criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float(
    "inf"
)

patience_counter = 0


print("\n")
print("=" * 80)
print("STARTING LSTM WORLD MODEL TRAINING")
print("=" * 80)


for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0

    total_train_state_loss = 0.0

    total_train_attack_loss = 0.0

    train_batches = 0


    for (
        X_batch,
        y_state_batch,
        y_attack_batch
    ) in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_state_batch = y_state_batch.to(
            DEVICE
        )

        y_attack_batch = y_attack_batch.to(
            DEVICE
        )

        optimizer.zero_grad()


        # Forward
        predicted_state, attack_logits = model(
            X_batch
        )


        # State loss
        state_loss = state_criterion(
            predicted_state,
            y_state_batch
        )


        # Attack loss
        attack_loss = attack_criterion(
            attack_logits,
            y_attack_batch
        )


        # Combined loss
        loss = (
            STATE_LOSS_WEIGHT
            * state_loss
            +
            ATTACK_LOSS_WEIGHT
            * attack_loss
        )


        # Backpropagation
        loss.backward()


        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )


        optimizer.step()


        total_train_loss += (
            loss.item()
        )

        total_train_state_loss += (
            state_loss.item()
        )

        total_train_attack_loss += (
            attack_loss.item()
        )

        train_batches += 1


    avg_train_loss = (
        total_train_loss
        / train_batches
    )

    avg_train_state_loss = (
        total_train_state_loss
        / train_batches
    )

    avg_train_attack_loss = (
        total_train_attack_loss
        / train_batches
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    total_val_loss = 0.0

    total_val_state_loss = 0.0

    total_val_attack_loss = 0.0

    val_batches = 0


    with torch.no_grad():

        for (
            X_batch,
            y_state_batch,
            y_attack_batch
        ) in val_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_state_batch = y_state_batch.to(
                DEVICE
            )

            y_attack_batch = y_attack_batch.to(
                DEVICE
            )


            predicted_state, attack_logits = model(
                X_batch
            )


            state_loss = state_criterion(
                predicted_state,
                y_state_batch
            )


            attack_loss = attack_criterion(
                attack_logits,
                y_attack_batch
            )


            loss = (
                STATE_LOSS_WEIGHT
                * state_loss
                +
                ATTACK_LOSS_WEIGHT
                * attack_loss
            )


            total_val_loss += (
                loss.item()
            )

            total_val_state_loss += (
                state_loss.item()
            )

            total_val_attack_loss += (
                attack_loss.item()
            )

            val_batches += 1


    avg_val_loss = (
        total_val_loss
        / val_batches
    )

    avg_val_state_loss = (
        total_val_state_loss
        / val_batches
    )

    avg_val_attack_loss = (
        total_val_attack_loss
        / val_batches
    )


    # --------------------------------------------------------
    # PRINT EPOCH RESULT
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss       : {avg_train_loss:.6f}"
    )

    print(
        f"Train State Loss : {avg_train_state_loss:.6f}"
    )

    print(
        f"Train Attack Loss: {avg_train_attack_loss:.6f}"
    )

    print(
        f"Val Loss         : {avg_val_loss:.6f}"
    )

    print(
        f"Val State Loss   : {avg_val_state_loss:.6f}"
    )

    print(
        f"Val Attack Loss  : {avg_val_attack_loss:.6f}"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        patience_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best model saved."
        )

    else:

        patience_counter += 1

        print(
            f"No improvement. "
            f"Patience: {patience_counter}/{PATIENCE}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Model saved:",
    MODEL_PATH
)

print(
    "Scaler saved:",
    SCALER_PATH
)

print("=" * 80)