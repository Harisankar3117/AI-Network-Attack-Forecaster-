import numpy as np
import pandas as pd


# ============================================================
# NETWORK STATE FEATURES
# ============================================================

STATE_FEATURES = [
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "ACK Flag Cnt",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Down/Up Ratio",
]


# ============================================================
# FEATURES THAT MUST BE NON-NEGATIVE
# ============================================================

NON_NEGATIVE_FEATURES = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "ACK Flag Cnt",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Down/Up Ratio",
]


# ============================================================
# CREATE NETWORK STATES
# ============================================================

def create_network_states(df):

    print("\nCreating network states...")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "create_network_states() expects a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in STATE_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing network features: {missing_features}"
        )

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    states = df[STATE_FEATURES].copy()

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    for column in STATE_FEATURES:
        states[column] = pd.to_numeric(
            states[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace Inf
    # --------------------------------------------------------

    states = states.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Remove NaN / Inf rows
    # --------------------------------------------------------

    before_cleaning = len(states)

    states = states.dropna()

    removed_nan = before_cleaning - len(states)

    # --------------------------------------------------------
    # Remove physically impossible negative values
    # --------------------------------------------------------

    invalid_negative_mask = (
        states[NON_NEGATIVE_FEATURES] < 0
    ).any(axis=1)

    removed_negative = invalid_negative_mask.sum()

    states = states.loc[
        ~invalid_negative_mask
    ]

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    network_states = states.to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("Network states created!")
    print("--------------------------------")
    print("Original samples    :", before_cleaning)
    print("NaN/Inf rows removed:", removed_nan)
    print("Negative rows removed:", removed_negative)
    print("Final samples       :", network_states.shape[0])
    print("Number of features  :", network_states.shape[1])
    print("State shape         :", network_states.shape)
    print("--------------------------------")

    return network_states


# ============================================================
# GET FEATURE NAMES
# ============================================================

def get_state_feature_names():

    return STATE_FEATURES.copy()