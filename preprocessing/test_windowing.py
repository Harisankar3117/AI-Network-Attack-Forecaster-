import pandas as pd

from features.network_state import create_network_states
from features.windowing import create_sequences


# ============================================================
# LOAD CLEANED CIC-IDS2018 DATA
# ============================================================

file_path = (
    "data/processed/CIC-IDS2018/"
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
)

print("Loading cleaned CIC-IDS2018 dataset...")

# Only load 1000 rows for testing
df = pd.read_csv(
    file_path,
    nrows=1000
)

print("Rows loaded   :", len(df))
print("Columns loaded:", len(df.columns))


# ============================================================
# CREATE NETWORK STATES
# ============================================================

network_states = create_network_states(df)


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================

X, y = create_sequences(
    network_states,
    sequence_length=10
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nTemporal Windowing Result")
print("=" * 70)

print("Input sequence shape :", X.shape)
print("Target state shape   :", y.shape)

print("\nExpected format:")
print("X = [samples, sequence_length, features]")
print("y = [samples, features]")


# ============================================================
# SHOW FIRST SEQUENCE
# ============================================================

print("\nFirst Sequence Shape:")
print(X[0].shape)

print("\nFirst Target State:")
print(y[0])


print("\nTemporal windowing test completed successfully!")