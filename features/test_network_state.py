import pandas as pd

from features.network_state import (
    create_network_states,
    get_state_feature_names
)


# Load one cleaned dataset
file_path = (
    "data/processed/CIC-IDS2018/"
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
)

print("Loading cleaned dataset...")

df = pd.read_csv(
    file_path,
    nrows=1000
)

print("Rows loaded:", len(df))
print("Columns:", len(df.columns))


# Create network states
states = create_network_states(df)


# Show feature names
print("\nNetwork State Features:")
for i, feature in enumerate(
    get_state_feature_names(),
    start=1
):
    print(f"{i}. {feature}")


# Show first state
print("\nFirst Network State:")
print(states[0])


print("\nTest completed successfully!")