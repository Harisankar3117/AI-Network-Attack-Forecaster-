import pandas as pd


file_path = (
    "data/processed/CIC-IDS2018/"
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
)


features = [
    "Flow Duration",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Pkt Len Mean",
    "Pkt Len Std",
]


print("Loading dataset...")

df = pd.read_csv(
    file_path,
    usecols=features,
    nrows=10000
)

print("\nFeature statistics:")
print("=" * 70)

print(df.describe().T)

print("\nNegative values:")
print("=" * 70)

for column in features:

    numeric = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    negative_count = (numeric < 0).sum()

    print(
        f"{column:20} : "
        f"{negative_count} negative values"
    )


print("\nDiagnostic test completed!")