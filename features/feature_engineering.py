import pandas as pd


def create_features(df):

    print("Starting feature engineering...")

    feature_columns = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "packet_count"
    ]

    # Select only features available in the dataset
    available_features = [
        col for col in feature_columns
        if col in df.columns
    ]

    features = df[available_features].copy()

    print("Features selected:")
    print(available_features)

    return features