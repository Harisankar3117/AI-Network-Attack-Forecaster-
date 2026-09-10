import pandas as pd


def create_labels(df):

    print("Creating labels...")

    df = df.copy()

    df["label_encoded"] = (
        df["label"]
        .str.strip()
        .str.lower()
        .map({
            "normal": 0,
            "attack": 1
        })
    )

    print("Label mapping:")
    print("Normal → 0")
    print("Attack → 1")

    return df