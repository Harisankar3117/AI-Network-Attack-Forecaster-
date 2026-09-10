from data_loader import load_dataset
from data_cleaner import clean_dataset
import sys

sys.path.append("../features")

from feature_engineering import create_features


file_path = "data/sample/sample_traffic.csv"

df = load_dataset(file_path)

df = clean_dataset(df)

features = create_features(df)

print("\nFinal Features:")
print(features)