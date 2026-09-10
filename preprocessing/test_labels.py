from data_loader import load_dataset
from data_cleaner import clean_dataset
import sys

sys.path.append("../features")

from labels import create_labels


file_path = "data/sample/sample_traffic.csv"

# Load dataset
df = load_dataset(file_path)

# Clean dataset
df = clean_dataset(df)

# Create labels
df = create_labels(df)

print("\nLabels:")
print(df[["label", "label_encoded"]])