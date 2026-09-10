from data_loader import load_dataset
from data_cleaner import clean_dataset
import sys

sys.path.append("../features")
sys.path.append("../models")

from feature_engineering import create_features
from labels import create_labels
from models.baseline import train_baseline, evaluate_baseline

from sklearn.model_selection import train_test_split


# Dataset path
file_path = "data/sample/sample_traffic.csv"


# 1. Load dataset
df = load_dataset(file_path)

# 2. Clean dataset
df = clean_dataset(df)

# 3. Create labels
df = create_labels(df)

# 4. Create features
features = create_features(df)

# 5. Create X and y
X = features
y = df["label_encoded"]


# 6. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))


# 7. Train baseline model
model = train_baseline(X_train, y_train)

# 8. Evaluate
evaluate_baseline(model, X_test, y_test)