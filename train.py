"""
train.py  —  Run this ONCE before starting the Flask app.
Preprocesses the dataset and saves all model artifacts to /models.

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "cellphones data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 55)
print("  Mobile Phone Recommender — Model Training")
print("=" * 55)

# ------------------------------------------------------------------
# Step 1: Load data
# ------------------------------------------------------------------
print("\n[1/7] Loading dataset & EDA...")
df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df)} phones, {df.shape[1]} columns.")
print("\n--- Exploratory Data Analysis (EDA) ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe(include='all').to_string())
print("---------------------------------------\n")

# ------------------------------------------------------------------
# Step 2: Parse dates & encode OS
# ------------------------------------------------------------------
print("[2/7] Parsing dates and encoding OS (Data Preprocessing)...")
df["release date"] = pd.to_datetime(df["release date"], format="%d/%m/%Y")
df["release_year"] = df["release date"].dt.year
df["os_encoded"]   = (df["operating system"] == "iOS").astype(int)

# ------------------------------------------------------------------
# Step 3: Check missing values
# ------------------------------------------------------------------
print("[3/7] Checking for missing values (Data Preprocessing)...")
FEATURE_COLS = [
    "internal memory", "RAM", "performance",
    "main camera",     "selfie camera",
    "battery size",    "screen size",
    "weight",          "price",
]
total_nulls = df[FEATURE_COLS].isnull().sum().sum()
if total_nulls:
    for col in FEATURE_COLS:
        n = df[col].isnull().sum()
        if n:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"      Filled {n} nulls in '{col}' with median.")
print(f"      Total nulls after fill: {df[FEATURE_COLS].isnull().sum().sum()}")

# ------------------------------------------------------------------
# Step 4: Normalize features
# ------------------------------------------------------------------
print("[4/7] Applying Min-Max normalization (Data Preprocessing)...")
df_norm = df.copy()
scaler  = MinMaxScaler()
df_norm[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
print("      Normalized:", FEATURE_COLS)

# ------------------------------------------------------------------
# Step 5: Define feature columns for recommender
# ------------------------------------------------------------------
SIMILARITY_COLS = [
    "internal memory", "RAM", "performance",
    "main camera",     "selfie camera",
    "battery size",    "screen size",
]
print("[5/7] Defining recommender features (similarity matrix removed as it is unwanted code)...")

# ------------------------------------------------------------------
# Step 6: Model Training & Evaluation (for OS prediction accuracy)
# ------------------------------------------------------------------
print("[6/7] Training OS prediction model for Evaluation Metrics...")
X = df_norm[FEATURE_COLS]
y = df_norm["os_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n      -> Model Accuracy: {acc * 100:.2f}%")
print("      -> Evaluation Metrics (Classification Report):")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------
# Step 7: Save artifacts
# ------------------------------------------------------------------
print("[7/7] Saving model artifacts with joblib...")

artifacts = {
    "phones_df.pkl":             df,
    "phones_df_normalized.pkl":  df_norm,
    "feature_cols.pkl":          SIMILARITY_COLS,
}

for fname, obj in artifacts.items():
    path = MODEL_DIR / fname
    joblib.dump(obj, path)
    size = os.path.getsize(path) / 1024
    print(f"      OK {fname:<35} ({size:.1f} KB)")

print("\nOK  Training complete! All models saved to /models")
print("    Start the web app with: python app.py")
print("=" * 55)
