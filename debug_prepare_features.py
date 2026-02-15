"""Debug prepare_features step by step."""

import pandas as pd
from src.preprocessing import reduce_cardinality
import yaml
from pathlib import Path

# Load config
config_path = Path("config/model_parameters.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Create test input
df = pd.DataFrame(
    {
        "Country": ["United States of America"],
        "YearsCode": [5.0],
        "EdLevel": ["Bachelor's degree (B.A., B.S., B.Eng., etc.)"],
        "DevType": ["Developer, full-stack"],
    }
)

print("=" * 70)
print("STEP-BY-STEP DEBUGGING OF prepare_features()")
print("=" * 70)

print("\n1. Original input:")
print(f"   Columns: {list(df.columns)}")
print(f"   Values: {df.iloc[0].to_dict()}")

# Step 2: Copy
df_processed = df.copy()

# Step 3: Unicode normalization
for col in ["Country", "EdLevel", "DevType"]:
    if col in df_processed.columns:
        df_processed[col] = df_processed[col].str.replace("\u2019", "'", regex=False)

print("\n2. After unicode normalization:")
print(f"   Columns: {list(df_processed.columns)}")

# Step 4: Fill missing values
df_processed["YearsCode"] = df_processed["YearsCode"].fillna(0)
df_processed["Country"] = df_processed["Country"].fillna("Unknown")
df_processed["EdLevel"] = df_processed["EdLevel"].fillna("Unknown")
df_processed["DevType"] = df_processed["DevType"].fillna("Unknown")

print("\n3. After filling missing values:")
print(f"   Columns: {list(df_processed.columns)}")
print(f"   Country value: '{df_processed['Country'].iloc[0]}'")
print(f"   EdLevel value: '{df_processed['EdLevel'].iloc[0]}'")
print(f"   DevType value: '{df_processed['DevType'].iloc[0]}'")

# Step 5: Reduce cardinality
print("\n4. Before cardinality reduction:")
print(f"   Country value: '{df_processed['Country'].iloc[0]}'")
df_processed["Country"] = reduce_cardinality(df_processed["Country"])
print(f"   After Country reduction: '{df_processed['Country'].iloc[0]}'")

print(f"   EdLevel value: '{df_processed['EdLevel'].iloc[0]}'")
df_processed["EdLevel"] = reduce_cardinality(df_processed["EdLevel"])
print(f"   After EdLevel reduction: '{df_processed['EdLevel'].iloc[0]}'")

print(f"   DevType value: '{df_processed['DevType'].iloc[0]}'")
df_processed["DevType"] = reduce_cardinality(df_processed["DevType"])
print(f"   After DevType reduction: '{df_processed['DevType'].iloc[0]}'")

# Step 6: Select feature columns
feature_cols = ["Country", "YearsCode", "EdLevel", "DevType"]
df_features = df_processed[feature_cols]

print("\n5. After selecting feature columns:")
print(f"   Columns: {list(df_features.columns)}")
print(f"   Values: {df_features.iloc[0].to_dict()}")

# Step 7: One-hot encode
drop_first = config["features"]["encoding"]["drop_first"]
print(f"\n6. One-hot encoding with drop_first={drop_first}:")
df_encoded = pd.get_dummies(df_features, drop_first=drop_first)

print(f"   Result shape: {df_encoded.shape}")
print(f"   Result columns: {list(df_encoded.columns)}")
print(f"   Non-zero values: {df_encoded.columns[df_encoded.iloc[0] != 0].tolist()}")
