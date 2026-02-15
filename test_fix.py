"""Test that the encoding fix works."""

# Force reload of modules
import sys

if "src.preprocessing" in sys.modules:
    del sys.modules["src.preprocessing"]
if "src.infer" in sys.modules:
    del sys.modules["src.infer"]

from src.preprocessing import prepare_features
import pandas as pd

# Create test inputs with different countries (values from valid_categories)
input1 = pd.DataFrame(
    {
        "Country": ["United States of America"],
        "YearsCode": [5.0],
        "EdLevel": ["Bachelor's degree (B.A., B.S., B.Eng., etc.)"],
        "DevType": ["Developer, full-stack"],
    }
)

input2 = pd.DataFrame(
    {
        "Country": ["Germany"],
        "YearsCode": [5.0],
        "EdLevel": ["Bachelor's degree (B.A., B.S., B.Eng., etc.)"],
        "DevType": ["Developer, full-stack"],
    }
)

print("Testing prepare_features with different countries...")
features1 = prepare_features(input1)
features2 = prepare_features(input2)

print(f"\nUSA features: {features1.shape}")
print(f"Columns: {list(features1.columns)[:10]}")

print(f"\nGermany features: {features2.shape}")
print(f"Columns: {list(features2.columns)[:10]}")

print(f"\nAre they different? {not features1.equals(features2)}")

if features1.shape[1] > 1:
    print("\n✅ SUCCESS: Categorical features are preserved!")
else:
    print("\n❌ FAIL: Still only has numeric features")
