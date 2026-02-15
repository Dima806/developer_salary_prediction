"""Diagnose why categorical features aren't affecting predictions."""

from src.preprocessing import prepare_features
import pandas as pd

# Create two inputs that differ ONLY in Country
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
        "Country": ["Germany"],  # Different!
        "YearsCode": [5.0],
        "EdLevel": ["Bachelor's degree (B.A., B.S., B.Eng., etc.)"],
        "DevType": ["Developer, full-stack"],
    }
)

print("=" * 70)
print("ENCODING DIAGNOSIS")
print("=" * 70)

# Process features
features1 = prepare_features(input1)
features2 = prepare_features(input2)

print("\nInput 1 (USA):")
print(f"  Shape: {features1.shape}")
print(f"  Columns: {list(features1.columns)}")
non_zero1 = [col for col in features1.columns if features1[col].iloc[0] != 0]
print(f"  Non-zero features ({len(non_zero1)}): {non_zero1}")

print("\nInput 2 (Germany):")
print(f"  Shape: {features2.shape}")
non_zero2 = [col for col in features2.columns if features2[col].iloc[0] != 0]
print(f"  Non-zero features ({len(non_zero2)}): {non_zero2}")

print(f"\nAre encoded features identical? {features1.equals(features2)}")

if features1.equals(features2):
    print("\n❌ PROBLEM: Different countries produce IDENTICAL encodings!")
    print("   This explains why categorical features don't affect predictions.")
else:
    print("\n✅ Encodings are different - categorical features should work.")

# Check what happens with Country specifically
print("\n" + "=" * 70)
print("COUNTRY ENCODING CHECK")
print("=" * 70)

# Test just Country encoding
test_countries = ["United States of America", "Germany", "India"]
for country in test_countries:
    test_df = pd.DataFrame(
        {
            "Country": [country],
            "YearsCode": [5.0],
            "EdLevel": ["Bachelor's degree (B.A., B.S., B.Eng., etc.)"],
            "DevType": ["Developer, full-stack"],
        }
    )
    encoded = prepare_features(test_df)
    country_cols = [col for col in encoded.columns if col.startswith("Country_")]
    non_zero_countries = [col for col in country_cols if encoded[col].iloc[0] != 0]
    print(f"{country:40s} -> {non_zero_countries}")
