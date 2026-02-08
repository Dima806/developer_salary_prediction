"""Test that changing input features actually changes predictions."""

from src.schema import SalaryInput
from src.infer import predict_salary, valid_categories


def test_years_experience_impact():
    """Test that changing years of experience changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 1: Years of Professional Coding Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
    }

    # Test with different years of experience
    years_tests = [0, 2, 5, 10, 20]
    predictions = []

    for years in years_tests:
        input_data = SalaryInput(**base_input, years_code_pro=years)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  Years: {years:2d} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    else:
        print(f"\n❌ FAIL: Only {unique_predictions}/{len(predictions)} unique predictions")
        return False


def test_country_impact():
    """Test that changing country changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 2: Country Impact")
    print("=" * 70)

    base_input = {
        "years_code_pro": 5.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
    }

    # Test with different countries (select diverse ones)
    test_countries = [
        "United States of America",
        "Germany",
        "India",
        "Brazil",
        "Poland"
    ]

    # Filter to only countries that exist in valid categories
    test_countries = [c for c in test_countries if c in valid_categories["Country"]]

    predictions = []
    for country in test_countries:
        input_data = SalaryInput(**base_input, country=country)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  Country: {country:40s} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    elif unique_predictions == 1:
        print(f"\n❌ FAIL: All predictions are IDENTICAL (${predictions[0]:,.2f})")
        print("   This indicates the model is NOT using country as a feature!")
        return False
    else:
        print(f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions")
        print(f"   Duplicate salaries found - possible feature issue")
        return False


def test_education_impact():
    """Test that changing education level changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 3: Education Level Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "years_code_pro": 5.0,
    }

    # Test with different education levels
    test_education = [
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
        "Some college/university study without earning a degree",
        "Associate degree (A.A., A.S., etc.)",
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
    ]

    # Filter to only education levels that exist in valid categories
    test_education = [e for e in test_education if e in valid_categories["EdLevel"]]

    predictions = []
    for education in test_education:
        input_data = SalaryInput(**base_input, education_level=education)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  Education: {education[:50]:50s} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    elif unique_predictions == 1:
        print(f"\n❌ FAIL: All predictions are IDENTICAL (${predictions[0]:,.2f})")
        print("   This indicates the model is NOT using education level as a feature!")
        return False
    else:
        print(f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions")
        print(f"   Duplicate salaries found - possible feature issue")
        return False


def test_combined_features():
    """Test that combining different features produces expected variations."""
    print("\n" + "=" * 70)
    print("TEST 4: Combined Feature Variations")
    print("=" * 70)

    # Create diverse combinations
    test_cases = [
        ("India", 2, "Bachelor's degree (B.A., B.S., B.Eng., etc.)"),
        ("Germany", 5, "Master's degree (M.A., M.S., M.Eng., MBA, etc.)"),
        ("United States of America", 10, "Master's degree (M.A., M.S., M.Eng., MBA, etc.)"),
        ("Poland", 15, "Bachelor's degree (B.A., B.S., B.Eng., etc.)"),
        ("Brazil", 5, "Some college/university study without earning a degree"),
    ]

    predictions = []
    for country, years, education in test_cases:
        # Skip if not in valid categories
        if country not in valid_categories["Country"] or education not in valid_categories["EdLevel"]:
            continue

        input_data = SalaryInput(
            country=country,
            years_code_pro=years,
            education_level=education
        )
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  {country[:20]:20s} | {years:2d}y | {education[:30]:30s} -> ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} combined predictions are different")
        return True
    else:
        print(f"\n⚠️  Only {unique_predictions}/{len(predictions)} unique predictions")
        print(f"   Some combinations produce identical salaries")
        return False


def print_feature_analysis():
    """Analyze which features the model is actually using."""
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    from src.infer import feature_columns

    print(f"\nTotal features in model: {len(feature_columns)}")

    # Count by type
    country_features = [f for f in feature_columns if f.startswith('Country_')]
    edlevel_features = [f for f in feature_columns if f.startswith('EdLevel_')]
    numeric_features = [f for f in feature_columns if not f.startswith(('Country_', 'EdLevel_'))]

    print(f"  - Numeric features: {len(numeric_features)} -> {numeric_features}")
    print(f"  - Country features: {len(country_features)}")
    print(f"  - Education features: {len(edlevel_features)}")

    if len(country_features) > 0:
        print(f"\nSample country features:")
        for feat in country_features[:5]:
            print(f"    - {feat}")

    if len(edlevel_features) > 0:
        print(f"\nSample education features:")
        for feat in edlevel_features[:5]:
            print(f"    - {feat}")

    # Check if there are any features at all
    if len(country_features) == 0:
        print("\n⚠️  WARNING: No country features found!")
    if len(edlevel_features) == 0:
        print("\n⚠️  WARNING: No education features found!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FEATURE IMPACT TESTS")
    print("Testing if changing inputs actually changes predictions")
    print("=" * 70)

    # First, analyze what features exist
    print_feature_analysis()

    # Run all tests
    results = {
        "Years of Experience": test_years_experience_impact(),
        "Country": test_country_impact(),
        "Education Level": test_education_impact(),
        "Combined Features": test_combined_features(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! The model is using all features correctly.")
    else:
        print("\n⚠️  Some tests failed. The model may not be using all features properly.")
        print("   This indicates potential training-testing skew or feature engineering issues.")


if __name__ == "__main__":
    main()
