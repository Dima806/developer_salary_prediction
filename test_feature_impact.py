"""Test that changing input features actually changes predictions."""

from src.schema import SalaryInput
from src.infer import predict_salary, valid_categories


def test_years_experience_impact():
    """Test that changing years of experience changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 1: Total Years of Coding Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    # Test with different years of experience
    years_tests = [0, 2, 5, 10, 20]
    predictions = []

    for years in years_tests:
        input_data = SalaryInput(**base_input, years_code=years)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  Years: {years:2d} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    else:
        print(
            f"\n❌ FAIL: Only {unique_predictions}/{len(predictions)} unique predictions"
        )
        return False


def test_country_impact():
    """Test that changing country changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 2: Country Impact")
    print("=" * 70)

    base_input = {
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    # Test with different countries (select diverse ones)
    test_countries = [
        "United States of America",
        "Germany",
        "India",
        "Brazil",
        "Poland",
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
        print(
            f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions"
        )
        print("   Duplicate salaries found - possible feature issue")
        return False


def test_education_impact():
    """Test that changing education level changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 3: Education Level Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
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
        print(
            f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions"
        )
        print("   Duplicate salaries found - possible feature issue")
        return False


def test_devtype_impact():
    """Test that changing developer type changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 4: Developer Type Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    # Test with different developer types (using actual values from trained model)
    test_devtypes = [
        "Developer, front-end",
        "Developer, back-end",
        "Developer, full-stack",
        "Data scientist",
        "Engineering manager",
        "DevOps engineer or professional",
    ]

    # Filter to only developer types that exist in valid categories
    test_devtypes = [d for d in test_devtypes if d in valid_categories["DevType"]]

    predictions = []
    for devtype in test_devtypes:
        input_data = SalaryInput(**base_input, dev_type=devtype)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  Dev Type: {devtype[:50]:50s} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    elif unique_predictions == 1:
        print(f"\n❌ FAIL: All predictions are IDENTICAL (${predictions[0]:,.2f})")
        print("   This indicates the model is NOT using developer type as a feature!")
        return False
    else:
        print(
            f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions"
        )
        print("   Duplicate salaries found - possible feature issue")
        return False


def test_industry_impact():
    """Test that changing industry changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 5: Industry Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    # Test with different industries (using actual values from trained model)
    test_industries = [
        "Software Development",
        "Fintech",
        "Banking/Financial Services",
        "Healthcare",
        "Manufacturing",
        "Government",
    ]

    # Filter to only industries that exist in valid categories
    test_industries = [i for i in test_industries if i in valid_categories["Industry"]]

    predictions = []
    for industry in test_industries:
        input_data = SalaryInput(**base_input, industry=industry)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  Industry: {industry[:50]:50s} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    elif unique_predictions == 1:
        print(f"\n❌ FAIL: All predictions are IDENTICAL (${predictions[0]:,.2f})")
        print("   This indicates the model is NOT using industry as a feature!")
        return False
    else:
        print(
            f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions"
        )
        print("   Duplicate salaries found - possible feature issue")
        return False


def test_age_impact():
    """Test that changing age changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 6: Age Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "ic_or_pm": "Individual contributor",
    }

    # Test with different age ranges (using actual values from trained model)
    test_ages = [
        "18-24 years old",
        "25-34 years old",
        "35-44 years old",
        "45-54 years old",
        "55-64 years old",
    ]

    # Filter to only ages that exist in valid categories
    test_ages = [a for a in test_ages if a in valid_categories["Age"]]

    predictions = []
    for age in test_ages:
        input_data = SalaryInput(**base_input, age=age)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  Age: {age[:50]:50s} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    elif unique_predictions == 1:
        print(f"\n❌ FAIL: All predictions are IDENTICAL (${predictions[0]:,.2f})")
        print("   This indicates the model is NOT using age as a feature!")
        return False
    else:
        print(
            f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions"
        )
        print("   Duplicate salaries found - possible feature issue")
        return False


def test_work_exp_impact():
    """Test that changing years of work experience changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 7: Work Experience Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "years_code": 10.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    # Test with different years of work experience
    work_exp_tests = [0, 1, 3, 5, 10, 20]
    predictions = []

    for work_exp in work_exp_tests:
        input_data = SalaryInput(**base_input, work_exp=work_exp)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  Work Exp: {work_exp:2d} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    elif unique_predictions == 1:
        print(f"\n❌ FAIL: All predictions are IDENTICAL (${predictions[0]:,.2f})")
        print("   This indicates the model is NOT using work experience as a feature!")
        return False
    else:
        print(
            f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions"
        )
        print("   Duplicate salaries found - possible feature issue")
        return False


def test_icorpm_impact():
    """Test that changing IC or PM changes prediction."""
    print("\n" + "=" * 70)
    print("TEST 8: IC or PM Impact")
    print("=" * 70)

    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
    }

    # Test with different IC/PM values (using actual values from trained model)
    test_icorpm = [
        "Individual contributor",
        "People manager",
    ]

    # Filter to only values that exist in valid categories
    test_icorpm = [v for v in test_icorpm if v in valid_categories["ICorPM"]]

    predictions = []
    for icorpm in test_icorpm:
        input_data = SalaryInput(**base_input, ic_or_pm=icorpm)
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(f"  IC/PM: {icorpm[:50]:50s} -> Salary: ${salary:,.2f}")

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} predictions are different")
        return True
    elif unique_predictions == 1:
        print(f"\n❌ FAIL: All predictions are IDENTICAL (${predictions[0]:,.2f})")
        print("   This indicates the model is NOT using IC/PM as a feature!")
        return False
    else:
        print(
            f"\n⚠️  PARTIAL: Only {unique_predictions}/{len(predictions)} unique predictions"
        )
        print("   Duplicate salaries found - possible feature issue")
        return False


def test_combined_features():
    """Test that combining different features produces expected variations."""
    print("\n" + "=" * 70)
    print("TEST 9: Combined Feature Variations")
    print("=" * 70)

    # Create diverse combinations (using actual values from trained model)
    test_cases = [
        (
            "India",
            2,
            1,
            "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
            "Developer, back-end",
            "Software Development",
            "18-24 years old",
            "Individual contributor",
        ),
        (
            "Germany",
            5,
            3,
            "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
            "Developer, full-stack",
            "Manufacturing",
            "25-34 years old",
            "Individual contributor",
        ),
        (
            "United States of America",
            10,
            8,
            "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
            "Engineering manager",
            "Fintech",
            "35-44 years old",
            "People manager",
        ),
        (
            "Poland",
            15,
            12,
            "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
            "Developer, front-end",
            "Healthcare",
            "45-54 years old",
            "Individual contributor",
        ),
        (
            "Brazil",
            5,
            3,
            "Some college/university study without earning a degree",
            "DevOps engineer or professional",
            "Government",
            "25-34 years old",
            "Individual contributor",
        ),
    ]

    predictions = []
    for (
        country,
        years,
        work_exp,
        education,
        devtype,
        industry,
        age,
        icorpm,
    ) in test_cases:
        # Skip if not in valid categories
        if (
            country not in valid_categories["Country"]
            or education not in valid_categories["EdLevel"]
            or devtype not in valid_categories["DevType"]
            or industry not in valid_categories["Industry"]
            or age not in valid_categories["Age"]
            or icorpm not in valid_categories["ICorPM"]
        ):
            continue

        input_data = SalaryInput(
            country=country,
            years_code=years,
            work_exp=work_exp,
            education_level=education,
            dev_type=devtype,
            industry=industry,
            age=age,
            ic_or_pm=icorpm,
        )
        salary = predict_salary(input_data)
        predictions.append(salary)
        print(
            f"  {country[:15]:15s} | {years:2d}y | {work_exp:2d}w | {education[:25]:25s} | {devtype[:25]:25s} | {industry[:20]:20s} | {age[:15]:15s} | {icorpm[:5]:5s} -> ${salary:,.2f}"
        )

    # Check if predictions are different
    unique_predictions = len(set(predictions))
    if unique_predictions == len(predictions):
        print(f"\n✅ PASS: All {len(predictions)} combined predictions are different")
        return True
    else:
        print(f"\n⚠️  Only {unique_predictions}/{len(predictions)} unique predictions")
        print("   Some combinations produce identical salaries")
        return False


def print_feature_analysis():
    """Analyze which features the model is actually using."""
    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)

    from src.infer import feature_columns

    print(f"\nTotal features in model: {len(feature_columns)}")

    # Count by type
    country_features = [f for f in feature_columns if f.startswith("Country_")]
    edlevel_features = [f for f in feature_columns if f.startswith("EdLevel_")]
    devtype_features = [f for f in feature_columns if f.startswith("DevType_")]
    industry_features = [f for f in feature_columns if f.startswith("Industry_")]
    age_features = [f for f in feature_columns if f.startswith("Age_")]
    icorpm_features = [f for f in feature_columns if f.startswith("ICorPM_")]
    numeric_features = [
        f
        for f in feature_columns
        if not f.startswith(
            ("Country_", "EdLevel_", "DevType_", "Industry_", "Age_", "ICorPM_")
        )
    ]

    print(f"  - Numeric features: {len(numeric_features)} -> {numeric_features}")
    print(f"  - Country features: {len(country_features)}")
    print(f"  - Education features: {len(edlevel_features)}")
    print(f"  - DevType features: {len(devtype_features)}")
    print(f"  - Industry features: {len(industry_features)}")
    print(f"  - Age features: {len(age_features)}")
    print(f"  - ICorPM features: {len(icorpm_features)}")

    if len(country_features) > 0:
        print("\nSample country features:")
        for feat in country_features[:5]:
            print(f"    - {feat}")

    if len(edlevel_features) > 0:
        print("\nSample education features:")
        for feat in edlevel_features[:5]:
            print(f"    - {feat}")

    if len(devtype_features) > 0:
        print("\nSample developer type features:")
        for feat in devtype_features[:5]:
            print(f"    - {feat}")

    if len(industry_features) > 0:
        print("\nSample industry features:")
        for feat in industry_features[:5]:
            print(f"    - {feat}")

    if len(age_features) > 0:
        print("\nSample age features:")
        for feat in age_features[:5]:
            print(f"    - {feat}")

    if len(icorpm_features) > 0:
        print("\nSample IC/PM features:")
        for feat in icorpm_features[:5]:
            print(f"    - {feat}")

    # Check if there are any features at all
    if len(country_features) == 0:
        print("\n⚠️  WARNING: No country features found!")
    if len(edlevel_features) == 0:
        print("\n⚠️  WARNING: No education features found!")
    if len(devtype_features) == 0:
        print("\n⚠️  WARNING: No developer type features found!")
    if len(industry_features) == 0:
        print("\n⚠️  WARNING: No industry features found!")
    if len(age_features) == 0:
        print("\n⚠️  WARNING: No age features found!")
    if len(icorpm_features) == 0:
        print("\n⚠️  WARNING: No IC/PM features found!")


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
        "Years of Coding": test_years_experience_impact(),
        "Country": test_country_impact(),
        "Education Level": test_education_impact(),
        "Developer Type": test_devtype_impact(),
        "Industry": test_industry_impact(),
        "Age": test_age_impact(),
        "Work Experience": test_work_exp_impact(),
        "IC or PM": test_icorpm_impact(),
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
        print(
            "\n⚠️  Some tests failed. The model may not be using all features properly."
        )
        print(
            "   This indicates potential training-testing skew or feature engineering issues."
        )


if __name__ == "__main__":
    main()
