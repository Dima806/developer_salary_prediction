"""Test that changing input features actually changes predictions."""

from src.infer import predict_salary, valid_categories
from src.schema import SalaryInput


def test_years_experience_impact():
    """Test that changing years of experience changes prediction."""
    base_input = {
        "country": "United States of America",
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    years_tests = [0, 2, 5, 10, 20]
    predictions = []
    for years in years_tests:
        input_data = SalaryInput(**base_input, years_code=years)
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) == len(predictions), (
        f"Expected {len(predictions)} unique predictions, got {len(set(predictions))}"
    )


def test_country_impact():
    """Test that changing country changes prediction."""
    base_input = {
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    test_countries = [
        c
        for c in [
            "United States of America",
            "Germany",
            "India",
            "Brazil",
            "Poland",
        ]
        if c in valid_categories["Country"]
    ]

    predictions = []
    for country in test_countries:
        input_data = SalaryInput(**base_input, country=country)
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) == len(predictions), (
        f"Expected {len(predictions)} unique predictions, got {len(set(predictions))}"
    )


def test_education_impact():
    """Test that changing education level changes prediction."""
    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    test_education = [
        e
        for e in [
            "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
            "Some college/university study without earning a degree",
            "Associate degree (A.A., A.S., etc.)",
            "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
            "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
            "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
        ]
        if e in valid_categories["EdLevel"]
    ]

    predictions = []
    for education in test_education:
        input_data = SalaryInput(**base_input, education_level=education)
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) == len(predictions), (
        f"Expected {len(predictions)} unique predictions, got {len(set(predictions))}"
    )


def test_devtype_impact():
    """Test that changing developer type changes prediction."""
    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    test_devtypes = [
        d
        for d in [
            "Developer, front-end",
            "Developer, back-end",
            "Developer, full-stack",
            "Data scientist",
            "Engineering manager",
            "DevOps engineer or professional",
        ]
        if d in valid_categories["DevType"]
    ]

    predictions = []
    for devtype in test_devtypes:
        input_data = SalaryInput(**base_input, dev_type=devtype)
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) == len(predictions), (
        f"Expected {len(predictions)} unique predictions, got {len(set(predictions))}"
    )


def test_industry_impact():
    """Test that changing industry changes prediction."""
    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    test_industries = [
        i
        for i in [
            "Software Development",
            "Fintech",
            "Banking/Financial Services",
            "Healthcare",
            "Manufacturing",
            "Government",
        ]
        if i in valid_categories["Industry"]
    ]

    predictions = []
    for industry in test_industries:
        input_data = SalaryInput(**base_input, industry=industry)
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) == len(predictions), (
        f"Expected {len(predictions)} unique predictions, got {len(set(predictions))}"
    )


def test_age_impact():
    """Test that changing age changes prediction."""
    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "ic_or_pm": "Individual contributor",
    }

    test_ages = [
        a
        for a in [
            "18-24 years old",
            "25-34 years old",
            "35-44 years old",
            "45-54 years old",
            "55-64 years old",
        ]
        if a in valid_categories["Age"]
    ]

    predictions = []
    for age in test_ages:
        input_data = SalaryInput(**base_input, age=age)
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) == len(predictions), (
        f"Expected {len(predictions)} unique predictions, got {len(set(predictions))}"
    )


def test_work_exp_impact():
    """Test that changing years of work experience changes prediction."""
    base_input = {
        "country": "United States of America",
        "years_code": 10.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
        "ic_or_pm": "Individual contributor",
    }

    work_exp_tests = [0, 1, 3, 5, 10, 20]
    predictions = []
    for work_exp in work_exp_tests:
        input_data = SalaryInput(**base_input, work_exp=work_exp)
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) >= len(predictions) - 1, (
        f"Expected at least {len(predictions) - 1} unique predictions, got {len(set(predictions))}"
    )


def test_icorpm_impact():
    """Test that changing IC or PM changes prediction."""
    base_input = {
        "country": "United States of America",
        "years_code": 5.0,
        "work_exp": 3.0,
        "education_level": "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        "dev_type": "Developer, full-stack",
        "industry": "Software Development",
        "age": "25-34 years old",
    }

    test_icorpm = [
        v
        for v in ["Individual contributor", "People manager"]
        if v in valid_categories["ICorPM"]
    ]

    predictions = []
    for icorpm in test_icorpm:
        input_data = SalaryInput(**base_input, ic_or_pm=icorpm)
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) == len(predictions), (
        f"Expected {len(predictions)} unique predictions, got {len(set(predictions))}"
    )


def test_combined_features():
    """Test that combining different features produces expected variations."""
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
        predictions.append(predict_salary(input_data))

    assert len(set(predictions)) == len(predictions), (
        f"Expected {len(predictions)} unique predictions, got {len(set(predictions))}"
    )
