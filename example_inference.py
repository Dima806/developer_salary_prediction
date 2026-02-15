"""Example script showing how to use the salary prediction model programmatically."""

from src.schema import SalaryInput
from src.infer import predict_salary


def main():
    """Run sample predictions with different input parameters."""

    print("=" * 60)
    print("Developer Salary Prediction - Sample Inference")
    print("=" * 60)

    # Example 1: Default parameters (same as Streamlit app defaults)
    print("\n📊 Example 1: Default Parameters")
    print("-" * 60)

    input_data_1 = SalaryInput(
        country="United States of America",
        years_code=5.0,
        work_exp=3.0,
        education_level="Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        dev_type="Developer, full-stack",
        industry="Software Development",
        age="25-34 years old",
        ic_or_pm="Individual contributor",
    )

    print(f"Country: {input_data_1.country}")
    print(f"Years of Coding (Total): {input_data_1.years_code}")
    print(f"Work Experience: {input_data_1.work_exp}")
    print(f"Education Level: {input_data_1.education_level}")
    print(f"Developer Type: {input_data_1.dev_type}")
    print(f"Industry: {input_data_1.industry}")
    print(f"Age: {input_data_1.age}")
    print(f"IC or PM: {input_data_1.ic_or_pm}")

    salary_1 = predict_salary(input_data_1)
    print(f"💰 Predicted Salary: ${salary_1:,.2f} USD/year")

    # Example 2: Junior developer
    print("\n📊 Example 2: Junior Developer")
    print("-" * 60)

    input_data_2 = SalaryInput(
        country="United States of America",
        years_code=2.0,
        work_exp=1.0,
        education_level="Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
        dev_type="Developer, front-end",
        industry="Fintech",
        age="18-24 years old",
        ic_or_pm="Individual contributor",
    )

    print(f"Country: {input_data_2.country}")
    print(f"Years of Coding (Total): {input_data_2.years_code}")
    print(f"Work Experience: {input_data_2.work_exp}")
    print(f"Education Level: {input_data_2.education_level}")
    print(f"Developer Type: {input_data_2.dev_type}")
    print(f"Industry: {input_data_2.industry}")
    print(f"Age: {input_data_2.age}")
    print(f"IC or PM: {input_data_2.ic_or_pm}")

    salary_2 = predict_salary(input_data_2)
    print(f"💰 Predicted Salary: ${salary_2:,.2f} USD/year")

    # Example 3: Senior developer with Master's degree
    print("\n📊 Example 3: Senior Developer")
    print("-" * 60)

    input_data_3 = SalaryInput(
        country="United States of America",
        years_code=10.0,
        work_exp=8.0,
        education_level="Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
        dev_type="Engineering manager",
        industry="Banking/Financial Services",
        age="35-44 years old",
        ic_or_pm="People manager",
    )

    print(f"Country: {input_data_3.country}")
    print(f"Years of Coding (Total): {input_data_3.years_code}")
    print(f"Work Experience: {input_data_3.work_exp}")
    print(f"Education Level: {input_data_3.education_level}")
    print(f"Developer Type: {input_data_3.dev_type}")
    print(f"Industry: {input_data_3.industry}")
    print(f"Age: {input_data_3.age}")
    print(f"IC or PM: {input_data_3.ic_or_pm}")

    salary_3 = predict_salary(input_data_3)
    print(f"💰 Predicted Salary: ${salary_3:,.2f} USD/year")

    # Example 4: Different country
    print("\n📊 Example 4: Different Country (Germany)")
    print("-" * 60)

    input_data_4 = SalaryInput(
        country="Germany",
        years_code=5.0,
        work_exp=3.0,
        education_level="Bachelor's degree (B.A., B.S., B.Eng., etc.)",
        dev_type="Developer, back-end",
        industry="Manufacturing",
        age="25-34 years old",
        ic_or_pm="Individual contributor",
    )

    print(f"Country: {input_data_4.country}")
    print(f"Years of Coding (Total): {input_data_4.years_code}")
    print(f"Work Experience: {input_data_4.work_exp}")
    print(f"Education Level: {input_data_4.education_level}")
    print(f"Developer Type: {input_data_4.dev_type}")
    print(f"Industry: {input_data_4.industry}")
    print(f"Age: {input_data_4.age}")
    print(f"IC or PM: {input_data_4.ic_or_pm}")

    salary_4 = predict_salary(input_data_4)
    print(f"💰 Predicted Salary: ${salary_4:,.2f} USD/year")

    print("\n" + "=" * 60)
    print("✅ All predictions completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("❌ Error: Model file not found!")
        print("Please train the model first by running:")
        print("  uv run python src/train.py")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
