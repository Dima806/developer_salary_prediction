"""Streamlit web app for salary prediction."""

import streamlit as st

from src.infer import predict_salary, get_local_currency, valid_categories
from src.schema import SalaryInput

# Page configuration
st.set_page_config(
    page_title="Developer Salary Predictor",
    page_icon="💰",
    layout="centered",
)

# Title and description
st.title("💰 Developer Salary Predictor")
st.write(
    "Predict developer salaries based on Stack Overflow Developer Survey data using a simple ML model."
)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write(
        """
        This app uses an XGBoost (gradient boosting) model trained on Stack Overflow
        Developer Survey data to predict annual salaries based on:
        - Country
        - Total years of coding experience (including education)
        - Years of professional work experience
        - Education level
        - Developer type
        - Industry
        - Age
        - Individual contributor or people manager
        - Organization size
        """
    )
    st.info("💡 Tip: Results are estimates based on survey averages.")

    st.divider()
    st.subheader("Model Coverage")
    st.write(f"**Countries:** {len(valid_categories['Country'])} available")
    st.write(f"**Education Levels:** {len(valid_categories['EdLevel'])} available")
    st.write(f"**Developer Types:** {len(valid_categories['DevType'])} available")
    st.write(f"**Industries:** {len(valid_categories['Industry'])} available")
    st.write(f"**Age Ranges:** {len(valid_categories['Age'])} available")
    st.write(f"**IC/PM Roles:** {len(valid_categories['ICorPM'])} available")
    st.write(f"**Org Sizes:** {len(valid_categories['OrgSize'])} available")
    st.caption("Only values from the training data are shown in the dropdowns.")

# Main input form
st.header("Enter Developer Information")

col1, col2 = st.columns(2)

# Get valid categories from training
valid_countries = valid_categories["Country"]
valid_education_levels = valid_categories["EdLevel"]
valid_dev_types = valid_categories["DevType"]
valid_industries = valid_categories["Industry"]
valid_ages = valid_categories["Age"]
valid_icorpm = valid_categories["ICorPM"]
valid_org_sizes = valid_categories["OrgSize"]

# Set default values (if available)
default_country = (
    "United States of America"
    if "United States of America" in valid_countries
    else valid_countries[0]
)
default_education = (
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)"
    if "Bachelor's degree (B.A., B.S., B.Eng., etc.)" in valid_education_levels
    else valid_education_levels[0]
)
default_dev_type = (
    "Developer, back-end"
    if "Developer, back-end" in valid_dev_types
    else valid_dev_types[0]
)
default_industry = (
    "Software Development"
    if "Software Development" in valid_industries
    else valid_industries[0]
)
default_age = "25-34 years old" if "25-34 years old" in valid_ages else valid_ages[0]
default_icorpm = (
    "Individual contributor"
    if "Individual contributor" in valid_icorpm
    else valid_icorpm[0]
)
default_org_size = (
    "20 to 99 employees"
    if "20 to 99 employees" in valid_org_sizes
    else valid_org_sizes[0]
)

with col1:
    country = st.selectbox(
        "Country",
        options=valid_countries,
        index=valid_countries.index(default_country),
        help="Developer's country of residence (only countries from training data)",
    )

    years = st.number_input(
        "Years of Coding (Total)",
        min_value=0,
        max_value=50,
        value=15,
        step=1,
        help="Including any education, how many years have you been coding in total?",
    )

    work_exp = st.number_input(
        "Years of Professional Work Experience",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="How many years of professional work experience do you have?",
    )

with col2:
    education = st.selectbox(
        "Education Level",
        options=valid_education_levels,
        index=valid_education_levels.index(default_education),
        help="Highest level of education completed (only levels from training data)",
    )

    dev_type = st.selectbox(
        "Developer Type",
        options=valid_dev_types,
        index=valid_dev_types.index(default_dev_type),
        help="Primary developer role (only types from training data)",
    )

industry = st.selectbox(
    "Industry",
    options=valid_industries,
    index=valid_industries.index(default_industry),
    help="Industry the developer works in (only industries from training data)",
)

age = st.selectbox(
    "Age",
    options=valid_ages,
    index=valid_ages.index(default_age),
    help="Developer's age range",
)

ic_or_pm = st.selectbox(
    "Individual Contributor or People Manager",
    options=valid_icorpm,
    index=valid_icorpm.index(default_icorpm),
    help="Are you an individual contributor or people manager?",
)

org_size = st.selectbox(
    "Organization Size",
    options=valid_org_sizes,
    index=valid_org_sizes.index(default_org_size),
    help="Approximate number of employees at the developer's company",
)

# Prediction button
if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
    try:
        # Create input model
        input_data = SalaryInput(
            country=country,
            years_code=years,
            work_exp=work_exp,
            education_level=education,
            dev_type=dev_type,
            industry=industry,
            age=age,
            ic_or_pm=ic_or_pm,
            org_size=org_size,
        )

        # Make prediction
        with st.spinner("Calculating prediction..."):
            salary = predict_salary(input_data)

        # Display result
        st.success("Prediction Complete!")

        # Show USD and local currency side by side
        local = get_local_currency(country, salary)
        if local and local["code"] != "USD":
            col_usd, col_local = st.columns(2)
            with col_usd:
                st.metric(
                    label="Estimated Annual Salary (USD)",
                    value=f"${salary:,.0f}",
                    help="Predicted annual compensation in USD",
                )
            with col_local:
                st.metric(
                    label=f"Estimated Annual Salary ({local['code']})",
                    value=f"{local['salary_local']:,.0f} {local['code']}",
                    help=f"Converted using survey-derived rate: 1 USD = {local['rate']} {local['code']} ({local['name']})",
                )
        else:
            st.metric(
                label="Estimated Annual Salary",
                value=f"${salary:,.0f}",
                help="Predicted annual compensation in USD",
            )

    except FileNotFoundError:
        st.error(
            """
            ❌ Model not found! Please train the model first by running:
            ```
            python src/train.py
            ```
            """
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.divider()
st.caption(
    "Built with Streamlit • Data from Stack Overflow Developer Survey • Model: XGBoost"
)
