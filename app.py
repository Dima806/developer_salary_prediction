"""Streamlit web app for salary prediction."""

import streamlit as st

from src.infer import predict_salary, valid_categories
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
        - Years of professional coding experience
        - Education level
        """
    )
    st.info("💡 Tip: Results are estimates based on survey averages.")

    st.divider()
    st.subheader("Model Coverage")
    st.write(f"**Countries:** {len(valid_categories['Country'])} available")
    st.write(f"**Education Levels:** {len(valid_categories['EdLevel'])} available")
    st.caption("Only values from the training data are shown in the dropdowns.")

# Main input form
st.header("Enter Developer Information")

col1, col2 = st.columns(2)

# Get valid categories from training
valid_countries = valid_categories["Country"]
valid_education_levels = valid_categories["EdLevel"]

# Set default values (if available)
default_country = "United States of America" if "United States of America" in valid_countries else valid_countries[0]
default_education = "Bachelor's degree (B.A., B.S., B.Eng., etc.)" if "Bachelor's degree (B.A., B.S., B.Eng., etc.)" in valid_education_levels else valid_education_levels[0]

with col1:
    country = st.selectbox(
        "Country",
        options=valid_countries,
        index=valid_countries.index(default_country),
        help="Developer's country of residence (only countries from training data)",
    )

    years = st.number_input(
        "Years of Professional Coding",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Years of professional coding experience",
    )

with col2:
    education = st.selectbox(
        "Education Level",
        options=valid_education_levels,
        index=valid_education_levels.index(default_education),
        help="Highest level of education completed (only levels from training data)",
    )

# Prediction button
if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
    try:
        # Create input model
        input_data = SalaryInput(
            country=country,
            years_code_pro=years,
            education_level=education,
        )

        # Make prediction
        with st.spinner("Calculating prediction..."):
            salary = predict_salary(input_data)

        # Display result
        st.success("Prediction Complete!")
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
