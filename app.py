"""Streamlit web app for salary prediction."""

import streamlit as st

from src.infer import predict_salary
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

# Main input form
st.header("Enter Developer Information")

col1, col2 = st.columns(2)

with col1:
    country = st.text_input(
        "Country",
        value="United States",
        help="Developer's country of residence",
    )

    years = st.number_input(
        "Years of Professional Coding",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Years of professional coding experience",
    )

with col2:
    education = st.selectbox(
        "Education Level",
        [
            "Bachelor's degree",
            "Master's degree",
            "Some college/university study",
            "Associate degree",
            "Professional degree",
            "Other doctoral degree",
            "Secondary school",
            "Primary/elementary school",
            "Something else",
        ],
        help="Highest level of education completed",
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

    except FileNotFoundError as e:
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
