"""Streamlit web app for salary prediction."""

import altair as alt
import pandas as pd
import streamlit as st

from src.infer import (
    predict_salaries_with_intervals,
    get_local_currency,
    valid_categories,
)
from src.schema import SalaryInput

# Single hue for the salary marks (one series, so no legend is needed).
INTERVAL_COLOR = "#4C78A8"


def pct_delta(new: float, old: float, base_year: str) -> str | None:
    """Format the % change from old to new, or None if old is zero."""
    if not old:
        return None
    return f"{(new / old - 1) * 100:+.1f}% vs {base_year}"


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
        Developer Survey data (2025 + 2026 surveys) to predict annual
        salaries based on:
        - Country
        - Total years of coding experience (including education)
        - Years of professional work experience
        - Education level
        - Developer type
        - Industry
        - Age
        - Individual contributor or people manager
        - Organization size
        - Employment status
        - Survey year (2025, 2026), both predicted automatically

        Each prediction comes with an 80% range (a conformal prediction
        interval) so you can see the uncertainty, not just a single number.
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
    st.write(f"**Employment:** {len(valid_categories['Employment'])} available")
    st.write(f"**Survey Years:** {', '.join(valid_categories['SurveyYear'])}")
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
valid_employment = valid_categories["Employment"]

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
default_employment = (
    "Employed" if "Employed" in valid_employment else valid_employment[0]
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

employment = st.selectbox(
    "Employment Status",
    options=valid_employment,
    index=valid_employment.index(default_employment),
    help="Current employment status",
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
            employment=employment,
        )

        # Make a prediction (with interval) for every survey year
        with st.spinner("Calculating prediction..."):
            results = predict_salaries_with_intervals(input_data)

        # Display result
        st.success("Prediction Complete!")

        salaries = {year: r["prediction"] for year, r in results.items()}

        # Central coverage of the conformal interval (e.g. 0.80), if present
        coverage = next(
            (r["coverage"] for r in results.values() if r["coverage"]), None
        )
        cov_pct = f"{coverage:.0%}" if coverage else None

        # One column per survey year: USD and local currency (each year uses
        # its own survey-derived rate); later years show % change vs the first
        base_year = min(salaries)
        base_usd = salaries[base_year]
        base_local = get_local_currency(country, base_usd, survey_year=base_year)
        year_columns = st.columns(len(results))
        for year_column, (year, result) in zip(year_columns, results.items()):
            salary = result["prediction"]
            lower, upper = result["lower"], result["upper"]
            has_interval = lower is not None and upper is not None
            is_base = year == base_year
            local = get_local_currency(country, salary, survey_year=year)
            with year_column:
                st.subheader(year)
                st.metric(
                    label="Estimated Annual Salary (USD)",
                    value=f"${salary:,.0f}",
                    delta=None if is_base else pct_delta(salary, base_usd, base_year),
                    help=f"Predicted annual compensation in USD ({year} survey)",
                )
                if has_interval:
                    # Escape $ so Streamlit markdown does not read $...$ as LaTeX
                    st.caption(rf"{cov_pct} range: \${lower:,.0f} – \${upper:,.0f}")
                if local and local["code"] != "USD":
                    code = local["code"]
                    # No % when the currency itself changed (e.g. BGN -> EUR)
                    local_delta = None
                    if not is_base and base_local and base_local["code"] == code:
                        local_delta = pct_delta(
                            local["salary_local"],
                            base_local["salary_local"],
                            base_year,
                        )
                    st.metric(
                        label=f"Estimated Annual Salary ({code})",
                        value=f"{local['salary_local']:,.0f} {code}",
                        delta=local_delta,
                        help=f"Converted using survey-derived rate: 1 USD = {local['rate']} {code} ({local['name']}, {local['rate_year']} survey)",
                    )
                    if has_interval:
                        local_lo = get_local_currency(country, lower, survey_year=year)
                        local_hi = get_local_currency(country, upper, survey_year=year)
                        if local_lo and local_hi:
                            st.caption(
                                f"{cov_pct} range: "
                                f"{local_lo['salary_local']:,.0f} – "
                                f"{local_hi['salary_local']:,.0f} {code}"
                            )
                    if not is_base and base_local and base_local["code"] != code:
                        st.caption(
                            f"Local currency changed from {base_local['code']} "
                            f"to {code}; no % change shown."
                        )
                    if local["rate_year"] != year:
                        st.caption(
                            f"No {year} rate for {country}; "
                            f"using the {local['rate_year']} survey rate."
                        )

        # Visual: central estimate (point) with the prediction interval (bar),
        # one row per survey year. One series, so no legend; the title says it.
        interval_rows = [
            {
                "year": y,
                "mean": r["prediction"],
                "lower": r["lower"],
                "upper": r["upper"],
            }
            for y, r in results.items()
            if r["lower"] is not None and r["upper"] is not None
        ]
        if interval_rows:
            chart_df = pd.DataFrame(interval_rows)
            base = alt.Chart(chart_df).encode(
                y=alt.Y(
                    "year:N",
                    title=None,
                    sort=sorted(salaries),
                    scale=alt.Scale(padding=0.5),
                    axis=alt.Axis(
                        labelFontSize=15,
                        labelFontWeight="bold",
                        labelPadding=12,
                        ticks=False,
                        domain=False,
                    ),
                )
            )
            # Soft rounded band = the interval; solid dot = the point estimate
            band = base.mark_rule(
                size=10, color=INTERVAL_COLOR, opacity=0.3, strokeCap="round"
            ).encode(
                x=alt.X(
                    "lower:Q",
                    title=f"Annual salary (USD) — point estimate + {cov_pct} interval",
                    axis=alt.Axis(format="$,.0f", grid=False),
                    scale=alt.Scale(zero=False, nice=True),
                ),
                x2="upper:Q",
            )
            dot = base.mark_point(
                size=180, filled=True, color=INTERVAL_COLOR, opacity=1
            ).encode(x="mean:Q")
            # Value label above each dot; theme ink (not the series color)
            label = base.mark_text(dy=-15, fontSize=13, fontWeight="bold").encode(
                x="mean:Q", text=alt.Text("mean:Q", format="$,.0f")
            )
            chart = (band + dot + label).properties(height=150)
            st.altair_chart(chart, use_container_width=True)

        if cov_pct:
            interval_note = (
                f"The bar shows an {cov_pct} prediction interval (conformal, "
                "calibrated per survey year on held-out responses): about "
                f"{cov_pct} of developers matching these inputs earn within it. "
                "The point is the model's central estimate. "
            )
        else:
            interval_note = ""
        st.caption(
            interval_note + "The year-over-year change is the difference between "
            "two survey-based estimates, not a guaranteed raise. The "
            "local-currency change also includes exchange-rate movements "
            "between the surveys."
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
    "Built with Streamlit • Data from Stack Overflow Developer Survey • "
    "Model: XGBoost + conformal prediction intervals"
)
