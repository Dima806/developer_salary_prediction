# Developer Salary Prediction

A minimal, local-first ML application that predicts developer salaries using Stack Overflow Developer Survey data. Built with Python, scikit-learn, Pydantic, and Streamlit.

## Features

- 🎯 Simple Linear Regression model for salary prediction
- ✅ Input validation with Pydantic
- 🌐 Interactive web UI with Streamlit
- 📊 Trained on Stack Overflow Developer Survey data
- 🔧 Easy setup with `uv` package manager

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Download Data

Download the Stack Overflow Developer Survey CSV file:

1. Visit: https://insights.stackoverflow.com/survey
2. Download the latest survey results (2024 or 2025)
3. Extract the `survey_results_public.csv` file
4. Place it in the `data/` directory:
   ```
   data/survey_results_public.csv
   ```

**Required columns:** `Country`, `YearsCodePro`, `EdLevel`, `ConvertedCompYearly`

### 3. Train the Model

```bash
python src/train.py
```

This will:
- Load and preprocess the survey data
- Train a Linear Regression model
- Save the model to `src/model.pkl`

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Web Interface

Launch the Streamlit app and enter:
- **Country**: Developer's country
- **Years of Professional Coding**: Experience in years
- **Education Level**: Highest degree completed

Click "Predict Salary" to see the estimated annual salary.

### Programmatic Usage

```python
from src.schema import SalaryInput
from src.infer import predict_salary

# Create input
input_data = SalaryInput(
    country="United States",
    years_code_pro=5.0,
    education_level="Bachelor's degree"
)

# Get prediction
salary = predict_salary(input_data)
print(f"Estimated salary: ${salary:,.0f}")
```

## Project Structure

```
.
├── data/
│   └── survey_results_public.csv    # Stack Overflow survey data (download required)
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── schema.py                    # Pydantic models
│   ├── train.py                     # Training script
│   ├── infer.py                     # Inference utilities
│   └── model.pkl                    # Trained model (generated)
├── app.py                           # Streamlit web app
├── pyproject.toml                   # Project dependencies
└── README.md                        # This file
```

## Tech Stack

- **Python 3.12+**
- **uv** - Package manager
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **pydantic** - Data validation
- **streamlit** - Web UI

## Development

For detailed development information, see [Claude.md](Claude.md).

### Re-training the Model

If you want to use a different survey year or update the model:

```bash
# Place new CSV in data/ directory
python src/train.py
```

### Running Tests

```python
# Quick test
python -c "
from src.schema import SalaryInput
from src.infer import predict_salary

test = SalaryInput(country='United States', years_code_pro=5.0, education_level='Bachelor\\'s degree')
print(f'Prediction: ${predict_salary(test):,.0f}')
"
```

## Troubleshooting

### "Model file not found"
- Run `python src/train.py` first to generate the model

### "Data file not found"
- Download the Stack Overflow survey CSV and place it in `data/`

### Dependencies issues
- Run `uv sync` to ensure all packages are installed

## Design Principles

- **Simplicity**: Under 200 lines of code total
- **Clarity**: Easy to understand and modify
- **Local-first**: No cloud dependencies
- **Hackable**: Plain Python, no complex frameworks

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

Data from [Stack Overflow Developer Survey](https://insights.stackoverflow.com/survey)
