# Developer Salary Prediction

A minimal, local-first ML application that predicts developer salaries using Stack Overflow Developer Survey data. Built with Python, scikit-learn, Pydantic, and Streamlit.

## Features

- 🎯 XGBoost (gradient boosting) model for salary prediction
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

**Required columns:** `Country`, `YearsCode`, `EdLevel`, `DevType`, `ConvertedCompYearly`

### 3. Train the Model

```bash
uv run python -m src.train
```

This will:
- Load configuration from `config/model_parameters.yaml`
- Load and preprocess the survey data (with cardinality reduction)
- Train an XGBoost model with early stopping
- Save the model to `models/model.pkl`
- Generate `config/valid_categories.yaml` with valid country, education, and developer type values

### 4. Run the Streamlit App

```bash
uv run streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Web Interface

Launch the Streamlit app and enter:
- **Country**: Developer's country
- **Years of Coding (Total)**: Total years coding including education
- **Education Level**: Highest degree completed
- **Developer Type**: Primary developer role

Click "Predict Salary" to see the estimated annual salary.

### Programmatic Usage

**Quick example:**

```python
from src.schema import SalaryInput
from src.infer import predict_salary

# Create input
input_data = SalaryInput(
    country="United States of America",
    years_code=5.0,
    education_level="Bachelor's degree (B.A., B.S., B.Eng., etc.)",
    dev_type="Developer, full-stack"
)

# Get prediction
salary = predict_salary(input_data)
print(f"Estimated salary: ${salary:,.0f}")
```

**Run the example script:**

```bash
uv run python example_inference.py
```

This will show predictions for multiple sample scenarios (junior, mid-level, senior developers, different countries).

## Input Validation

The model validates inputs against actual training data categories:

- **Valid Countries**: Only countries from `config/valid_categories.yaml` (~21 countries)
- **Valid Education Levels**: Only education levels from training data (~9 levels)
- **Valid Developer Types**: Only developer types from training data (~20 types)

The Streamlit app uses dropdown menus with only valid options. If you use the programmatic API with invalid values, you'll get a helpful error message pointing to the valid categories file.

**Example validation:**
```python
from src.infer import predict_salary
from src.schema import SalaryInput

# This will raise ValueError - Japan not in training data after cardinality reduction
invalid_input = SalaryInput(
    country="Japan",  # Invalid!
    years_code=5.0,
    education_level="Bachelor's degree (B.A., B.S., B.Eng., etc.)",
    dev_type="Developer, back-end"
)
```

**View valid categories:**
```bash
cat config/valid_categories.yaml
```

## Configuration

Model parameters are centralized in [config/model_parameters.yaml](config/model_parameters.yaml). You can customize:

- **Data Processing**: Salary thresholds, percentile bounds, train/test split ratio
- **Feature Engineering**: Cardinality reduction settings (max categories, min frequency)
- **Model Hyperparameters**: Learning rate, tree depth, early stopping, etc.
- **Training Settings**: Verbosity, model save path

**To modify parameters:**

```bash
# Edit the config file
nano config/model_parameters.yaml

# Then retrain the model
uv run python -m src.train
```

**Example parameter changes:**
```yaml
# Increase model complexity
model:
  max_depth: 8                 # Default: 6
  n_estimators: 10000          # Default: 5000

# Keep more categories
features:
  cardinality:
    max_categories: 30         # Default: 20
    min_frequency: 100         # Default: 50
```

## Project Structure

```
.
├── config/
│   ├── model_parameters.yaml        # Model configuration
│   └── valid_categories.yaml        # Valid input categories (generated)
├── data/
│   └── survey_results_public.csv    # Stack Overflow survey data (download required)
├── models/
│   └── model.pkl                    # Trained model (generated)
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── schema.py                    # Pydantic models
│   ├── preprocessing.py             # Feature engineering utilities
│   ├── train.py                     # Training script
│   └── infer.py                     # Inference utilities
├── app.py                           # Streamlit web app
├── example_inference.py             # Example inference script
├── pyproject.toml                   # Project dependencies
└── README.md                        # This file
```

## Tech Stack

- **Python 3.12+**
- **uv** - Package manager
- **pandas** - Data manipulation
- **xgboost** - Gradient boosting model
- **scikit-learn** - ML utilities (train/test split)
- **pydantic** - Data validation
- **streamlit** - Web UI

## Development

For detailed development information, see [Claude.md](Claude.md).

### Re-training the Model

If you want to use a different survey year or update the model:

```bash
# Place new CSV in data/ directory
uv run python -m src.train
```

### Running Tests

**Quick one-liner test:**
```bash
uv run python -c "from src.schema import SalaryInput; from src.infer import predict_salary; test = SalaryInput(country='United States of America', years_code=5.0, education_level='Bachelor'\''s degree (B.A., B.S., B.Eng., etc.)', dev_type='Developer, full-stack'); print(f'Prediction: \${predict_salary(test):,.0f}')"
```

**Or run the full example script:**
```bash
uv run python example_inference.py
```

## Deployment

### Hugging Face Spaces

This application is Docker-ready for deployment on Hugging Face Spaces:

**1. Build the Docker image:**
```bash
docker build -t developer-salary-predictor .
```

**2. Test locally:**
```bash
docker run -p 8501:8501 developer-salary-predictor
```

Then visit `http://localhost:8501`

**3. Deploy to Hugging Face:**

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
2. Select "Docker" as the SDK
3. Clone your Space repository
4. Copy these files to your Space:

   ```text
   Dockerfile
   requirements.txt
   app.py
   src/
   config/
   models/
   ```

5. Push to your Space:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push
   ```

**Note:** The pre-trained model (`models/model.pkl`) and configuration (`config/valid_categories.yaml`) are included in the Docker image. If you want to use a different model, retrain locally first, then rebuild the Docker image.

### Alternative: Local Deployment

**Using uv (recommended for development):**
```bash
uv run streamlit run app.py
```

**Using pip:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Troubleshooting

### "Model file not found"
- Run `uv run python -m src.train` first to generate the model

### "Data file not found"
- Download the Stack Overflow survey CSV and place it in `data/`

### "Configuration file not found"
- The `config/model_parameters.yaml` file should exist in the project root
- Check that you're running commands from the project root directory

### Dependencies issues
- Run `uv sync` to ensure all packages are installed

## Design Principles

- **Simplicity**: Under 200 lines of code total
- **Clarity**: Easy to understand and modify
- **Local-first**: No cloud dependencies
- **Hackable**: Plain Python, no complex frameworks

## License

Apache 2.0 License - see [LICENSE](LICENSE) file

## Acknowledgments

Data from [Stack Overflow Developer Survey](https://insights.stackoverflow.com/survey)
