# Claude Development Guide

## Project Overview
This is a minimal, local-first ML application built in Python that predicts developer salaries using Stack Overflow Developer Survey data. The project emphasizes clarity and simplicity over production completeness.

## Tech Stack
- **Python 3.11+**
- **uv** - Package & virtual environment management
- **pandas** - Data manipulation
- **scikit-learn** - ML modeling
- **pydantic** - Input validation
- **streamlit** - Web UI
- **xgboost** - Advanced gradient boosting (optional)

## Project Structure
```
.
├── data/
│   └── survey_results_public.csv    # Stack Overflow survey data
├── models/
│   └── model.pkl                    # Serialized trained model
├── src/
│   ├── schema.py                    # Pydantic validation models
│   ├── train.py                     # Model training script
│   └── infer.py                     # Inference utilities
├── app.py                           # Streamlit web application
├── example_inference.py             # Example inference script
├── pyproject.toml                   # Project dependencies (uv)
├── uv.lock                          # Locked dependencies
└── README.md                        # Project documentation
```

## Setup & Installation

### Initial Setup
```bash
# The virtual environment is already created at .venv/
# Activate it:
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows

# Install/sync dependencies with uv:
uv sync
```

### Adding New Dependencies
```bash
uv add <package-name>
```

## Key Workflows

### Training the Model
```bash
python src/train.py
```
This will:
- Load data from `data/survey_results_public.csv`
- Clean and preprocess features
- Train the regression model
- Save model to `models/model.pkl`

### Running the Streamlit App
```bash
streamlit run app.py
```
Opens a browser interface for salary predictions.

### Running Inference Programmatically
```python
from src.schema import SalaryInput
from src.infer import predict_salary

input_data = SalaryInput(
    country="United States",
    years_code=5.0,
    education_level="Bachelor's degree",
    dev_type="Developer, back-end",
    industry="Software Development",
    age="25-34 years old"
)
salary = predict_salary(input_data)
```

## Key Files

### [src/schema.py](src/schema.py)
Contains Pydantic models for:
- Input validation (`SalaryInput`)
- Type safety across the application

### [src/train.py](src/train.py)
Training pipeline:
- Data loading and cleaning
- Feature engineering
- Model training
- Model persistence

### [src/infer.py](src/infer.py)
Inference utilities:
- Model loading
- Prediction logic
- Validated input processing

### [app.py](app.py)
Streamlit UI:
- User input forms
- Real-time predictions
- Results visualization

## Development Guidelines

### Code Style
- Keep code simple and readable
- Total codebase should remain under ~200 lines
- Focus on clarity over cleverness
- Use type hints where helpful

### Data Requirements
The dataset must include these columns:
- `Country` - Developer location
- `YearsCode` - Total years of coding (including education)
- `EdLevel` - Education level
- `DevType` - Developer type
- `Industry` - Industry the developer works in
- `Age` - Developer's age range
- `ConvertedCompYearly` - Annual salary (target variable)

### Model Expectations
- Basic regression model (LinearRegression or similar)
- Simple feature encoding (one-hot for categoricals)
- No hyperparameter tuning required
- Focus on working end-to-end pipeline

## Common Tasks

### Debugging Training Issues
1. Check if data file exists: `ls -la data/`
2. Verify CSV columns: `head -1 data/survey_results_public.csv`
3. Check for missing values in target column
4. Review data types and encoding

### Updating Features
1. Modify `SalaryInput` schema in [src/schema.py](src/schema.py)
2. Update feature extraction in [src/train.py](src/train.py)
3. Update inference logic in [src/infer.py](src/infer.py)
4. Update UI inputs in [app.py](app.py)
5. Retrain the model

### Testing Predictions
```python
# Quick test in Python REPL
from src.infer import predict_salary
from src.schema import SalaryInput

test_input = SalaryInput(
    country="United States",
    years_code=3.0,
    education_level="Bachelor's degree",
    dev_type="Developer, back-end",
    industry="Software Development",
    age="25-34 years old"
)
print(predict_salary(test_input))
```

## Non-Goals (Intentionally Excluded)
- Cloud deployment or serving
- Hyperparameter tuning
- Model registry or experiment tracking
- Advanced feature engineering
- Production monitoring
- API endpoints (beyond Streamlit)

## Useful Commands

```bash
# Check environment
which python
python --version

# Verify uv installation
uv --version

# List installed packages
uv pip list

# Run with specific Python version
uv run python src/train.py

# Clean generated files
rm -f models/model.pkl

# Check data file size
du -h data/survey_results_public.csv
```

## Troubleshooting

### Model file not found
- Run training first: `python src/train.py`
- Check file exists: `ls -la models/model.pkl`

### Missing dependencies
- Sync environment: `uv sync`
- Verify pyproject.toml has all required packages

### Data file issues
- Ensure CSV is in `data/` directory
- Check file encoding (should be UTF-8)
- Verify required columns exist

### Streamlit won't start
- Check port 8501 is available
- Try specifying port: `streamlit run app.py --server.port 8502`

## Additional Resources
- [PRD](.llm/prd.md) - Full product requirements
- [README.md](README.md) - Project readme
- [Stack Overflow Survey](https://insights.stackoverflow.com/survey) - Data source

## Working with Claude Code
When asking Claude to help with this project:
- Reference specific files using markdown links: [filename](path)
- Be specific about which component needs changes
- Mention if you need training, inference, or UI updates
- Provide error messages in full when debugging
- Ask for explanations of model choices if unclear
