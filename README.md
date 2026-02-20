---
title: Developer Salary Prediction
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Developer salary prediction using 2025 Stackoverflow survey
license: apache-2.0
---

# Developer Salary Prediction

A minimal, local-first ML application that predicts developer salaries using Stack Overflow Developer Survey data. Built with Python, XGBoost, Pydantic, and Streamlit.

## Features

- 🎯 XGBoost (gradient boosting) model for salary prediction
- ✅ Input validation with Pydantic (schema) and runtime guardrails (valid categories)
- 🌐 Interactive web UI with Streamlit
- 📊 Trained on Stack Overflow 2025 Developer Survey data
- 🔧 Easy setup with `uv` package manager

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Download Data

Download the Stack Overflow Developer Survey CSV file:

1. Visit: https://insights.stackoverflow.com/survey
2. Download the latest survey results (2025)
3. Extract the `survey_results_public.csv` file
4. Place it in the `data/` directory:

   ```text
   data/survey_results_public.csv
   ```

**Required columns:** `Country`, `YearsCode`, `WorkExp`, `EdLevel`, `DevType`, `Industry`, `Age`, `ICorPM`, `ConvertedCompYearly`

### 3. Train the Model

```bash
uv run python -m src.train
```

This will:

- Load configuration from `config/model_parameters.yaml`
- Filter salaries and reduce cardinality of categorical features
- Run 5-fold cross-validation and report mean MAPE per fold
- Train a final XGBoost model on the full dataset with early stopping
- Save the model artifact to `models/model.pkl`
- Generate `config/valid_categories.yaml` — valid input values for runtime guardrails
- Generate `config/currency_rates.yaml` — per-country median currency conversion rates

### 4. Run the Streamlit App

```bash
uv run streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Development Cycle

The full development workflow from data to deployment:

```text
data/ ──► (optional) tune ──► train ──► test ──► commit ──► CI passes ──► deploy
```

### Step-by-step

#### 1. (Optional) Tune hyperparameters

Run Optuna to search for optimal XGBoost hyperparameters. The search space is
defined in `config/optuna_config.yaml`. Best parameters are written directly
back into `config/model_parameters.yaml`.

```bash
make tune
# or with a custom number of trials:
uv run python -m src.tune --n-trials 50
```

#### 2. Train the model

```bash
uv run python -m src.train
```

#### 3. Check code quality (lint + test + complexity + security)

```bash
make check
```

This runs all quality gates in sequence:

| Target | Tool | What it checks |
| ------ | ---- | -------------- |
| `make lint` | ruff | Style and linting errors |
| `make format` | ruff | Auto-formats code |
| `make test` | pytest | Unit and integration tests |
| `make coverage` | pytest-cov | Test coverage report |
| `make complexity` | radon CC | Cyclomatic complexity |
| `make maintainability` | radon MI | Maintainability index |
| `make audit` | pip-audit | Dependency vulnerability scan |
| `make security` | bandit | Static security analysis |

`make check` runs lint, test, complexity, maintainability, audit, and security together.
`make all` is an alias for `make check`.

#### 4. Run all pre-commit checks manually

```bash
uv run pre-commit run --all-files
```

## Usage

### Web Interface

Launch the Streamlit app and enter:

- **Country**: Developer's country
- **Years of Coding (Total)**: Total years coding including education
- **Years of Professional Work Experience**: Years of professional work experience
- **Education Level**: Highest degree completed
- **Developer Type**: Primary developer role
- **Industry**: Industry the developer works in
- **Age**: Developer's age range
- **IC or PM**: Individual contributor or people manager

Click "Predict Salary" to see the estimated annual salary in USD plus a local
currency equivalent where available.

### Programmatic Usage

```python
from src.schema import SalaryInput
from src.infer import predict_salary

input_data = SalaryInput(
    country="United States of America",
    years_code=5.0,
    work_exp=3.0,
    education_level="Bachelor's degree (B.A., B.S., B.Eng., etc.)",
    dev_type="Developer, full-stack",
    industry="Software Development",
    age="25-34 years old",
    ic_or_pm="Individual contributor"
)

salary = predict_salary(input_data)
print(f"Estimated salary: ${salary:,.0f}")
```

**Run the example script:**

```bash
uv run python example_inference.py
```

## Input Validation and Guardrails

Validation is enforced at two layers:

### Layer 1 — Pydantic schema (`src/schema.py`)

Checked at object construction time:

- All 8 fields are required
- `years_code` must be `>= 0`
- `work_exp` must be `>= 0`

### Layer 2 — Runtime category guardrails (`src/infer.py`)

Checked at inference time against `config/valid_categories.yaml`, which is
generated during training to reflect only categories that appeared frequently
enough in the training data (controlled by `features.cardinality.min_frequency`
in `config/model_parameters.yaml`):

- **Valid Countries** (~21) — low-frequency countries collapsed to `Other`, which is then dropped
- **Valid Education Levels** (~9)
- **Valid Developer Types** (~20) — `Other` dropped
- **Valid Industries** (~15) — `Other` dropped
- **Valid Age Ranges** (~7) — `Other` dropped
- **Valid IC/PM Values** (~3) — `Other` dropped

Passing an invalid value raises a `ValueError` with a message pointing to
`config/valid_categories.yaml`.

**Example:**

```python
from src.infer import predict_salary
from src.schema import SalaryInput

# Raises ValueError: "Invalid country: 'Japan'. Check config/valid_categories.yaml"
predict_salary(SalaryInput(country="Japan", ...))
```

**View valid categories:**

```bash
cat config/valid_categories.yaml
```

### Model guardrails (`config/model_parameters.yaml`)

The `guardrails` section defines thresholds used during training evaluation:

```yaml
guardrails:
  max_mape_per_category: 100   # max acceptable MAPE per category (%)
  max_abs_pct_diff: 100        # max acceptable absolute % difference
```

## Testing

Tests live in `tests/` and cover all major modules:

| File | What it tests |
| ---- | ------------- |
| `test_schema.py` | Pydantic validation — required fields, `ge=0` constraints |
| `test_infer.py` | Inference pipeline — valid predictions, `ValueError` on invalid categories, currency lookup |
| `test_train.py` | Training helpers — salary filtering, cardinality reduction, valid category extraction, currency rate computation |
| `test_preprocessing.py` | Feature engineering — one-hot encoding, numeric transforms |
| `test_tune.py` | Optuna helpers — parameter sampling, objective function construction, best-param saving |
| `test_feature_impact.py` | Model sanity — changing each input feature (country, education, dev type, etc.) produces a distinct prediction |

Run all tests:

```bash
make test
```

Run with coverage:

```bash
make coverage
```

## Configuration

All runtime parameters are centralised in two YAML files:

### `config/model_parameters.yaml`

Controls data processing, feature engineering, model hyperparameters, training
settings, and guardrail thresholds. You can customise:

- **Data Processing**: Salary thresholds, percentile bounds, train/test split ratio
- **Feature Engineering**: Cardinality reduction settings (max categories, min frequency)
- **Model Hyperparameters**: Learning rate, tree depth, early stopping, etc.
- **Training Settings**: Verbosity, model save path
- **Guardrails**: MAPE thresholds for model evaluation

**Example parameter changes:**

```yaml
# Increase model complexity
model:
  max_depth: 8                 # Default: 3
  n_estimators: 10000          # Default: 5000

# Keep more categories
features:
  cardinality:
    max_categories: 30         # Default: 30
    min_frequency: 50          # Default: 50
```

### `config/optuna_config.yaml`

Controls the Optuna hyperparameter search — search space (type, bounds, log
scale), number of trials, CV folds, and fixed parameters that are not tuned
(e.g. `n_estimators`, `random_state`).

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI (lint + test)
├── config/
│   ├── model_parameters.yaml        # Model configuration and guardrails
│   ├── optuna_config.yaml           # Optuna hyperparameter search space
│   ├── valid_categories.yaml        # Valid input categories (generated by training)
│   └── currency_rates.yaml          # Per-country currency rates (generated by training)
├── data/
│   └── survey_results_public.csv    # Stack Overflow survey data (download required)
├── models/
│   └── model.pkl                    # Trained model artifact (generated by training)
├── src/
│   ├── __init__.py
│   ├── schema.py                    # Pydantic input model
│   ├── preprocessing.py             # Feature engineering (one-hot encoding, scaling)
│   ├── train.py                     # Training pipeline
│   ├── tune.py                      # Optuna hyperparameter optimisation
│   └── infer.py                     # Inference with runtime guardrails
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── test_schema.py
│   ├── test_infer.py
│   ├── test_train.py
│   ├── test_preprocessing.py
│   ├── test_tune.py
│   └── test_feature_impact.py
├── app.py                           # Streamlit web app
├── example_inference.py             # Inference usage examples
├── Makefile                         # Developer workflow commands
├── .pre-commit-config.yaml          # Pre-commit hooks
├── pyproject.toml                   # Project dependencies
└── README.md                        # This file (also Hugging Face Space config)
```

## Tech Stack

- **Python 3.12+**
- **uv** — Package manager
- **pandas** — Data manipulation
- **xgboost** — Gradient boosting model
- **scikit-learn** — Cross-validation and train/test split
- **optuna** — Hyperparameter optimisation
- **pydantic** — Input schema validation
- **streamlit** — Web UI
- **ruff** — Linting and formatting
- **radon** — Complexity and maintainability metrics
- **bandit** — Static security analysis
- **pip-audit** — Dependency vulnerability scanning

## Development

For detailed development information, see [Claude.md](Claude.md).

### Code Quality

#### Pre-commit hooks

The project uses [pre-commit](https://pre-commit.com) to enforce code quality checks before each commit. Hooks are defined in [.pre-commit-config.yaml](.pre-commit-config.yaml) and run:

- **ruff format** — auto-formats Python files (`make format`)
- **ruff lint** — checks for linting errors (`make lint`)
- **Standard checks** — trailing whitespace, end-of-file newline, LF line endings, valid YAML/TOML/JSON, large files, merge conflict markers, stray debug statements

**Install hooks** (once, after cloning):

```bash
uv run pre-commit install
```

Hooks will then run automatically on every `git commit`. To run them manually against all files:

```bash
uv run pre-commit run --all-files
```

#### GitHub Actions CI

A CI workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs automatically on every push to any branch. It:

1. Sets up Python 3.12 and installs `uv`
2. Installs all dependencies (`uv sync --all-extras`)
3. Runs `make lint` — ruff linting
4. Runs `make test` — full pytest suite

The workflow must pass before merging changes.

### Re-training the Model

If you want to use a different survey year or update the model:

```bash
# 1. Place new CSV in data/
# 2. (Optional) tune first
make tune
# 3. Retrain
uv run python -m src.train
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make coverage

# Run a specific test file
uv run pytest tests/test_infer.py -v
```

## Versioning

This project follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`):

| Version bump | When to use | Examples |
| --- | --- | --- |
| **MAJOR** | Breaking changes to the public interface | New required input field, incompatible model artifact format, renamed API |
| **MINOR** | Backward-compatible new features | New optional input field, new supported country, new Makefile target, UI addition |
| **PATCH** | Backward-compatible fixes and improvements | Bug fixes, model retrain with same schema, config tuning, dependency updates |

**Pre-release suffixes** (for work in progress):

```text
v1.0.0-alpha.1   # early development, unstable
v1.0.0-beta.1    # feature-complete, under testing
v1.0.0-rc.1      # release candidate, final validation
```

Tags are applied on `main` after a successful CI run:

```bash
git tag v1.2.0
git push origin v1.2.0
```

## Branching Strategy

The project uses a **GitFlow-inspired** branching model:

```text
main ◄──────────────────────────────────── hotfix/v1.0.1
  ▲                                              │
  │ merge + tag                                  │
  │                                         (urgent fix)
develop ◄──── feature/add-currency-display
    ◄──── feature/new-dev-types
    ◄──── fix/invalid-category-message
    │
    └──► release/v1.1.0 ──► (final testing) ──► main + tag v1.1.0
```

### Branches

| Branch | Purpose | Merges into |
| ------ | ------- | ----------- |
| `main` | Production-ready code, always deployable. Tagged on every release. | — |
| `develop` | Integration branch for completed features. Base for new work. | `main` via release branch |
| `feature/<name>` | New features or improvements (e.g. `feature/add-local-currency`) | `develop` |
| `fix/<name>` | Non-urgent bug fixes (e.g. `fix/guardrail-error-message`) | `develop` |
| `release/v<semver>` | Release preparation — version bump, changelog, final QA | `main` and back to `develop` |
| `hotfix/v<semver>` | Urgent production fixes (e.g. `hotfix/v1.0.1`) | `main` and back to `develop` |

### Rules

- **`main`** is protected — no direct pushes; merge only via PR after CI passes
- **`develop`** is the default branch for day-to-day work
- Branch names use lowercase kebab-case: `feature/optuna-cv-splits`
- Every merge to `main` is tagged with a semver version
- Hotfixes branch off `main` directly and merge back to both `main` and `develop`

### Typical workflow

```bash
# Start a new feature
git checkout develop
git pull origin develop
git checkout -b feature/add-local-currency

# ... work, commit, push ...
git push -u origin feature/add-local-currency

# Open a PR into develop, CI must pass before merging

# Prepare a release
git checkout -b release/v1.1.0 develop
# bump version in pyproject.toml, update changelog
git push -u origin release/v1.1.0
# Open PR into main, merge, tag

git tag v1.1.0
git push origin v1.1.0
```

## Deployment

### Hugging Face Spaces

The app is deployed on [Hugging Face Spaces](https://huggingface.co/spaces) using the Docker SDK. The Space configuration is embedded in the frontmatter at the top of this README, which Hugging Face reads automatically:

- **SDK**: Docker (runs the `Dockerfile` in the repo root)
- **Port**: 8501 (Streamlit default)
- **License**: Apache 2.0

To deploy your own copy:

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) and select "Docker" as the SDK
2. Push this repository to your Space:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space main
   ```

**Note:** The pre-trained model (`models/model.pkl`) and configuration (`config/valid_categories.yaml`, `config/currency_rates.yaml`) must be present before building the Docker image. Train locally first if needed.

### Local Docker

**Build and run:**

```bash
docker build -t developer-salary-predictor .
docker run -p 8501:8501 developer-salary-predictor
```

Then visit `http://localhost:8501`

### Local (without Docker)

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

### "Valid categories file not found"

- Run `uv run python -m src.train` — training generates both `models/model.pkl`
  and `config/valid_categories.yaml`

### "Data file not found"

- Download the Stack Overflow survey CSV and place it in `data/`

### "Configuration file not found"

- The `config/model_parameters.yaml` file should exist in the project root
- Check that you're running commands from the project root directory

### Dependencies issues

- Run `uv sync` to ensure all packages are installed

## Design Principles

- **Simplicity**: Minimal codebase, easy to read and modify
- **Separation of concerns**: Schema validation, preprocessing, training, and inference are distinct modules
- **Config-driven**: All tunable parameters in YAML — no magic numbers in code
- **Local-first**: No cloud dependencies for training or inference
- **Testable**: Every public function has unit tests; model sanity covered by feature-impact tests

## License

Apache 2.0 License - see [LICENSE](LICENSE) file

## Acknowledgments

Data from [Stack Overflow Developer Survey](https://insights.stackoverflow.com/survey)
