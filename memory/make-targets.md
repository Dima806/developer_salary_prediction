# Makefile Targets

## Quality checks (no data/model needed)
| Target | What it does |
|--------|-------------|
| `make lint` | ruff check |
| `make format` | ruff format (auto-fixes) |
| `make test` | pytest |
| `make coverage` | pytest --cov |
| `make complexity` | radon CC |
| `make maintainability` | radon MI |
| `make audit` | pip-audit |
| `make security` | bandit (medium+ severity) |
| `make check` | lint + test + complexity + maintainability + audit + security |
| `make ci` | lint + test (mirrors `.github/workflows/ci.yml`) |
| `make pre-commit` | `prek run --all-files` (reads `.pre-commit-config.yaml`) |

## Data / model pipeline (requires `data/survey_results_public.csv`)
| Target | What it does | Requires |
|--------|-------------|---------|
| `make pre-process` | Validate columns, write valid_categories + currency_rates | data CSV |
| `make tune` | Guardrail check → Optuna hyperparameter search | data CSV |
| `make train` | Full training pipeline, writes model.pkl | data CSV |
| `make guardrails` | Per-category CV evaluation, exits 1 if Abs % Diff violations | data CSV |
| `make smoke-test` | Runs example_inference.py against live model | model.pkl |
| `make app` | Launches Streamlit UI | model.pkl |
| `make all` | format → lint → test → coverage → complexity → maintainability → audit → security → pre-process → train → smoke-test → guardrails | data CSV |

## Key dependency: `make tune`
`tune` runs `check_guardrails(config)` **inside `src/tune.py`** before starting Optuna.
If any category exceeds `max_abs_pct_diff`, tuning is aborted. MAPE is reported but not enforced.

## Pre-commit setup (once per clone)
```bash
uv tool install prek && prek install
```
