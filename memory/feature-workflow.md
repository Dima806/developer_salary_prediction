# Adding a New Input Feature

Current features (v3.0.0): Country, YearsCode, WorkExp, EdLevel, DevType,
Industry, Age, ICorPM, OrgSize, Employment

## Checklist (18 steps, in order)

1. `config/model_parameters.yaml` — add to `features.cardinality.drop_other_from` if applicable
2. `src/schema.py` — add `field: str` to `SalaryInput`; update JSON example in model_config
3. `src/preprocessing.py` — add to `_categorical_cols` list; add `.fillna("Unknown")`; add to `feature_cols`
4. `src/train.py` — add to `CATEGORICAL_FEATURES` and `usecols`
5. `src/tune.py` — add to `usecols`
6. `src/preprocess.py` — add to `REQUIRED_COLUMNS`
7. `src/infer.py` — add validation block + add to `input_df` dict
8. `guardrail_evaluation.py` — add to `CATEGORICAL_FEATURES` and `usecols`
9. `app.py` — add `valid_X`, `default_X`, selectbox, sidebar entry, `SalaryInput(x=x)`
10. `tests/conftest.py` — add to `sample_salary_input` fixture
11. `tests/test_schema.py` — add to 3 direct `SalaryInput(...)` calls in missing-field tests
12. `tests/test_infer.py` — add invalid-value test
13. `tests/test_feature_impact.py` — add to all 9+ `base_input` dicts + direct `SalaryInput` call
14. `tests/test_preprocessing.py` — add column to all `pd.DataFrame(...)` fixtures
15. `tests/test_train.py` — add to `_make_salary_df()` helper + all inline DataFrames
16. `README.md` — required columns, web interface list, code example, guardrails list
17. `Claude.md` — data requirements table, field counts, code example, version
18. `example_inference.py` — add to all `SalaryInput` calls
19. Retrain: `uv run python -m src.train`

## Versioning on new required field
New required field = **MAJOR** version bump.
Update `pyproject.toml`, then: `git tag v<N>.0.0 && git push origin v<N>.0.0`

## Valid categories gotcha
`valid_categories.yaml` is pre-seeded manually, then overwritten by `make train`.
The linter (ruff) may simplify YAML values — pre-seeded values must match exactly
what the linter will produce. Use short-form values (e.g. `"Employed"` not
`"Employed, full-time"`).
