# Tests for Pediatric Appendicitis Diagnosis

This directory contains all tests for the Pediatric Appendicitis Diagnosis application.

## Test Structure

The tests are organized into the following categories:

- `unit/`: Unit tests for individual functions and classes
- `integration/`: Tests for interactions between different components
- `api/`: Tests for the Flask API endpoints
- `data_processing/`: Tests for data preprocessing functions
- `explainability/`: Tests for SHAP and model explanation features
- `models/`: Tests for model training, evaluation, and persistence

## Running Tests

To run all tests:

```bash
python -m pytest
```

To run tests with coverage report:

```bash
python -m pytest --cov=src
```

To run a specific test category:

```bash
python -m pytest tests/unit/
```

## Test Requirements

Testing dependencies are specified in `requirements-dev.txt` and include:
- pytest
- pytest-cov
- pytest-flask (for API testing)

## Writing New Tests

When adding new features or fixing bugs, please add corresponding tests. Follow these guidelines:

1. Place tests in the appropriate directory based on what you're testing
2. Name test files with the prefix `test_`
3. Name test functions with the prefix `test_`
4. Include docstrings describing what each test is verifying
5. Use meaningful assertions that clearly indicate what's being tested

## Continuous Integration

Tests are automatically run via GitHub Actions when code is pushed or a pull request is created.
