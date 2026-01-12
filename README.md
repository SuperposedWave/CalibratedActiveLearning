# Calibrated Active Learning

A Python package for calibrated active learning using empirical likelihood methods.

## Installation

### Option 1: Install as an editable package (Recommended)

From the project root directory, run:

```bash
pip install -e .
```

This will install the package in editable mode, allowing you to import modules from anywhere.

### Option 2: Add to PYTHONPATH

Alternatively, you can add the project root to your PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/CalibratedActiveLearning"
```

## Usage

After installation, you can import the modules:

```python
from Code import el_newton_lambda, el_weights
```

## Project Structure

```
CalibratedActiveLearning/
├── Code/
│   ├── __init__.py
│   └── solve_empirical_likelihood.py
├── Simulation/
│   └── test/
│       └── test.py
├── Misc/
│   └── note.md
├── setup.py
├── pyproject.toml
└── README.md
```

## Running Tests

After installation, you can run the test script:

```bash
python Simulation/test/test.py
```

