# MAE-263F Homework 1

This `homework_1` folder is the submission folder for Homework 1. It contains a modular one-dimensional multi-mass spring-damper simulator. The implementation uses element assembly: each spring and damper contributes a local residual vector and a local Jacobian block.

## Required submission files

The homework handout requires the following files in the repository's `homework_1` folder:

- `Homework1.py` - the main source file. It implements `gradEs`, `hessEs`, `dampingForce`, `dampingJacobian`, `getFJ`, `implicitEulerStep`, `explicitEulerStep`, and the time-integration helper `simulate`.
- `README.md` - this file, including instructions for running the code and a description of the source file.

`Homework1.py` is the only main file intended for grading. The wall is node 0 and remains fixed. Arrays `m`, `k`, and `b` have length `N + 1`; entry 0 is unused and entries 1 through `N` correspond to the masses, springs, and dampers.

## Requirements

- Python 3.11 or later
- NumPy

Install NumPy if needed:

```powershell
python -m pip install numpy
```

## Running the code

Open a terminal in this folder and run the default example:

```powershell
python Homework1.py
```

Run the numerical checks supplied in the handout:

```powershell
python Homework1.py --test
```

The test command should print:

```text
All Homework 1 tests passed.
```

In VS Code, select the Python interpreter that has NumPy installed, then run `Homework1.py`. The included VS Code launch configuration also provides `Run Homework1.py` and `Test Homework1.py`.

## GitHub layout

The repository should contain this folder as follows:

```text
homework_1/
├── Homework1.py
└── README.md
```

The local `.venv`, `__pycache__`, and editor settings are for local development and should not be uploaded as homework source files.
