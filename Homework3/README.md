# Homework 3

## Run



Run all Q1--Q11 calculations:

```powershell
py -3.11 Homework3.py
```

Run the checks, or only one part:

```powershell
py -3.11 Homework3.py --test
py -3.11 Homework3.py --part 1
py -3.11 Homework3.py --part 2
```

Partial runs replace `results/answers.txt` with the selected part's report.
Use `--output-dir results_part1` or `--output-dir results_part2` to keep separate reports.

## Files

- `Homework3.py`: homework setup, implicit Euler/Newton solver, Q1--Q11 sweeps,
  analytical predictions, steady-state checks and plotting.
- `der_functions.py`: course DER frames and force assembly; local dependency path adjusted.
- `gradEs_hessEs.py`: stretching energy derivatives.
- `gradEb_hessEb.py`: bending energy derivatives.
- `gradEt_hessEt.py`: twisting energy derivatives, with the Panetta correction.
- `computekappa.py`: discrete curvature.
- `rotateAxisAngle.py`, `crossMat.py`: rotation and vector algebra helpers.
- `requirements.txt`: NumPy, SciPy and Matplotlib dependencies.
- `README.md`: run instructions and file descriptions.
- `THIRD_PARTY_NOTICES.md`: source and attribution for the course helpers.

