# Homework 4

## Run



Run all Q1--Q11 calculations and explanations:

```powershell
py -3.11 Homework4.py
```

If your interpreter is available as `python`, `python Homework4.py` also works.
This homework runs quickly; it does not require a full shell dynamics simulation.
Numerical questions Q1--Q8 are computed, not just printed as preset answers.
Q9--Q11 are conceptual explanations. Answer letters follow the handout order.

Run independent checks:

```powershell
py -3.11 Homework4.py --test
```

The tests check mesh topology, signed angles, individual coordinate perturbations,
the analytical gradient against central differences and an independent chain rule,
the Hessian, translation invariance, and oscillator energy under implicit Euler
and average-acceleration Newmark. The homework answer uses the REQUIRED forward
difference with epsilon=0.01 m; the more accurate central difference is only a test.

Results are saved beside the script in `results/`: `answers.txt`, mesh arrays,
`energy_gradient_check.csv`, and the analytical Hessian. No plots are required by
this assignment. Use `--output-dir another_folder` to save elsewhere.

## Files

- `Homework4.py`: mesh counting, hinge energy, all perturbations, gradient comparison,
  Q1--Q11 reporting and tests.
- `gradEb_hessEb_Shell.py`: unchanged course hinge-angle, gradient and Hessian functions.
- `requirements.txt`: NumPy dependency.
- `THIRD_PARTY_NOTICES.md`: course source and attribution.
- `README.md`: instructions and file descriptions.

