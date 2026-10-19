# Homework 2

`Homework2.py` contains both beam simulators, analytical checks, and the Q1--Q11 calculations. `README.md` contains the run instructions.

## Run the complete homework

Open a terminal in this folder. Install the dependencies once, using the same Python interpreter that runs the script:

```powershell
python -m pip install numpy matplotlib
```

Run:

```powershell
python Homework2.py
```

This computes Q1--Q11, prints the numerical results and answer choices, and saves the following files in `results/` beside the script:

- `answers.txt`: all results and explanations.
- `Q8_time_history.png`: maximum downward displacement versus time, with a transient zoom.
- `Q10_load_deflection.png`: simulated and Euler--Bernoulli deflections for loads from 20 to 20,000 N.
- CSV files: falling-beam history, Q5 node study, Q8 time history, Q10 load sweep, and Q11 segment extensions.

The full run can take several minutes. Progress appears in the terminal; plots are saved automatically. Each load starts from a straight beam at rest.

## Run the checks

```powershell
python Homework2.py --test
```

## Alternative run commands

`--answers` is equivalent to the default complete run:

```powershell
python Homework2.py --answers
```

Run just one part:

```powershell
python Homework2.py --part 1
python Homework2.py --part 2
```

These commands save `answers_part1.txt` or `answers_part2.txt`, respectively. Part 2 generates both required plots. To use a different output folder:

```powershell
python Homework2.py --output-dir my_results
```
