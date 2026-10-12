#263 HW1

import argparse
import numpy as np


def gradEs(xa, xb, k):
    return k * (xb - xa) * np.array([-1.0, 1.0])

print("test:gradEs", gradEs(0, 0.1, 10))  # test

def hessEs(k):

    return k * np.array([[1.0, -1.0], [-1.0, 1.0]])
print("test:hessEs", hessEs(10))

def dampingForce(xa_new, xb_new, xa_old, xb_old, b, dt):
    
    ua_new = (xa_new - xa_old) / dt
    ub_new = (xb_new - xb_old) / dt
    return b * (ub_new - ua_new) * np.array([-1.0, 1.0])
print("test:dampingForce", dampingForce(0, 0.01, 0, 0, 2, 0.01))

def dampingJacobian(b, dt):
    """Derivative of dampingForce with respect to the new positions."""
    return (b / dt) * np.array([[1.0, -1.0], [-1.0, 1.0]])
print("test:dampingJacobian ", dampingJacobian(2,0.01)  )

def getFJ(x_new, x_old, u_old, params):
    """Assemble the full residual and Jacobian for implicit Euler."""
    m = np.asarray(params["m"], dtype=float)
    k = np.asarray(params["k"], dtype=float)
    b = np.asarray(params["b"], dtype=float)
    dt = float(params["dt"])
    g = float(params.get("g", 9.8))
    n = len(m) - 1

    x_new = np.asarray(x_new, dtype=float)
    x_old = np.asarray(x_old, dtype=float)
    u_old = np.asarray(u_old, dtype=float)
    f = (m / dt**2) * (x_new - x_old - dt * u_old) - m * g
    J = np.diag(m / dt**2)

    # Each element contributes only to the two nodes it connects.
    for i in range(1, n + 1):
        idx = [i - 1, i]
        f[idx] += gradEs(x_new[i - 1], x_new[i], k[i])
        f[idx] += dampingForce(
            x_new[i - 1], x_new[i], x_old[i - 1], x_old[i], b[i], dt
        )
        J[np.ix_(idx, idx)] += hessEs(k[i])
        J[np.ix_(idx, idx)] += dampingJacobian(b[i], dt)

    return f, J
tparams = {
        "m": [0, 0, 0, 0],
        "k": [0, 1, 2, 3],
        "b": [0, 0, 0, 0],
        "dt": 1,
        "g": 0
    }
tx_new = [0, 0, 0, 0]
tx_old = [0, 0, 0, 0]
tu_old = [0, 0, 0, 0]
testf, testJ=getFJ(tx_new, tx_old, tu_old, tparams) 
J_free = testJ[1:, 1:]
print(J_free)
def implicitEulerStep(x_old, u_old, params):
    """Advance one implicit Euler step using Newton iteration."""
    x_old = np.asarray(x_old, dtype=float)
    u_old = np.asarray(u_old, dtype=float)
    x_new = x_old.copy()
    n = len(x_old) - 1
    free = np.arange(1, n + 1)
    # A slightly relaxed absolute tolerance avoids chasing floating point noise
    # after many thousands of steps while preserving the exact linear solve.
    tol = float(params.get("tol", 1e-8))
    max_iter = int(params.get("max_iter", 20))

    for _ in range(max_iter):
        f, J = getFJ(x_new, x_old, u_old, params)
        dx = np.linalg.solve(J[np.ix_(free, free)], f[free])
        x_new[free] -= dx
        x_new[0] = 0.0
        f_after, _ = getFJ(x_new, x_old, u_old, params)
        if np.linalg.norm(f_after[free]) < tol:
            break
    else:
        raise RuntimeError("Newton iteration did not converge")

    u_new = (x_new - x_old) / float(params["dt"])
    u_new[0] = 0.0
    return x_new, u_new


def explicitEulerStep(x_old, u_old, params):
    """Advance one forward Euler step using old positions and velocities."""
    m = np.asarray(params["m"], dtype=float)
    k = np.asarray(params["k"], dtype=float)
    b = np.asarray(params["b"], dtype=float)
    dt = float(params["dt"])
    g = float(params.get("g", 9.8))
    n = len(m) - 1
    x_old = np.asarray(x_old, dtype=float)
    u_old = np.asarray(u_old, dtype=float)

    internal = np.zeros(n + 1)
    for i in range(1, n + 1):
        idx = [i - 1, i]
        internal[idx] += gradEs(x_old[i - 1], x_old[i], k[i])
        internal[idx] += b[i] * (u_old[i] - u_old[i - 1]) * np.array([-1.0, 1.0])

    acceleration = np.zeros(n + 1)
    acceleration[1:] = g - internal[1:] / m[1:]
    x_new = x_old.copy()
    u_new = u_old.copy()
    x_new[1:] = x_old[1:] + dt * u_old[1:]
    u_new[1:] = u_old[1:] + dt * acceleration[1:]
    x_new[0] = 0.0
    u_new[0] = 0.0
    return x_new, u_new


def simulate(x0, u0, params, total_time, method="implicit"):
    """Simulate and return times, positions, and velocities at every time step."""
    dt = float(params["dt"])
    steps = int(round(total_time / dt))
    x = np.asarray(x0, dtype=float).copy()
    u = np.asarray(u0, dtype=float).copy()
    xs = np.empty((steps + 1, len(x)))
    us = np.empty_like(xs)
    ts = np.arange(steps + 1, dtype=float) * dt
    xs[0], us[0] = x, u
    stepper = implicitEulerStep if method == "implicit" else explicitEulerStep
    for step in range(steps):
        x, u = stepper(x, u, params)
        xs[step + 1], us[step + 1] = x, u
    return ts, xs, us


def run_tests():
    """Run the eleven numerical checks supplied in the assignment."""
    assert np.allclose(gradEs(0, 0.1, 10), [-1, 1])
    assert np.allclose(hessEs(10), [[10, -10], [-10, 10]])
    assert np.allclose(dampingForce(0, 0.01, 0, 0, 2, 0.01), [-2, 2])
    assert np.allclose(dampingJacobian(2, 0.01), [[200, -200], [-200, 200]])

    p = {"m": np.zeros(4), "k": np.array([0., 1., 2., 3.]),
         "b": np.zeros(4), "g": 0., "dt": 1.}
    f, J = getFJ(np.zeros(4), np.zeros(4), np.zeros(4), p)
    assert np.allclose(J[1:, 1:], [[3, -2, 0], [-2, 5, -3], [0, -3, 3]])

    p = {"m": np.array([0., 1., 2., 3.]), "k": np.array([0., 10., 20., 30.]),
         "b": np.array([0., 1., 2., 3.]), "g": 9.8, "dt": 0.1}
    xo = np.array([0., 0., 0., 0.])
    uo = np.array([0., 0.1, 0.2, 0.3])
    xn = np.array([0., 0.01, 0.03, 0.06])
    f, J = getFJ(xn, xo, uo, p)
    assert np.allclose(f[1:], [-10.4, -18.6, -18.6])
    assert np.allclose(J[1:, 1:], [[160, -40, 0], [-40, 300, -60], [0, -60, 360]])
    x, u = implicitEulerStep(xo, uo, p)
    assert np.allclose(x, [0, 0.096696, 0.116786, 0.126131], atol=2e-6)
    assert np.allclose(u, [0, 0.966964, 1.167857, 1.261310], atol=2e-6)

    p = {"m": np.array([0., 1.]), "k": np.array([0., 0.]),
         "b": np.array([0., 0.]), "g": 9.8, "dt": 0.01}
    x, u = implicitEulerStep(np.zeros(2), np.zeros(2), p)
    assert np.allclose(x, [0, 0.00098])
    assert np.allclose(u, [0, 0.098])

    p = {"m": np.array([0., 1.]), "k": np.array([0., 1.]),
         "b": np.array([0., 0.]), "g": 0., "dt": 0.1}
    x, u = explicitEulerStep(np.array([0., 1.]), np.zeros(2), p)
    assert np.allclose(x, [0, 1])
    assert np.allclose(u, [0, -0.1])

    p = {"m": np.array([0., 1., 2., 3.]), "k": np.array([0., 10., 20., 30.]),
         "b": np.array([0., 1., 2., 3.]), "g": 9.8, "dt": 0.1}
    x, u = explicitEulerStep(np.array([0., .01, .03, .06]),
                             np.array([0., .1, .2, .3]), p)
    assert np.allclose(x, [0, .02, .05, .09])
    assert np.allclose(u, [0, 1.12, 1.21, 1.24])
    print("All Homework 1 tests passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run supplied test cases")
    args = parser.parse_args()
    if args.test:
        run_tests()
        return

    # A small default run, so `python Homework1.py` works immediately.
    n = 3
    params = {"m": np.array([0., 1., 1., 1.]),
              "k": np.array([0., 10., 10., 10.]),
              "b": np.array([0., 1., 1., 1.]),
              "g": 9.8, "dt": 0.01}
    x0 = np.zeros(n + 1)
    u0 = np.zeros(n + 1)
    _, positions, velocities = simulate(x0, u0, params, 1.0)
    print("Final positions:", positions[-1])
    print("Final velocities:", velocities[-1])


if __name__ == "__main__":
    main()
