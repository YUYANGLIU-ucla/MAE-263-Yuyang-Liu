"""MAE-263F Homework 2: falling beam and simply-supported beam.

Run ``python Homework2.py --test`` for the checks listed in the handout.
Run ``python Homework2.py --answers`` to compute the Gradescope quantities.
"""

import argparse
import numpy as np


def cross_mat(a):
    """Matrix C such that C @ b = a cross b."""
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def gradEs(xk, yk, xkp1, ykp1, l0, EA):
    """Gradient of one stretching energy with respect to four coordinates."""
    d = np.array([xkp1 - xk, ykp1 - yk], dtype=float)
    length = np.linalg.norm(d)
    if length <= 1e-14:
        raise ValueError("Adjacent beam nodes cannot coincide")
    g = EA * (length - l0) / (l0 * length) * d
    return np.r_[-g, g]


def hessEs(xk, yk, xkp1, ykp1, l0, EA):
    """Hessian of one stretching energy with respect to four coordinates."""
    d = np.array([xkp1 - xk, ykp1 - yk], dtype=float)
    length = np.linalg.norm(d)
    if length <= 1e-14:
        raise ValueError("Adjacent beam nodes cannot coincide")
    H = EA / l0 * (1.0 - l0 / length) * np.eye(2)
    H += EA * np.outer(d, d) / length**3
    return np.block([[H, -H], [-H, H]])


def _bend_geometry(values):
    """Return curvature and its first/second coordinate derivatives."""
    x0, y0, x1, y1, x2, y2 = values
    node0 = np.array([x0, y0, 0.0])
    node1 = np.array([x1, y1, 0.0])
    node2 = np.array([x2, y2, 0.0])
    ee = node1 - node0
    ef = node2 - node1
    ne = np.linalg.norm(ee)
    nf = np.linalg.norm(ef)
    if ne <= 1e-14 or nf <= 1e-14:
        raise ValueError("Adjacent beam nodes cannot coincide")
    te = ee / ne
    tf = ef / nf
    chi = 1.0 + np.dot(te, tf)
    if chi <= 1e-12:
        raise ValueError("Bending angle is too close to 180 degrees")
    m2e = np.array([0.0, 0.0, 1.0])
    m2f = np.array([0.0, 0.0, 1.0])
    kb = 2.0 * np.cross(te, tf) / chi
    kappa = kb[2]
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi
    Dke = (-kappa * tilde_t + np.cross(tf, tilde_d2)) / ne
    Dkf = (-kappa * tilde_t - np.cross(te, tilde_d2)) / nf
    grad = np.zeros(6)
    grad[0:2] = -Dke[0:2]
    grad[2:4] = Dke[0:2] - Dkf[0:2]
    grad[4:6] = Dkf[0:2]

    norm2e, norm2f = ne**2, nf**2
    I3 = np.eye(3)
    tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf, tilde_d2)
    tf_d_tt = np.outer(tmp, tilde_t)
    kb_d_e = np.outer(kb, m2e)
    D2ee = ((2*kappa*tt - tf_d_tt - tf_d_tt.T) / norm2e
            - kappa/(chi*norm2e)*(I3 - np.outer(te, te))
            + (kb_d_e + kb_d_e.T)/(4*norm2e))
    tmp = np.cross(te, tilde_d2)
    te_d_tt = np.outer(tmp, tilde_t)
    tt_te_d = te_d_tt.T
    kb_d_f = np.outer(kb, m2f)
    D2ff = ((2*kappa*tt + te_d_tt + te_d_tt.T) / norm2f
            - kappa/(chi*norm2f)*(I3 - np.outer(tf, tf))
            + (kb_d_f + kb_d_f.T)/(4*norm2f))
    D2ef = (-kappa/(chi*ne*nf)*(I3 + np.outer(te, tf))
            + (2*kappa*tt - tf_d_tt + tt_te_d - cross_mat(tilde_d2))/(ne*nf))
    D2fe = D2ef.T
    DD = np.zeros((6, 6))
    a, b, c = D2ee[:2, :2], D2ef[:2, :2], D2ff[:2, :2]
    DD[0:2, 0:2] = a
    DD[0:2, 2:4] = -a + b
    DD[0:2, 4:6] = -b
    DD[2:4, 0:2] = -a + D2fe[:2, :2]
    DD[2:4, 2:4] = a - b - D2fe[:2, :2] + c
    DD[2:4, 4:6] = b - c
    DD[4:6, 0:2] = -D2fe[:2, :2]
    DD[4:6, 2:4] = D2fe[:2, :2] - c
    DD[4:6, 4:6] = c
    return kappa, grad, DD


def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """Gradient of one discrete bending energy."""
    kappa, grad_kappa, _ = _bend_geometry(
        np.array([xkm1, ykm1, xk, yk, xkp1, ykp1], dtype=float))
    return grad_kappa * EI * (kappa - curvature0) / l_k


def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """Hessian of one discrete bending energy."""
    kappa, grad_kappa, dd_kappa = _bend_geometry(
        np.array([xkm1, ykm1, xk, yk, xkp1, ykp1], dtype=float))
    dk = kappa - curvature0
    return EI / l_k * (np.outer(grad_kappa, grad_kappa) + dk * dd_kappa)


def elastic_gradient_hessian(q, EA, EI, deltaL):
    """Assemble the full elastic energy gradient and Hessian."""
    nv = q.size // 2
    grad = np.zeros_like(q)
    hess = np.zeros((q.size, q.size))
    for k in range(nv - 1):
        idx = np.arange(2*k, 2*k + 4)
        args = q[idx]
        grad[idx] += gradEs(*args, deltaL, EA)
        hess[np.ix_(idx, idx)] += hessEs(*args, deltaL, EA)
    for k in range(1, nv - 1):
        idx = np.arange(2*k - 2, 2*k + 4)
        args = q[idx]
        grad[idx] += gradEb(*args, 0.0, deltaL, EI)
        hess[np.ix_(idx, idx)] += hessEb(*args, 0.0, deltaL, EI)
    return grad, hess


def implicit_step(q_old, u_old, dt, mass, EA, EI, deltaL,
                  external=None, damping=None, fixed=None,
                  tol=1e-3, max_iter=100):
    """One implicit Euler step, solving only for free DOFs."""
    q_old = np.asarray(q_old, dtype=float)
    u_old = np.asarray(u_old, dtype=float)
    ndof = q_old.size
    external = np.zeros(ndof) if external is None else np.asarray(external, dtype=float)
    damping = np.zeros(ndof) if damping is None else np.asarray(damping, dtype=float)
    fixed = np.array([], dtype=int) if fixed is None else np.asarray(fixed, dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)
    q_new = q_old.copy()
    mass_vec = np.full(ndof, float(mass)) if np.isscalar(mass) else np.asarray(mass, dtype=float)
    mass_mat = np.diag(mass_vec)
    for _ in range(max_iter):
        grad, hess = elastic_gradient_hessian(q_new, EA, EI, deltaL)
        velocity = (q_new - q_old) / dt
        f = mass_vec * (q_new - q_old) / dt**2 - mass_vec * u_old / dt
        f += grad + damping * velocity - external
        J = mass_mat / dt**2 + hess + np.diag(damping / dt)
        if np.linalg.norm(f[free]) <= tol:
            return q_new, True
        dq = np.linalg.solve(J[np.ix_(free, free)], f[free])
        q_new[free] -= dq
    return q_new, False


def simulate_falling(nv=21, dt=1e-2, totalTime=50.0, plotShapes=False,
                     visc=1000.0, R_all=None, rho_metal=7000.0,
                     rho_fluid=1000.0, g=9.8):
    """Simulate Part 1 and return time, middle-node position/velocity, final q."""
    if nv < 3 or nv % 2 == 0:
        raise ValueError("nv must be odd and at least 3")
    length = 0.1
    deltaL = length / (nv - 1)
    radius = np.full(nv, deltaL / 10.0)
    radius[(nv - 1)//2] = 0.025
    if R_all is not None:
        radius[:] = R_all
    r0, E = 1e-3, 1e9
    EI = E * np.pi * r0**4 / 4.0
    EA = E * np.pi * r0**2
    mass = np.repeat(4.0/3.0*np.pi*radius**3*rho_metal, 2)
    weight = np.zeros(2*nv)
    weight[1::2] = -4.0/3.0*np.pi*radius**3*(rho_metal-rho_fluid)*g
    damping = np.repeat(6.0*np.pi*visc*radius, 2)
    q = np.zeros(2*nv)
    q[0::2] = np.arange(nv)*deltaL
    u = np.zeros_like(q)
    steps = round(totalTime/dt)
    times = np.arange(steps + 1)*dt
    mid = (nv - 1)//2
    y_mid = np.zeros(steps + 1)
    v_mid = np.zeros(steps + 1)
    for step in range(1, steps + 1):
        q_new, ok = implicit_step(q, u, dt, mass, EA, EI, deltaL,
                                  external=weight, damping=damping,
                                  tol=1e-3*EI/length**2)
        if not ok:
            raise RuntimeError(f"falling-beam Newton solve failed at t={times[step]:g}")
        u = (q_new - q)/dt
        q = q_new
        y_mid[step] = q[2*mid + 1]
        v_mid[step] = u[2*mid + 1]
    return times, y_mid, v_mid, q


def simply_supported_beam(P, N=50, dt=1e-2, totalTime=1.0):
    """Simulate Part 2 and return time, ymax history, and final coordinates."""
    length = 1.0
    R, r = 0.013, 0.011
    E, rho = 70e9, 2700.0
    deltaL = length/(N - 1)
    EI = E*np.pi*(R**4 - r**4)/4.0
    EA = E*np.pi*(R**2 - r**2)
    mass = np.pi*(R**2-r**2)*length*rho/(N-1)
    q = np.zeros(2*N)
    q[0::2] = np.arange(N)*deltaL
    u = np.zeros_like(q)
    external = np.zeros(2*N)
    load_node = int(np.argmin(np.abs(q[0::2] - 0.75)))
    external[2*load_node + 1] = -P
    fixed = np.array([0, 1, 2*N - 1])
    steps = round(totalTime/dt)
    times = np.arange(steps + 1)*dt
    ymax = np.zeros(steps + 1)
    for step in range(1, steps + 1):
        q_new, ok = implicit_step(q, u, dt, mass, EA, EI, deltaL,
                                  external=external, fixed=fixed,
                                  tol=1e-3*EI/length**2)
        if not ok:
            raise RuntimeError(f"supported-beam Newton solve failed at t={times[step]:g}")
        u = (q_new-q)/dt
        q = q_new
        ymax[step] = np.min(q[1::2])
    return times, ymax, q


def euler_bernoulli_deflection(P, d=0.75, length=1.0, EI=765.29):
    """Maximum downward deflection from the handout's Eq. (6)."""
    c = min(d, length-d)
    # Eq. (6): c (l^2-c^2)^(3/2) is the closed-form maximum-deflection
    # factor for a point load on a simply supported beam.
    return P*c*(length**2-c**2)**1.5/(9.0*np.sqrt(3.0)*EI*length)


def run_checks():
    """Check geometry/material values and simple analytical predictions."""
    N, length, R, r, E, rho = 50, 1.0, 0.013, 0.011, 70e9, 2700.0
    dl = length/(N-1)
    EI = E*np.pi*(R**4-r**4)/4
    EA = E*np.pi*(R**2-r**2)
    mass = np.pi*(R**2-r**2)*length*rho/(N-1)
    assert np.isclose(dl, 1/49)
    assert np.isclose(EI, 765.29, rtol=2e-4)
    assert np.isclose(EA, 1.0556e7, rtol=2e-4)
    assert np.isclose(mass, 8.309e-3, rtol=2e-4)
    q = np.zeros(6)
    q[0], q[2], q[4] = 0.0, dl, 2*dl
    assert np.allclose(gradEb(*q, 0.0, dl, EI), 0.0)
    print("Geometry, material, and straight-beam checks passed.")


def print_answers():
    """Run the handout's numerical questions and print the relevant values."""
    t, y, v, _ = simulate_falling(21, .01, 1., visc=0., R_all=.025)
    print(f"Q1 velocity: {v[-1]:.8g} m/s")
    t, y, v, _ = simulate_falling(21, .01, 1., visc=1000., R_all=.025)
    print(f"Q2 velocity: {v[-1]:.8g} m/s")
    t, y, v, _ = simulate_falling(21, .01, 50.)
    print(f"Q3 velocity: {v[-1]:.8g} m/s")
    print(f"Q4 position: {y[-1]:.8g} m")
    for n in (3, 5, 11, 21):
        _, _, vn, _ = simulate_falling(n, .01, 50.)
        print(f"Q5 N={n}: {vn[-1]:.8g} m/s")
    for load in (2000., 20000.):
        _, ym, q = simply_supported_beam(load)
        x_at_max = q[0::2][np.argmin(q[1::2])]
        print(f"P={load:g} N: deflection={abs(ym[-1]):.8g} m, x_max={x_at_max:.8g} m")
        print(f"  EB={euler_bernoulli_deflection(load):.8g} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run quick checks")
    parser.add_argument("--answers", action="store_true", help="run Gradescope simulations")
    args = parser.parse_args()
    if args.test:
        run_checks()
    elif args.answers:
        print_answers()
    else:
        t, y, v, q = simulate_falling()
        print(f"Falling beam final middle-node velocity: {v[-1]:.6g} m/s")
        _, ymax, _ = simply_supported_beam(2000.0)
        print(f"Supported beam P=2000 N final deflection: {abs(ymax[-1]):.6g} m")


if __name__ == "__main__":
    main()
