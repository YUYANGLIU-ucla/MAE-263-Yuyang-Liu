"""MAE-263F Homework 2: falling beam and simply-supported beam.

Run ``python Homework2.py`` for Q1--Q11, plots, and CSV data.
Run ``python Homework2.py --test`` for analytical and derivative checks.
"""

import argparse
from pathlib import Path

import numpy as np


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
    """Planar curvature k=2*tan(theta/2) and its exact derivatives.

    This is the 2D form of the course's discrete bending model. Writing
    derivatives through the turning angle avoids repeated 3D cross products.
    """
    nodes = np.asarray(values, dtype=float).reshape(3, 2)
    e, f = nodes[1] - nodes[0], nodes[2] - nodes[1]
    ne, nf = np.linalg.norm(e), np.linalg.norm(f)
    if ne <= 1e-14 or nf <= 1e-14:
        raise ValueError("Adjacent beam nodes cannot coincide")
    chi = 1.0 + np.dot(e, f)/(ne*nf)
    if chi <= 1e-12:
        raise ValueError("Bending angle is too close to 180 degrees")
    kappa = 2.0*(e[0]*f[1] - e[1]*f[0])/(ne*nf*chi)

    def angle_derivatives(edge):
        x, y = edge
        r2 = x*x + y*y
        grad = np.array([-y, x])/r2
        hess = np.array([[2*x*y, y*y-x*x],
                         [y*y-x*x, -2*x*y]])/r2**2
        return grad, hess

    ge, He = angle_derivatives(e)
    gf, Hf = angle_derivatives(f)
    # theta = angle(f) - angle(e); transform edge derivatives to node DOFs.
    grad_theta = np.r_[ge, -ge-gf, gf]
    hess_theta = np.zeros((6, 6))
    hess_theta[:2, :2] = -He
    hess_theta[:2, 2:4] = He
    hess_theta[2:4, :2] = He
    hess_theta[2:4, 2:4] = -He + Hf
    hess_theta[2:4, 4:6] = -Hf
    hess_theta[4:6, 2:4] = -Hf
    hess_theta[4:6, 4:6] = Hf
    factor = 1.0 + kappa*kappa/4.0  # dkappa/dtheta
    grad = factor*grad_theta
    hess = (factor*hess_theta
            + kappa*factor/2.0*np.outer(grad_theta, grad_theta))
    return kappa, grad, hess


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
        idx = slice(2*k, 2*k + 4)
        args = q[idx]
        grad[idx] += gradEs(*args, deltaL, EA)
        hess[idx, idx] += hessEs(*args, deltaL, EA)
    for k in range(1, nv - 1):
        idx = slice(2*k - 2, 2*k + 4)
        kappa, gk, Hk = _bend_geometry(q[idx])
        grad[idx] += EI/deltaL*kappa*gk
        hess[idx, idx] += EI/deltaL*(np.outer(gk, gk) + kappa*Hk)
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
    free_block = np.ix_(free, free)
    diagonal = np.diag_indices(ndof)
    for _ in range(max_iter):
        grad, hess = elastic_gradient_hessian(q_new, EA, EI, deltaL)
        velocity = (q_new - q_old) / dt
        f = mass_vec * (q_new - q_old) / dt**2 - mass_vec * u_old / dt
        f += grad + damping * velocity - external
        if np.linalg.norm(f[free]) <= tol:
            return q_new, True
        J = hess
        J[diagonal] += mass_vec/dt**2 + damping/dt
        dq = np.linalg.solve(J[free_block], f[free])
        q_new[free] -= dq
    return q_new, False


def simulate_falling(nv=21, dt=1e-2, totalTime=50.0, plotShapes=False,
                     visc=1000.0, R_all=None, rho_metal=7000.0,
                     rho_fluid=1000.0, g=9.8, progress=False):
    """Simulate Part 1 and return time, middle-node position/velocity, final q."""
    if nv < 3 or nv % 2 == 0:
        raise ValueError("nv must be odd and at least 3")
    if dt <= 0 or totalTime <= 0 or visc < 0:
        raise ValueError("dt and totalTime must be positive; visc must be nonnegative")
    length = 0.1
    deltaL = length / (nv - 1)
    radius = np.full(nv, deltaL / 10.0)
    radius[(nv - 1)//2] = 0.025
    if R_all is not None:
        if R_all <= 0:
            raise ValueError("R_all must be positive")
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
    if steps < 1 or not np.isclose(steps*dt, totalTime):
        raise ValueError("totalTime must be a positive integer multiple of dt")
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
        if progress and step % max(1, steps//5) == 0:
            print(f"  N={nv}: {times[step]:g}/{totalTime:g} s simulated", flush=True)
    if plotShapes:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(q[0::2], q[1::2], "o-")
        ax.set(xlabel="x (m)", ylabel="y (m)", title="Falling beam: final shape")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        plt.show()
    return times, y_mid, v_mid, q


def simply_supported_beam(P, N=50, dt=1e-2, totalTime=1.0):
    """Simulate Part 2 and return time, ymax history, and final coordinates."""
    if N < 3 or dt <= 0 or totalTime <= 0:
        raise ValueError("N must be at least 3; dt and totalTime must be positive")
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
    if steps < 1 or not np.isclose(steps*dt, totalTime):
        raise ValueError("totalTime must be a positive integer multiple of dt")
    times = np.arange(steps + 1)*dt
    ymax = np.zeros(steps + 1)
    for step in range(1, steps + 1):
        q_new, ok = implicit_step(q, u, dt, mass, EA, EI, deltaL,
                                  external=external, fixed=fixed,
                                  # The sweep starts at only 20 N: use a tighter
                                  # residual tolerance than the handout default.
                                  tol=1e-6*EI/length**2)
        if not ok:
            raise RuntimeError(f"supported-beam Newton solve failed at t={times[step]:g}")
        u = (q_new-q)/dt
        q = q_new
        ymax[step] = np.min(q[1::2])
    return times, ymax, q


def euler_bernoulli_deflection(P, d=0.75, length=1.0, EI=None):
    """Maximum downward deflection from the handout's Eq. (6)."""
    if EI is None:
        EI = 70e9*np.pi*(0.013**4 - 0.011**4)/4.0
    c = min(d, length-d)
    # Eq. (6): c (l^2-c^2)^(3/2) is the closed-form maximum-deflection
    # factor for a point load on a simply supported beam.
    return P*c*(length**2-c**2)**1.5/(9.0*np.sqrt(3.0)*EI*length)


def run_checks():
    """Check the handout values, exact derivatives, and analytical dynamics."""
    N, length, R, r, E, rho = 50, 1.0, 0.013, 0.011, 70e9, 2700.0
    dl = length/(N-1)
    EI = E*np.pi*(R**4-r**4)/4
    EA = E*np.pi*(R**2-r**2)
    mass = np.pi*(R**2-r**2)*length*rho/(N-1)
    assert np.isclose(dl, 1/49)
    assert np.isclose(EI, 765.29, rtol=2e-4)
    assert np.isclose(EA, 1.0556e7, rtol=2e-4)
    assert np.isclose(mass, 8.309e-3, rtol=2e-4)
    initial_x = np.arange(N)*dl
    assert np.argmin(np.abs(initial_x - 0.75)) == 37
    assert 2*37 + 1 == 75
    fixed = np.array([0, 1, 2*N - 1])
    assert len(np.setdiff1d(np.arange(2*N), fixed)) == 97
    q = np.zeros(6)
    q[0], q[2], q[4] = 0.0, dl, 2*dl
    assert np.allclose(gradEb(*q, 0.0, dl, EI), 0.0)

    # Finite differences independently check energy -> gradient -> Hessian.
    def energy(q, EA, EI, l0):
        edges = np.diff(q.reshape(-1, 2), axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        cross = edges[:-1, 0]*edges[1:, 1] - edges[:-1, 1]*edges[1:, 0]
        denom = (lengths[:-1]*lengths[1:]
                 + np.sum(edges[:-1]*edges[1:], axis=1))
        kappa = 2*cross/denom
        return (EA/(2*l0)*np.sum((lengths-l0)**2)
                + EI/(2*l0)*np.sum(kappa**2))

    for bent in (np.array([0., 0., .8, .2, 1.8, .1]),
                 np.array([0., 0., .8, -.3, 1.2, -1.1])):
        EA_test, EI_test, l0 = 12., 2., .9
        grad, hess = elastic_gradient_hessian(bent, EA_test, EI_test, l0)
        fd_grad = np.zeros(6)
        fd_hess = np.zeros((6, 6))
        for j in range(6):
            offset = np.eye(6)[j]*1e-6
            fd_grad[j] = (energy(bent+offset, EA_test, EI_test, l0)
                          - energy(bent-offset, EA_test, EI_test, l0))/2e-6
            gp, _ = elastic_gradient_hessian(bent+offset, EA_test, EI_test, l0)
            gm, _ = elastic_gradient_hessian(bent-offset, EA_test, EI_test, l0)
            fd_hess[:, j] = (gp-gm)/2e-6
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(hess, fd_hess, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(hess, hess.T, atol=1e-10)
        np.testing.assert_allclose(grad.reshape(-1, 2).sum(axis=0), 0., atol=1e-10)

    _, _, v, _ = simulate_falling(3, .01, 1., visc=0., R_all=.025)
    np.testing.assert_allclose(v[-1], -8.4, atol=1e-8)
    _, _, v, _ = simulate_falling(3, .01, 1., R_all=.025)
    sphere_terminal = -2*(7000.-1000.)*9.8*.025**2/(9*1000.)
    np.testing.assert_allclose(v[-1], sphere_terminal, atol=1e-8)
    _, ymax, q = simply_supported_beam(20.)
    np.testing.assert_array_equal(q[fixed], [0., 0., 0.])
    eb_discrete_load = euler_bernoulli_deflection(20., d=37/49)
    np.testing.assert_allclose(abs(ymax[-1]), eb_discrete_load, rtol=.005)
    print("PASS: geometry, load node, 97 free DOFs, and material values.")
    print("PASS: energy gradients/Hessians and internal-force cancellation.")
    print("PASS: buoyant free fall, Stokes velocity, supports, and small-load EB limit.")


def terminal_velocity(nv):
    """Whole-beam analytical check from total buoyant weight / total drag."""
    radii = np.full(nv, .1/(nv-1)/10.)
    radii[nv//2] = .025
    return -2*(7000.-1000.)*9.8*np.sum(radii**3)/(9*1000.*np.sum(radii))


def closest_option(value, options):
    """Select the closest numerical choice, in the handout's a/b/c/d order."""
    return "abcd"[int(np.argmin(np.abs(np.asarray(options)-value)))]


def save_plots(output_dir, small_run, loads, deflections, eb):
    """Save the two plots requested in Q8 and Q10 without opening GUI windows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, ymax, _ = small_run
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax in axes:
        ax.plot(times, ymax, "o-", markersize=3, linewidth=1.2)
        ax.axhline(ymax[-1], color="tab:orange", linestyle="--", label="Final value")
        ax.set(xlabel="Time (s)", ylabel=r"$y_{\max}=\min_k y_k$ (m)")
        ax.grid(True, alpha=.3)
    axes[0].set_title("Q8: P = 2000 N, implicit Euler")
    axes[0].legend()
    axes[1].set(xlim=(0., .15), title="Initial transient (zoom)")
    fig.savefig(output_dir / "Q8_time_history.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.plot(loads, eb, "--", label="Euler-Bernoulli, d = 0.75 m")
    ax.plot(loads, deflections, "o-", markersize=4, label="Simulation at t = 1 s")
    ax.set(xlabel="Point load P (N)", ylabel="Maximum downward deflection (m)",
           title="Q10: simulation and linear beam theory", xlim=(0., 20000.))
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.savefig(output_dir / "Q10_load_deflection.png", dpi=180)
    plt.close(fig)


def print_answers(output_dir, part="all"):
    """Compute all numerical questions; explain conceptual Q5, Q8 and Q11."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = "answers.txt" if part == "all" else f"answers_part{part}.txt"
    # Flush each answer to the terminal and report as it becomes available.
    with (output_dir / report_name).open("w", encoding="utf-8") as report:
        def emit(line=""):
            print(line, flush=True)
            report.write(line + "\n")
            report.flush()

        emit("MAE-263F Homework 2: computed results and interpretations")
        if part in ("all", "1"):
            emit("\nPart 1: N=21, dt=0.01 s (except the Q5 node sweep)")
            _, _, v, _ = simulate_falling(21, .01, 1., visc=0., R_all=.025)
            letter = closest_option(v[-1], [-11.2, -9.8, -8.4, -1.4])
            emit(f"Q1: ({letter}) v_mid(1 s) = {v[-1]:.9g} m/s; buoyant acceleration = -8.4 m/s^2.")
            _, _, v, _ = simulate_falling(21, .01, 1., R_all=.025)
            letter = closest_option(v[-1], [-9.53e-3, -8.17e-3, -5.83e-3, -1.36e-3])
            stokes = -2*(7000.-1000.)*9.8*.025**2/(9*1000.)
            emit(f"Q2: ({letter}) v_mid(1 s) = {v[-1]:.9g} m/s; Stokes prediction = {stokes:.9g} m/s.")
            print("Running Q3/Q4: 50 s falling-beam simulation...", flush=True)
            times, y, v, q = simulate_falling(progress=True)
            letter = closest_option(v[-1], [-9.53e-3, -8.17e-3, -6.81e-3, -5.83e-3])
            emit(f"Q3: ({letter}) v_mid(50 s) = {v[-1]:.9g} m/s; force-balance prediction = {terminal_velocity(21):.9g} m/s.")
            letter = closest_option(y[-1], [-2.9, -.41, -.29, -.029])
            emit(f"Q4: ({letter}) y_mid(50 s) = {y[-1]:.9g} m.")
            np.savetxt(output_dir / "falling_history.csv", np.column_stack((times, y, v)),
                       delimiter=",", header="time_s,y_mid_m,v_mid_m_per_s", comments="")
            rows = []
            for n in (3, 5, 11, 21):
                if n == 21:
                    vn = v  # Reuse Q3; this is exactly the same run.
                else:
                    print(f"Running Q5: N={n}, total time=50 s...", flush=True)
                    _, _, vn, _ = simulate_falling(n, .01, 50., progress=True)
                rows.append((n, vn[-1], terminal_velocity(n)))
                emit(f"Q5 N={n:2d}: simulated v = {vn[-1]:.9g}, analytical v = {terminal_velocity(n):.9g} m/s.")
            decreasing = np.all(np.diff(np.abs(np.asarray(rows)[:, 1])) < 0)
            emit(f"Q5: (a) magnitude decreases slightly toward 5.83e-3 m/s; decreasing in these runs: {decreasing}.")
            np.savetxt(output_dir / "Q5_discretization.csv", rows, delimiter=",",
                       header="N,simulated_velocity_m_per_s,analytical_velocity_m_per_s", comments="")

        if part in ("all", "2"):
            emit("\nPart 2: N=50, dt=0.01 s, total time=1 s, fresh straight beam for every load")
            # Include all four Q10 choices exactly, plus points spanning 20--20000 N.
            loads = np.unique(np.r_[np.geomspace(20., 20000., 18),
                                    [1000., 2000., 4000., 8000., 20000.]])
            runs = {}
            for i, load in enumerate(loads, start=1):
                print(f"  Load sweep {i}/{len(loads)}: P={load:.7g} N", flush=True)
                runs[float(load)] = simply_supported_beam(float(load))
            small_run = runs[2000.]
            times, ym, q_small = small_run
            deflection = abs(ym[-1])
            x_at_max = q_small[0::2][np.argmin(q_small[1::2])]
            letter = closest_option(deflection, [.019, .037, .074, .37])
            emit(f"Q6: ({letter}) |y_max(1 s)| = {deflection:.9g} m at P=2000 N.")
            letter = closest_option(x_at_max, [.25, .5, .56, .75])
            emit(f"Q7: ({letter}) current x-coordinate at maximum deflection = {x_at_max:.9g} m.")
            # First time after which all remaining samples stay within 1% of final.
            errors = np.abs(ym-ym[-1])
            future_max = np.maximum.accumulate(errors[::-1])[::-1]
            settled = np.flatnonzero(future_max <= .01*deflection)
            settling_time = times[settled[0]] if settled.size else np.nan
            emit(f"Q8: (b) implicit Euler introduces numerical damping; within 1% of the final value after t={settling_time:.3g} s.")
            emit("    There is no physical damping in this model. See Q8_time_history.png.")
            np.savetxt(output_dir / "Q8_time_history.csv", np.column_stack((times, ym)),
                       delimiter=",", header="time_s,y_max_m", comments="")

            _, ym_large, q_large = runs[20000.]
            letter = closest_option(abs(ym_large[-1]), [.24, .38, .57, .76])
            emit(f"Q9: ({letter}) |y_max(1 s)| = {abs(ym_large[-1]):.9g} m at P=20000 N.")
            simulated = np.array([abs(runs[float(load)][1][-1]) for load in loads])
            eb = euler_bernoulli_deflection(loads)
            relative = (eb-simulated)/eb
            emit("Q10 comparison: P (N), simulation (m), EB (m), relative difference")
            choices = (1000., 2000., 4000., 8000.)
            exceeds = []
            for load in choices:
                idx = np.flatnonzero(loads == load)[0]
                emit(f"    {load:5.0f}, {simulated[idx]:.9g}, {eb[idx]:.9g}, {relative[idx]:.3%}")
                if relative[idx] > .10:
                    exceeds.append(load)
            if exceeds:
                first = min(exceeds)
                letter = "abcd"[choices.index(first)]
                emit(f"Q10: ({letter}) smallest listed load with e(P)>10% is {first:g} N.")
            else:
                emit("Q10: no listed load exceeds 10%; inspect the model and convergence.")
            emit("    EB uses d=0.75 m; the discrete load is at 37/49=0.755102 m.")
            emit("    See Q10_load_deflection.png for the complete 20--20000 N sweep.")
            np.savetxt(output_dir / "Q10_load_sweep.csv",
                       np.column_stack((loads, simulated, eb, relative)), delimiter=",",
                       header="load_N,simulation_m,EB_m,relative_difference", comments="")

            edge_lengths = np.linalg.norm(np.diff(q_large.reshape(-1, 2), axis=0), axis=1)
            extension = edge_lengths - 1/49
            emit("Q11: (a) linear EB assumes small slopes and undeformed geometry;")
            emit("     the simulation includes finite rotations and the moving roller.")
            emit(f"     At P=20000 N: EB={euler_bernoulli_deflection(20000.):.9g} m,")
            emit(f"     simulation={abs(ym_large[-1]):.9g} m; last-node x={q_large[-2]:.9g} m (initially 1 m).")
            emit(f"     Max |segment length change|={np.max(np.abs(extension)):.6g} m;")
            emit(f"     max |axial strain|={np.max(np.abs(extension))*49:.6g}.")
            np.savetxt(output_dir / "Q11_segment_extension.csv",
                       np.column_stack((np.arange(49), edge_lengths, extension, extension*49)),
                       delimiter=",", header="segment_index,length_m,extension_m,axial_strain", comments="")
            save_plots(output_dir, small_run, loads, simulated, eb)
            emit("\nSaved Q8/Q10 plots and CSV data.")
        emit(f"Results folder: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true", help="run analytical and derivative checks")
    mode.add_argument("--answers", action="store_true", help="run all results (also the default)")
    parser.add_argument("--part", choices=("all", "1", "2"), default="all",
                        help="run both parts (default), or only Part 1 / Part 2")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results",
                        help="folder for answers, plots, and CSV data (default: results beside this script)")
    args = parser.parse_args()
    if args.test:
        run_checks()
    else:
        print_answers(args.output_dir, args.part)


if __name__ == "__main__":
    main()
