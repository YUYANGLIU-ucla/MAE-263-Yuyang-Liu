"""
snap.py -- building blocks for the bistable-beam study: buckling, point-force loading,
curvature actuation, snap dynamics.  Uses beam.py.  MAE 263F (Fall 2026), Yuyang Liu.
"""
import numpy as np
from beam import Beam, energy, grad_hess, static_solve, advance, step_trapezoid, curvature


# ----------------------------------------------------------------------------------
# 1. Buckled (open) state
# ----------------------------------------------------------------------------------
def relax(beam, q, k0=None, Fext=None, nmax=4000):
    """
    Go from q to a nearby STABLE equilibrium: heavily damped implicit Euler steps
    (Algorithm 1 used as dynamic relaxation), then a static Newton polish.
    """
    gamma = 20.0 / beam.tstar
    dt = beam.tstar / 20
    u = np.zeros(beam.ndof)
    for n in range(nmax):
        q_new, u, g = advance(beam, q, u, dt, k0=k0, gamma=gamma, Fext=Fext)
        moved = np.max(np.abs(q_new - q))
        q = q_new
        if moved < 1e-9 * beam.L:
            break
    return static_solve(beam, q, k0=k0, Fext=Fext)


def buckle(beam, eps, nsteps=20):
    """
    Compress the straight strip by dL = eps*L in static steps and return the buckled
    'open' state (arch on the +y side) and the axial compressive force.
    The straight strip stays an equilibrium after the Euler load, but an unstable one
    (lam_min < 0).  When that happens the state is pushed along the unstable
    eigenvector towards +y and relaxed to the stable arch.
    """
    q = beam.straight()
    free = beam.free
    for fr in np.linspace(0, 1, nsteps + 1)[1:]:
        q = beam.clamp(q, fr * eps * beam.L)
        q, ok, lam, g = static_solve(beam, q)
        if ok and lam < 0:
            H = grad_hess(beam, q)[1][np.ix_(free, free)]
            vec = np.zeros(beam.ndof)
            vec[free] = np.linalg.eigh(H)[1][:, 0]
            if vec[2 * beam.mid + 1] < 0:
                vec = -vec
            q = q + 0.01 * beam.L * vec / np.max(np.abs(vec))
            q, ok, lam, g = relax(beam, q)
            if beam.ymid(q) < 0:          # landed on the -y side: use the mirror image
                q[1::2] *= -1
                g[1::2] *= -1
        if not ok:
            raise RuntimeError(f'buckle: no convergence at fraction {fr:.3f}')
    if lam <= 0 or beam.ymid(q) <= 0:
        raise RuntimeError('buckle: final state is not the stable open arch')
    P = -(g[beam.ndof - 2] + g[beam.ndof - 4])       # axial compressive force
    return q, P


def mirror(beam, q):
    """Mirror image about the clamp line: the 'closed' state of an 'open' state."""
    qm = q.copy()
    qm[1::2] *= -1
    return qm


def natural_frequencies(beam, q, k0=None, nmodes=4):
    """Angular frequencies of small vibrations about an equilibrium q [rad/s]."""
    g, H = grad_hess(beam, q, k0)
    f = beam.free
    sm = 1 / np.sqrt(beam.m[f])
    A = H[np.ix_(f, f)] * sm[:, None] * sm[None, :]
    lam = np.linalg.eigvalsh(A)
    return np.sqrt(np.maximum(lam[:nmodes], 0))


# ----------------------------------------------------------------------------------
# 2. Point force (used for validation and for the holding force)
# ----------------------------------------------------------------------------------
def critical_point_force(beam, q_open, delta=0.5, dF0=None, rtol=1e-4):
    """
    Push the open arch down with a vertical dead load at arc-length fraction delta and
    return the load at which the open state loses stability (lam_min -> 0), i.e. the
    snap-through force under force control.  Also returns the state just before.
    """
    node = int(round(delta * beam.N)) + 1
    Fscale = beam.EI / beam.L**2
    dF = dF0 if dF0 is not None else 0.5 * Fscale
    F, q = 0.0, q_open.copy()
    ymid0 = beam.ymid(q_open)
    while dF > rtol * Fscale:
        Fext = np.zeros(beam.ndof)
        Fext[2 * node + 1] = -(F + dF)
        qt, ok, lam, g = static_solve(beam, q, Fext=Fext)
        jumped = (not ok) or lam <= 0 or abs(beam.ymid(qt) - beam.ymid(q)) > 0.25 * abs(ymid0)
        if jumped:
            dF /= 2                       # reject and retry with a smaller increment
        else:
            F, q = F + dF, qt
    return F, q


def indentation_curve(beam, q_open, ws=None):
    """
    Displacement control: prescribe the height of the midpoint (values ws, default from
    w0 down to -w0, with small steps at the start where the branch switches) and record the vertical reaction force F (positive = pushing down is resisted).
    When the symmetric branch becomes unstable the state is perturbed along the
    unstable eigenvector so that the solver follows the stable (asymmetric) branch.
    Returns arrays w, F, elastic energy, and a flag (1 = asymmetric branch).
    """
    dof = 2 * beam.mid + 1
    free = np.setdiff1d(beam.free, [dof])
    w0 = beam.ymid(q_open)
    if ws is None:
        ws = w0 * np.concatenate([np.linspace(1, 0.9, 41), np.linspace(0.9, -0.9, 91)[1:],
                                  np.linspace(-0.9, -1, 41)[1:]])
    q = q_open.copy()
    out = []
    for w in ws:
        q[dof] = w
        q, ok, lam, g = static_solve(beam, q, fixed_extra=[dof])
        if ok and lam < 0:                        # unstable: switch branch
            gH = grad_hess(beam, q)[1][np.ix_(free, free)]
            vec = np.linalg.eigh(gH)[1][:, 0]
            qp = q.copy()
            qp[free] += 0.02 * beam.L * vec / np.max(np.abs(vec)) * 0.05
            qp, ok2, lam2, g2 = static_solve(beam, qp, fixed_extra=[dof])
            if ok2 and lam2 > 0:
                q, lam, g = qp, lam2, g2
        x = q.reshape(-1, 2)
        asym = np.max(np.abs(x[:, 1] - x[::-1, 1])) > 1e-6 * beam.L
        out.append((w, g[dof], energy(beam, q), float(asym), lam))
    return np.array(out).T


# ----------------------------------------------------------------------------------
# 3. Curvature actuation: quasi-static trigger
# ----------------------------------------------------------------------------------
def command_sign(beam, q_open, s_a, l_a):
    """
    Heuristic sign of the command: drive the patch towards the curvature it has in the
    closed state, i.e. opposite to its (patch-averaged) curvature in the open state.
    It fails when that average is close to zero (patch centred near s = 1/4), so
    design.evaluate() tries both signs instead.
    """
    w = beam.patch_weights(s_a, l_a)
    return -np.sign(np.sum(w * curvature(beam, q_open))) or 1.0


def trigger(beam, q_open, s_a, l_a, dk0=0.25, kmax=40.0, rtol=1e-4, sign=None):
    """
    Ramp the commanded curvature kbar of the patch quasi-statically (static solves,
    adaptive increments) until the open state loses stability (lam_min -> 0).
    sign = +1 or -1 fixes the direction of the command; None uses command_sign().
    Returns dict:
      kcrit  : |kbar| * L at loss of stability (None if not reached below kmax),
      sign   : sign of the command,
      W_act  : gross actuator work up to kcrit [J]: eq. (3) of the proposal, counting
               only the increments in which the actuator gives energy to the strip
               (an actuator cannot recover the energy that the strip gives back),
      W_net  : net work, eq. (3) as written; equals dE for a quasi-static ramp,
      dE     : change of elastic energy [J],
      strain : largest bending strain h |kappa_k| / (2 l_k) along the path,
      path   : rows (kbar*L, ymid, lam_min),  q : state at kcrit.
    """
    L = beam.L
    if sign is None:
        sign = command_sign(beam, q_open, s_a, l_a)
    coef = beam.EI / beam.lk
    k, q, dk = 0.0, q_open.copy(), dk0
    W_net, W_gross = 0.0, 0.0
    E0 = energy(beam, q)
    ymid0 = beam.ymid(q_open)
    path = [(0.0, ymid0, np.nan)]
    lam0 = None
    kap = curvature(beam, q)
    strain = np.max(np.abs(kap) / beam.lk) * beam.h / 2
    while dk > rtol and k < kmax:
        k0_old = beam.kappa0(s_a, l_a, sign * k / L)
        k0_new = beam.kappa0(s_a, l_a, sign * (k + dk) / L)
        qt, ok, lam, g = static_solve(beam, q, k0=k0_new)
        jumped = (not ok) or lam <= 0 or abs(beam.ymid(qt) - beam.ymid(q)) > 0.25 * abs(ymid0)
        if jumped:
            dk /= 2                       # reject the increment and try a smaller one
            continue
        kap_new = curvature(beam, qt)
        # actuator work: sum_k EI/l_k (kappa0_k - kappa_k) d kappa0_k, trapezoidal rule
        mom_old = coef * (k0_old - kap)
        mom_new = coef * (k0_new - kap_new)
        inc = 0.5 * (mom_old + mom_new) * (k0_new - k0_old)      # work increment per node
        W_net += np.sum(inc)
        W_gross += np.sum(np.maximum(inc, 0.0))
        k, q, kap = k + dk, qt, kap_new
        strain = max(strain, np.max(np.abs(kap) / beam.lk) * beam.h / 2)
        path.append((k, beam.ymid(q), lam))
        if lam0 is None:
            lam0 = lam
        if lam < 0.02 * lam0:
            dk = min(dk, 0.02)            # small increments close to the instability
    reached = k < kmax
    dE = energy(beam, q, beam.kappa0(s_a, l_a, sign * k / L)) - E0
    return dict(kcrit=k if reached else None, sign=sign, W_act=W_gross if reached else None,
                W_net=W_net if reached else None, dE=dE if reached else None, strain=strain, path=np.array(path), q=q)


# ----------------------------------------------------------------------------------
# 4. Snap dynamics
# ----------------------------------------------------------------------------------
def settle_and_release(beam, q, k0_cmd):
    """
    Quasi-static end of a switching cycle: hold the command until the strip has settled
    (state q_on), then switch the actuator off and let it settle again (state q_off).
    Returns q_on, q_off.
    """
    q_on = relax(beam, q, k0=k0_cmd)[0]
    q_off = relax(beam, q_on)[0]
    return q_on, q_off


def snap_dynamics(beam, q_open, k0_cmd, q_start=None, w_target=None, zeta=0.02,
                  dt_frac=1 / 4000, t_max=3.0, hold=0.0, record=False, scheme='euler',
                  store_every=0):
    """
    Step command: at t = 0 the natural curvature jumps to k0_cmd (array from
    beam.kappa0) and stays there.  The strip starts at rest from q_start (default: the
    open state).  Time stepping: Algorithm 1 (implicit Euler), dt = dt_frac * tstar;
    scheme='trapezoid' uses the trapezoidal rule instead (cross-check).
    zeta = damping ratio of the first vibration mode of the open arch.
    w_target = midpoint height that counts as "closed"; default: the height of the
    settled state with the command on (from settle_and_release).
    Returns dict (times in units of tstar):
      t_snap : time at which the midpoint first reaches w_target (None if never),
      hist   : rows t, ymid, kinetic energy, elastic energy, dissipated energy,
      q      : last state,
      shapes : list of (t, q) every store_every steps (if store_every > 0).
    """
    dt = dt_frac * beam.tstar
    if w_target is None:
        w_target = beam.ymid(settle_and_release(beam, q_open if q_start is None else q_start, k0_cmd)[0])
    w1 = natural_frequencies(beam, q_open, nmodes=1)[0]
    gamma = 2 * zeta * w1
    q = q_open.copy() if q_start is None else q_start.copy()
    u = np.zeros(beam.ndof)
    t_snap, n_stop, diss = None, None, 0.0
    hist, shapes = [], []
    for n in range(int(round(t_max / dt_frac))):
        if scheme == 'euler':
            q, u, g = advance(beam, q, u, dt, k0=k0_cmd, gamma=gamma)
            diss += gamma * np.sum(beam.m * u**2) * dt
        else:                               # trapezoidal rule, cross-check only
            u_prev = u
            q, u, g = step_trapezoid(beam, q, u, dt, k0=k0_cmd, gamma=gamma)
            diss += gamma * np.sum(beam.m * (0.5 * (u + u_prev))**2) * dt
        if record:
            hist.append(((n + 1) * dt_frac, beam.ymid(q), 0.5 * np.sum(beam.m * u**2),
                         energy(beam, q, k0_cmd), diss))
        if store_every and (n + 1) % store_every == 0:
            shapes.append(((n + 1) * dt_frac, q.copy()))
        if t_snap is None and beam.ymid(q) <= w_target:
            t_snap = (n + 1) * dt_frac
            n_stop = n + int(round(hold / dt_frac))
        if n_stop is not None and n >= n_stop:
            break
    return dict(t_snap=t_snap, hist=np.array(hist).T if record else None, q=q, w_target=w_target,
                shapes=shapes)
