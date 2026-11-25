"""
beam.py -- planar discrete elastic beam for the bistable snap-through project.
MAE 263F (Fall 2026), Yuyang Liu.

Model (same energies as the course notes, "Discrete Simulation of Slender Structures"):
    E = sum_edges 1/2 EA dl (|e|/dl - 1)^2  +  sum_nodes 1/2 EI/l_k (kappa_k - kappa0_k)^2
    kappa_k = 2 tan(phi_k / 2),  phi_k = signed turning angle at interior node k.

The formulas are the planar case of the course routines gradEs/hessEs/gradEb/hessEb
(https://github.com/StructuresComp/mae-263F, folder "appendix"). Here they are written
for all edges / nodes at once with NumPy arrays, because the design study needs many
thousands of solves. test_against_course() checks them against the course routines.

DOF layout: q = [x_0, y_0, x_1, y_1, ...].  The beam has N free edges between the
clamps plus one clamped edge at each end, so N + 3 nodes.  Nodes 0, 1 (left clamp) and
N+1, N+2 (right clamp) are prescribed.
"""
from dataclasses import dataclass, field
import numpy as np


# ----------------------------------------------------------------------------------
# Beam definition
# ----------------------------------------------------------------------------------
@dataclass
class Beam:
    N: int = 40             # number of edges in the free span
    L: float = 0.10         # free span of the unbuckled strip [m]
    b: float = 0.010        # width [m]
    h: float = 0.0005       # thickness [m]
    E: float = 3.0e9        # Young's modulus [Pa]
    rho: float = 1380.0     # density [kg/m^3]
    half_clamp: bool = True  # Voronoi length dl/2 at the two clamp nodes (see README)
    # derived quantities (filled in __post_init__)
    dl: float = field(init=False)
    nv: int = field(init=False)
    ndof: int = field(init=False)
    EI: float = field(init=False)
    EA: float = field(init=False)
    lk: np.ndarray = field(init=False)      # Voronoi length of interior nodes 1..nv-2
    m: np.ndarray = field(init=False)       # lumped mass per DOF
    fixed: np.ndarray = field(init=False)
    free: np.ndarray = field(init=False)
    mid: int = field(init=False)            # index of the midpoint node
    tstar: float = field(init=False)        # bending time scale L^2 sqrt(rho A / EI)
    s: np.ndarray = field(init=False)       # arc length of each node / L (0 at left clamp)

    def __post_init__(self):
        assert self.N % 2 == 0, 'N must be even so that a midpoint node exists'
        self.dl = self.L / self.N
        self.nv = self.N + 3
        self.ndof = 2 * self.nv
        self.EI = self.E * self.b * self.h**3 / 12
        self.EA = self.E * self.b * self.h
        # Interior nodes 1 .. nv-2 carry a bending spring.  At the clamp nodes (1 and
        # nv-2) one of the two half-edges lies inside the rigid clamp, so only dl/2 of
        # the Voronoi cell can bend.
        self.lk = np.full(self.nv - 2, self.dl)
        if self.half_clamp:
            self.lk[0] = self.lk[-1] = self.dl / 2
        node_mass = np.full(self.nv, self.rho * self.b * self.h * self.dl)
        node_mass[0] = node_mass[-1] = node_mass[0] / 2
        self.m = np.repeat(node_mass, 2)
        self.fixed = np.array([0, 1, 2, 3, self.ndof - 4, self.ndof - 3, self.ndof - 2, self.ndof - 1])
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        self.mid = self.nv // 2
        self.tstar = self.L**2 * np.sqrt(self.rho * self.b * self.h / self.EI)
        self.s = (np.arange(self.nv) - 1) / self.N

    # -- helpers ------------------------------------------------------------------
    def straight(self):
        """Straight, unstressed configuration: node 1 at x = 0, node nv-2 at x = L."""
        q = np.zeros(self.ndof)
        q[0::2] = (np.arange(self.nv) - 1) * self.dl
        return q

    def clamp(self, q, dL):
        """Return a copy of q with the right clamp moved towards the left one by dL."""
        q = q.copy()
        q[self.ndof - 4] = self.L - dL
        q[self.ndof - 2] = self.L - dL + self.dl
        q[self.ndof - 3] = q[self.ndof - 1] = 0.0
        q[0], q[2] = -self.dl, 0.0
        q[1] = q[3] = 0.0
        return q

    def patch_weights(self, s_a, l_a):
        """
        Length of each interior node's Voronoi cell that lies inside the actuator
        patch [s_a - l_a/2, s_a + l_a/2] * L.  The discrete natural curvature of a node
        is the commanded curvature integrated over this length, so the actuation
        varies continuously with s_a and l_a.
        """
        s_int = self.s[1:-1] * self.L
        lo_cell = s_int - self.lk / 2
        hi_cell = s_int + self.lk / 2
        if self.half_clamp:                     # clamp nodes: only the free half bends
            lo_cell[0], hi_cell[0] = s_int[0], s_int[0] + self.lk[0]
            lo_cell[-1], hi_cell[-1] = s_int[-1] - self.lk[-1], s_int[-1]
        lo, hi = (s_a - l_a / 2) * self.L, (s_a + l_a / 2) * self.L
        return np.clip(np.minimum(hi_cell, hi) - np.maximum(lo_cell, lo), 0.0, None)

    def kappa0(self, s_a, l_a, kbar):
        """Discrete natural curvature of every interior node for a command kbar [1/m]."""
        return kbar * self.patch_weights(s_a, l_a)

    def ymid(self, q):
        return q[2 * self.mid + 1]


# ----------------------------------------------------------------------------------
# Elastic energy, gradient, Hessian (vectorized planar formulas)
# ----------------------------------------------------------------------------------
def curvature(beam, q):
    """Discrete curvature kappa_k = 2 tan(phi_k/2) at the interior nodes."""
    x = q.reshape(-1, 2)
    e = x[1:-1] - x[:-2]       # edge entering each interior node
    f = x[2:] - x[1:-1]        # edge leaving each interior node
    ne = np.linalg.norm(e, axis=1)
    nf = np.linalg.norm(f, axis=1)
    c = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]      # |e||f| sin(phi)
    d = ne * nf + np.sum(e * f, axis=1)            # |e||f| (1 + cos(phi))
    return 2 * c / d


def energy(beam, q, k0=None):
    """Total elastic energy E_s + E_b."""
    x = q.reshape(-1, 2)
    edge = x[1:] - x[:-1]
    lam = np.linalg.norm(edge, axis=1) / beam.dl
    Es = 0.5 * beam.EA * beam.dl * np.sum((lam - 1) ** 2)
    kap = curvature(beam, q)
    if k0 is None:
        k0 = 0.0
    Eb = 0.5 * beam.EI * np.sum((kap - k0) ** 2 / beam.lk)
    return Es + Eb


def grad_hess(beam, q, k0=None, need_hess=True):
    """Gradient (ndof) and Hessian (ndof x ndof) of the elastic energy."""
    nv, ndof = beam.nv, beam.ndof
    x = q.reshape(-1, 2)
    g = np.zeros(ndof)
    H = np.zeros((ndof, ndof)) if need_hess else None
    I2 = np.eye(2)

    # ---- stretching: one spring per edge, local DOFs (x_k, y_k, x_k+1, y_k+1) ----
    edge = x[1:] - x[:-1]
    n = np.linalg.norm(edge, axis=1)
    t = edge / n[:, None]
    strain = n / beam.dl - 1
    ge = beam.EA * strain[:, None] * t                    # dE/d(edge vector)
    ke = np.arange(nv - 1)
    np.add.at(g, 2 * ke, -ge[:, 0]); np.add.at(g, 2 * ke + 1, -ge[:, 1])
    np.add.at(g, 2 * ke + 2, ge[:, 0]); np.add.at(g, 2 * ke + 3, ge[:, 1])
    if need_hess:
        tt = t[:, :, None] * t[:, None, :]
        He = beam.EA * (tt / beam.dl + (strain / n)[:, None, None] * (I2 - tt))   # d2E/d(edge)^2
        for a in range(2):
            for c_ in range(2):
                np.add.at(H, (2 * ke + a, 2 * ke + c_), He[:, a, c_])
                np.add.at(H, (2 * ke + 2 + a, 2 * ke + 2 + c_), He[:, a, c_])
                np.add.at(H, (2 * ke + a, 2 * ke + 2 + c_), -He[:, a, c_])
                np.add.at(H, (2 * ke + 2 + a, 2 * ke + c_), -He[:, a, c_])

    # ---- bending: one spring per interior node, local DOFs of nodes k-1, k, k+1 ----
    e = x[1:-1] - x[:-2]
    f = x[2:] - x[1:-1]
    ne = np.linalg.norm(e, axis=1)
    nf = np.linalg.norm(f, axis=1)
    c = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
    d = ne * nf + np.sum(e * f, axis=1)
    kap = 2 * c / d
    # first derivatives of c and d with respect to e and f
    c_e = np.stack([f[:, 1], -f[:, 0]], axis=1)
    c_f = np.stack([-e[:, 1], e[:, 0]], axis=1)
    d_e = (nf / ne)[:, None] * e + f
    d_f = (ne / nf)[:, None] * f + e
    k_e = 2 * (c_e / d[:, None] - (c / d**2)[:, None] * d_e)
    k_f = 2 * (c_f / d[:, None] - (c / d**2)[:, None] * d_f)
    gk = np.concatenate([-k_e, k_e - k_f, k_f], axis=1)   # d kappa / d(6 local DOFs)
    if k0 is None:
        k0 = np.zeros(nv - 2)
    dk = kap - k0
    coef = beam.EI / beam.lk
    kb = np.arange(nv - 2)                                # interior node k = kb + 1
    loc = 2 * kb[:, None] + np.arange(6)[None, :]         # global DOFs of each local triple
    np.add.at(g, loc, (coef * dk)[:, None] * gk)

    if need_hess:
        te = e / ne[:, None]
        tf = f / nf[:, None]
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])           # d2c / de df
        d_ee = (nf / ne)[:, None, None] * (I2 - te[:, :, None] * te[:, None, :])
        d_ff = (ne / nf)[:, None, None] * (I2 - tf[:, :, None] * tf[:, None, :])
        d_ef = te[:, :, None] * tf[:, None, :] + I2

        def second(c_a, c_b, d_a, d_b, c_ab, d_ab):
            """d2(2c/d)/da db for vector arguments a, b."""
            return 2 * (c_ab / d[:, None, None]
                        - (c_a[:, :, None] * d_b[:, None, :] + d_a[:, :, None] * c_b[:, None, :]) / (d**2)[:, None, None]
                        - (c / d**2)[:, None, None] * d_ab
                        + 2 * (c / d**3)[:, None, None] * d_a[:, :, None] * d_b[:, None, :])

        zero = np.zeros((nv - 2, 2, 2))
        k_ee = second(c_e, c_e, d_e, d_e, zero, d_ee)
        k_ff = second(c_f, c_f, d_f, d_f, zero, d_ff)
        k_ef = second(c_e, c_f, d_e, d_f, np.broadcast_to(J, (nv - 2, 2, 2)), d_ef)
        k_fe = np.transpose(k_ef, (0, 2, 1))
        DD = np.zeros((nv - 2, 6, 6))                     # d2 kappa / d(local DOFs)^2
        DD[:, 0:2, 0:2] = k_ee
        DD[:, 0:2, 2:4] = -k_ee + k_ef
        DD[:, 0:2, 4:6] = -k_ef
        DD[:, 2:4, 0:2] = -k_ee + k_fe
        DD[:, 2:4, 2:4] = k_ee - k_ef - k_fe + k_ff
        DD[:, 2:4, 4:6] = k_ef - k_ff
        DD[:, 4:6, 0:2] = -k_fe
        DD[:, 4:6, 2:4] = k_fe - k_ff
        DD[:, 4:6, 4:6] = k_ff
        Hb = coef[:, None, None] * (gk[:, :, None] * gk[:, None, :] + dk[:, None, None] * DD)
        np.add.at(H, (loc[:, :, None], loc[:, None, :]), Hb)
    return g, H


# ----------------------------------------------------------------------------------
# Static equilibrium (Newton with backtracking) and stability
# ----------------------------------------------------------------------------------
def static_solve(beam, q, k0=None, Fext=None, fixed_extra=(), tol=None, maxit=60):
    """
    Solve grad E(q) = Fext on the free DOFs, starting from q (fixed DOFs keep their
    values in q).  fixed_extra: additional DOFs to hold (displacement control).
    Returns q, converged, lam_min (smallest eigenvalue of the free Hessian; > 0 = stable),
    and the gradient at the solution (its fixed entries are the support reactions).
    """
    free = np.setdiff1d(beam.free, np.asarray(fixed_extra, dtype=int))
    if Fext is None:
        Fext = np.zeros(beam.ndof)
    if tol is None:
        tol = 1e-7 * beam.EI / beam.L**2
    q = q.copy()
    g, H = grad_hess(beam, q, k0)
    r = (g - Fext)[free]
    ok = False
    for it in range(maxit):
        if np.linalg.norm(r) < tol:
            ok = True
            break
        try:
            dq = np.linalg.solve(H[np.ix_(free, free)], r)
        except np.linalg.LinAlgError:
            break
        step = 1.0
        while True:                                   # backtracking on the residual norm
            qt = q.copy()
            qt[free] -= step * dq
            gt, Ht = grad_hess(beam, qt, k0)
            rt = (gt - Fext)[free]
            if np.all(np.isfinite(rt)) and np.linalg.norm(rt) < (1 - 1e-4 * step) * np.linalg.norm(r):
                break
            step /= 2
            if step < 1e-6:
                break
        if step < 1e-6:
            break
        q, g, H, r = qt, gt, Ht, rt
    lam_min = np.linalg.eigvalsh(H[np.ix_(free, free)])[0]
    return q, ok, lam_min, g


# ----------------------------------------------------------------------------------
# Dynamics: Algorithm 1 of the proposal (implicit Euler + Newton, step halving)
# ----------------------------------------------------------------------------------
def step_implicit(beam, q_old, u_old, dt, k0=None, q_fixed=None, gamma=0.0, Fext=None,
                  tol=None, maxit=40):
    """One implicit Euler step.  Raises RuntimeError if Newton does not converge."""
    free, m = beam.free, beam.m
    if Fext is None:
        Fext = np.zeros(beam.ndof)
    if tol is None:
        tol = 1e-7 * beam.EI / beam.L**2
    q = q_old.copy()
    if q_fixed is not None:
        q[beam.fixed] = q_fixed
    mass_diag = m[free] / dt**2 + gamma * m[free] / dt
    for it in range(maxit):
        g, H = grad_hess(beam, q, k0)
        f = m * ((q - q_old) / dt - u_old) / dt + gamma * m * (q - q_old) / dt + g - Fext
        if np.linalg.norm(f[free]) < tol:
            return q, (q - q_old) / dt, g
        Jff = H[np.ix_(free, free)]
        Jff[np.diag_indices_from(Jff)] += mass_diag
        q[free] -= np.linalg.solve(Jff, f[free])
        if not np.all(np.isfinite(q)):
            break
    raise RuntimeError('Newton did not converge')


def step_trapezoid(beam, q_old, u_old, dt, k0=None, gamma=0.0, tol=None, maxit=40):
    """
    One step of the trapezoidal rule (Newmark average acceleration).  Unlike implicit
    Euler it has no numerical dissipation, so it is used only as a cross-check of the
    snap time and of the energy balance:
        q_new = q_old + dt/2 (u_old + u_new)
        M (u_new - u_old)/dt + C (u_old + u_new)/2 + (grad E(q_old) + grad E(q_new))/2 = 0
    """
    free, m = beam.free, beam.m
    if tol is None:
        tol = 1e-7 * beam.EI / beam.L**2
    g_old = grad_hess(beam, q_old, k0, need_hess=False)[0]
    q = q_old.copy()
    q[free] += dt * u_old[free]
    for it in range(maxit):
        g, H = grad_hess(beam, q, k0)
        u = 2 * (q - q_old) / dt - u_old
        f = m * (u - u_old) / dt + gamma * m * 0.5 * (u + u_old) + 0.5 * (g + g_old)
        if np.linalg.norm(f[free]) < tol:
            return q, u, g
        Jff = 0.5 * H[np.ix_(free, free)]
        Jff[np.diag_indices_from(Jff)] += 2 * m[free] / dt**2 + gamma * m[free] / dt
        q[free] -= np.linalg.solve(Jff, f[free])
    raise RuntimeError('Newton did not converge (trapezoidal rule)')


def advance(beam, q_old, u_old, dt, depth=0, **kw):
    """Implicit Euler step; if Newton fails, take two half steps instead (up to 2^-8 dt)."""
    try:
        return step_implicit(beam, q_old, u_old, dt, **kw)
    except (RuntimeError, np.linalg.LinAlgError):
        if depth >= 8:
            raise
        kw1 = dict(kw)
        if kw.get('q_fixed') is not None:
            kw1['q_fixed'] = 0.5 * (q_old[beam.fixed] + kw['q_fixed'])
        q1, u1, _ = advance(beam, q_old, u_old, dt / 2, depth + 1, **kw1)
        return advance(beam, q1, u1, dt / 2, depth + 1, **kw)


# ----------------------------------------------------------------------------------
# Check against the course routines
# ----------------------------------------------------------------------------------
def test_against_course(seed=0):
    """Max difference between this file and the course gradEb/hessEb/gradEs/hessEs."""
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'course'))
    from gradEb import gradEb
    from hessEb import hessEb
    from gradEs import gradEs
    from hessEs import hessEs
    rng = np.random.default_rng(seed)
    beam = Beam(N=10, half_clamp=False)
    q = beam.straight() + 0.2 * beam.dl * rng.standard_normal(beam.ndof)
    k0 = 0.3 * rng.standard_normal(beam.nv - 2)
    g, H = grad_hess(beam, q, k0)
    g_ref = np.zeros(beam.ndof); H_ref = np.zeros((beam.ndof, beam.ndof))
    for k in range(beam.nv - 1):
        ind = np.arange(2 * k, 2 * k + 4)
        g_ref[ind] += gradEs(*q[ind], beam.dl, beam.EA)
        H_ref[np.ix_(ind, ind)] += hessEs(*q[ind], beam.dl, beam.EA)
    for k in range(1, beam.nv - 1):
        ind = np.arange(2 * k - 2, 2 * k + 4)
        g_ref[ind] += gradEb(*q[ind], k0[k - 1], beam.dl, beam.EI)
        H_ref[np.ix_(ind, ind)] += hessEb(*q[ind], k0[k - 1], beam.dl, beam.EI)
    return (np.max(np.abs(g - g_ref)) / np.max(np.abs(g_ref)),
            np.max(np.abs(H - H_ref)) / np.max(np.abs(H_ref)))


if __name__ == '__main__':
    eg, eH = test_against_course()
    print(f'relative difference to the course routines: gradient {eg:.2e}, Hessian {eH:.2e}')
