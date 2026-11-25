"""
design.py -- design evaluation, grid maps and constrained search (Algorithm 2 of the
proposal).  MAE 263F (Fall 2026), Yuyang Liu.

    python design.py grid      # evaluate the grid of designs  -> results/grid.csv   (~20 min)
    python design.py search    # Nelder-Mead search            -> results/search.json (~15 min)

Design variables d = (eps, s_a, l_a):
    eps  end shortening / L,  s_a  patch centre / L,  l_a  patch length / L.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
import numpy as np
from beam import Beam
from snap import buckle, critical_point_force, indentation_curve, trigger, snap_dynamics, settle_and_release

# ---- requirements of the gripper element (design choices) ---------------------------
F_REQ = 0.5          # required holding force [N]
T_MAX = 0.010        # required closing time [s]
STRAIN_LIM = 0.01    # allowed bending strain
DELTA = 0.10         # the command is (1 + DELTA) times the trigger curvature
BOUNDS = dict(eps=(0.005, 0.05), l_a=(0.10, 0.30))     # s_a between l_a/2 and 0.5

_open = {}


def open_state(beam, eps):
    """Buckled open state for this end shortening (cached)."""
    key = (beam.N, round(eps, 10))
    if key not in _open:
        _open[key] = buckle(beam, eps)[0]
    return _open[key]


def holding(beam, eps, barrier=False):
    """Holding force [N] of the closed state = snap-through force of the midpoint."""
    qo = open_state(beam, eps)
    F = critical_point_force(beam, qo, 0.5)[0]
    if not barrier:
        return F
    w, Fw, E, asym, lam = indentation_curve(beam, qo)
    return F, E.max() - E[0]


def evaluate(beam, eps, s_a, l_a, delta=DELTA, dt_frac=1 / 4000):
    """All performance measures of one design (function EVALUATE of Algorithm 2)."""
    qo = open_state(beam, eps)
    res = dict(eps=eps, s_a=s_a, l_a=l_a, F_hold=holding(beam, eps), w0=beam.ymid(qo),
               kcrit=np.nan, sign=0.0, W_act=np.nan, W_net=np.nan, strain=np.nan, w_on=np.nan, t_snap=np.nan, switched=0.0)
    # try both directions of the command and keep the one that needs less work
    best = None
    for sgn in (+1.0, -1.0):
        t_ = trigger(beam, qo, s_a, l_a, sign=sgn)
        if t_['kcrit'] is not None and (best is None or t_['W_act'] < best['W_act']):
            best = t_
    tr = best
    if tr is None:
        return res                                    # neither direction switches the arch
    res.update(kcrit=tr['kcrit'], sign=tr['sign'], W_act=tr['W_act'], W_net=tr['W_net'], strain=tr['strain'])
    k0 = beam.kappa0(s_a, l_a, tr['sign'] * (1 + delta) * tr['kcrit'] / beam.L)
    # hold the command until the strip settles, then switch the actuator off
    q_on, q_off = settle_and_release(beam, tr['q'], k0)
    res['w_on'] = beam.ymid(q_on)
    res['switched'] = float(beam.ymid(q_off) < -0.5 * res['w0'])
    if res['switched'] and res['w_on'] < 0:
        dyn = snap_dynamics(beam, qo, k0, w_target=res['w_on'], dt_frac=dt_frac)
        if dyn['t_snap'] is not None:
            res['t_snap'] = dyn['t_snap'] * beam.tstar    # seconds
    return res


def _job(args):
    return evaluate(Beam(N=40), *args)


COLS = ['eps', 's_a', 'l_a', 'F_hold', 'w0', 'kcrit', 'sign', 'W_act', 'W_net', 'strain', 'w_on', 't_snap', 'switched']


def run_grid():
    eps_list = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    sa_list = np.round(np.arange(0.05, 0.5001, 0.025), 3)
    la_list = [0.1, 0.2, 0.3]
    jobs = [(e, s, l) for l in la_list for e in eps_list for s in sa_list if s >= l / 2 - 1e-9]
    t0 = time.time()
    rows = []
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap(_job, jobs, chunksize=2)):
            rows.append([r[c] for c in COLS])
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(jobs)} designs, {time.time() - t0:.0f} s', flush=True)
                np.savetxt('results/grid.csv', np.array(rows), delimiter=',', header=','.join(COLS), comments='')
    np.savetxt('results/grid.csv', np.array(rows), delimiter=',', header=','.join(COLS), comments='')
    print(f'grid done: {len(rows)} designs, {time.time() - t0:.0f} s')


# ---- constrained search ----------------------------------------------------------------
def objective(beam, d, W_ref, log=None):
    """Penalized objective of eq. (5): W_act/W_ref + 100 * (sum of constraint violations)."""
    eps, s_a, l_a = d
    lo, hi = BOUNDS['eps']
    if not (lo <= eps <= hi and BOUNDS['l_a'][0] <= l_a <= BOUNDS['l_a'][1] and l_a / 2 <= s_a <= 0.5):
        return 1e6                                    # outside the design space
    r = evaluate(beam, eps, s_a, l_a)
    if not np.isfinite(r['kcrit']) or not r['switched'] or not np.isfinite(r['t_snap']):
        J = 1e3
    else:
        viol = (max(0.0, (F_REQ - r['F_hold']) / F_REQ) + max(0.0, (r['t_snap'] - T_MAX) / T_MAX)
                + max(0.0, (r['strain'] - STRAIN_LIM) / STRAIN_LIM))
        J = r['W_act'] / W_ref + 100.0 * viol
    if log is not None:
        log.append(dict(r, J=J))
    return J


def nelder_mead(f, x0, step, xtol=2e-3, ftol=1e-2, maxit=80):
    """Minimal Nelder-Mead simplex search (reflection, expansion, contraction, shrink)."""
    n = len(x0)
    simplex = [np.array(x0, float)] + [np.array(x0, float) + step[i] * np.eye(n)[i] for i in range(n)]
    fval = [f(x) for x in simplex]
    for it in range(maxit):
        order = np.argsort(fval)
        simplex = [simplex[i] for i in order]
        fval = [fval[i] for i in order]
        size = max(np.max(np.abs(x - simplex[0]) / np.abs(step)) for x in simplex[1:])
        if size < xtol / np.max(np.abs(step)) or (fval[-1] - fval[0]) < ftol * abs(fval[0]) and size < 0.2:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < fval[0]:
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = f(xe)
            simplex[-1], fval[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fval[-2]:
            simplex[-1], fval[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fval[-1]:
                simplex[-1], fval[-1] = xc, fc
            else:                                      # shrink towards the best point
                simplex = [simplex[0]] + [simplex[0] + 0.5 * (x - simplex[0]) for x in simplex[1:]]
                fval = [fval[0]] + [f(x) for x in simplex[1:]]
    i = int(np.argmin(fval))
    return simplex[i], fval[i], it + 1


def run_search():
    grid = np.genfromtxt('results/grid.csv', delimiter=',', names=True)
    ok = (grid['switched'] > 0) & np.isfinite(grid['t_snap']) & (grid['F_hold'] >= F_REQ) \
        & (grid['t_snap'] <= T_MAX) & (grid['strain'] <= STRAIN_LIM)
    best = grid[ok][np.argmin(grid[ok]['W_act'])]
    print('best feasible grid point:', {c: float(best[c]) for c in COLS})
    beam = Beam(N=40)
    W_ref = float(best['W_act'])
    log = []
    t0 = time.time()
    x, J, nit = nelder_mead(lambda d: objective(beam, d, W_ref, log),
                            [best['eps'], best['s_a'], best['l_a']], step=[-0.003, 0.02, 0.03])
    print(f'Nelder-Mead pass 1: {nit} iterations, x = {x}, J = {J:.4f}')
    # restart from the result with a smaller simplex to confirm the optimum
    x, J, nit2 = nelder_mead(lambda d: objective(beam, d, W_ref, log), x, step=[0.001, -0.007, 0.01])
    print(f'Nelder-Mead pass 2: {nit2} iterations, x = {x}, J = {J:.4f}')
    print(f'{len(log)} evaluations, {time.time() - t0:.0f} s')
    final40 = evaluate(beam, *x)
    final80 = evaluate(Beam(N=80), *x, dt_frac=1 / 8000)
    print('optimum (N=40):', final40)
    print('re-check (N=80, half time step):', final80)
    # reference designs at the same end shortening and patch length
    ref_center = evaluate(beam, x[0], 0.5, x[2])
    ref_035 = evaluate(beam, x[0], 0.35, x[2])
    # sensitivity: patch position errors and different overdrive
    sens = [evaluate(beam, x[0], x[1] + ds, x[2]) for ds in (-0.02, -0.01, 0.01, 0.02)]
    over = [evaluate(beam, *x, delta=dlt) for dlt in (0.02, 0.05, 0.2, 0.4)]
    for ds, r in zip((-0.02, -0.01, 0.01, 0.02), sens):
        print(f'  s_a {ds:+.2f}: kcrit {r["kcrit"]:.3f}, W_act {r["W_act"]*1e3:.4f} mJ, strain {r["strain"]*100:.2f}%, '
              f't_snap {r["t_snap"]*1e3:.2f} ms, switched {r["switched"]}')
    for dlt, r in zip((0.02, 0.05, 0.2, 0.4), over):
        print(f'  delta {dlt}: t_snap {r["t_snap"]*1e3:.2f} ms, switched {r["switched"]}')
    with open('results/search.json', 'w') as fh:
        json.dump(dict(requirements=dict(F_req=F_REQ, t_max=T_MAX, strain_lim=STRAIN_LIM, delta=DELTA),
                       grid_best={c: float(best[c]) for c in COLS}, x=list(map(float, x)), J=float(J),
                       iterations=[nit, nit2], final_N40=final40, final_N80=final80,
                       ref_center=ref_center, ref_035=ref_035,
                       sensitivity_s_a={str(ds): r for ds, r in zip((-0.02, -0.01, 0.01, 0.02), sens)},
                       overdrive={str(dlt): r for dlt, r in zip((0.02, 0.05, 0.2, 0.4), over)},
                       history=log),
                  fh, indent=1, default=float)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'grid':
        run_grid()
    elif len(sys.argv) > 1 and sys.argv[1] == 'search':
        run_search()
    else:
        print(__doc__)
