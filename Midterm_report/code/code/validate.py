"""
validate.py -- validation of the solver (items 1-5 of the proposal).
Run:  python validate.py          (about 5 minutes)
Writes results/validation.txt, results/validation.json and figures/val_*.pdf/png.
MAE 263F (Fall 2026), Yuyang Liu.
"""
import json
import os
import time
import numpy as np
from beam import Beam, energy, grad_hess, test_against_course
from snap import (buckle, critical_point_force, indentation_curve, trigger, snap_dynamics,
                  natural_frequencies, mirror)
import theory
from plotstyle import plt, save, BLUE, ORANGE, AQUA, INK, MUTED

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
out = {}
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(s)


t_start = time.time()
beam = Beam(N=40)
S = beam.h**2 / (12 * beam.L**2)            # stretchability
FS = beam.EI / beam.L**2                    # force scale

# ------------------------------------------------------------------ 1. derivatives
log('1. DERIVATIVES')
eg, eH = test_against_course()
log(f'   vectorized vs course routines: gradient {eg:.1e}, Hessian {eH:.1e} (relative)')
rng = np.random.default_rng(0)
q = beam.straight() + 0.1 * beam.dl * rng.standard_normal(beam.ndof)
k0 = 0.2 * rng.standard_normal(beam.nv - 2)
g, H = grad_hess(beam, q, k0)
d = 1e-7 * beam.dl
gfd = np.zeros(beam.ndof)
Hfd = np.zeros((beam.ndof, beam.ndof))
for i in range(beam.ndof):
    qp, qm = q.copy(), q.copy()
    qp[i] += d
    qm[i] -= d
    gfd[i] = (energy(beam, qp, k0) - energy(beam, qm, k0)) / (2 * d)
    Hfd[:, i] = (grad_hess(beam, qp, k0, False)[0] - grad_hess(beam, qm, k0, False)[0]) / (2 * d)
e1 = np.max(np.abs(g - gfd)) / np.max(np.abs(g))
e2 = np.max(np.abs(H - Hfd)) / np.max(np.abs(H))
log(f'   centered finite differences (random state, with natural curvature): gradient {e1:.1e}, Hessian {e2:.1e}')
out['derivatives'] = dict(course_grad=eg, course_hess=eH, fd_grad=e1, fd_hess=e2)

# ------------------------------------------------------------------ 2. buckling
log('\n2. BUCKLING (N = 80)')
b80 = Beam(N=80)
eps_list = [0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
rows = []
for eps in eps_list:
    qo, P = buckle(b80, eps)
    eps_b = eps - P / b80.EA                 # part of the shortening taken by axial strain
    w_el, P_el = theory.elastica_clamped(eps_b)
    w_sm, P_sm = theory.small_eps(eps)
    rows.append((eps, b80.ymid(qo) / b80.L, w_el, w_sm, P / (b80.EI / b80.L**2), P_el, P_sm))
    log(f'   eps={eps:.3f}: w0/L sim {rows[-1][1]:.5f}, elastica {w_el:.5f} ({(rows[-1][1]/w_el-1)*100:+.2f}%), '
        f'small-eps {w_sm:.5f} ({(rows[-1][1]/w_sm-1)*100:+.2f}%);  P L^2/EI sim {rows[-1][4]:.3f}, '
        f'elastica {P_el:.3f} ({(rows[-1][4]/P_el-1)*100:+.2f}%)')
rows = np.array(rows)
out['buckling'] = rows.tolist()
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.75))
ee = np.linspace(0.001, 0.05, 100)
el = np.array([theory.elastica_clamped(e) for e in ee])
ax[0].plot(ee * 100, el[:, 0], color=INK, lw=1, label='elastica')
ax[0].plot(ee * 100, 2 / np.pi * np.sqrt(ee), color=MUTED, lw=1, ls='--', label=r'$\frac{2}{\pi}\sqrt{\varepsilon}$')
ax[0].plot(rows[:, 0] * 100, rows[:, 1], 'o', color=BLUE, label='simulation')
ax[0].set_xlabel(r'end shortening $\varepsilon$ (%)'); ax[0].set_ylabel('$w_0/L$')
ax[0].legend(loc='lower right', handlelength=1.6)
ax[1].plot(ee * 100, el[:, 1], color=INK, lw=1)
ax[1].axhline(4 * np.pi**2, color=MUTED, lw=1, ls='--')
ax[1].plot(rows[:, 0] * 100, rows[:, 4], 'o', color=BLUE)
ax[1].annotate(r'$4\pi^2$', (3.6, 4 * np.pi**2), va='bottom', color=INK)
ax[1].set_xlabel(r'end shortening $\varepsilon$ (%)'); ax[1].set_ylabel('$PL^2/EI$')
ax[1].set_ylim(39.2, 40.7)
fig.tight_layout(w_pad=0.6)
save(fig, 'val_buckling')

# ------------------------------------------------------------------ 3. static snap-through
log('\n3. STATIC SNAP-THROUGH UNDER A POINT FORCE (N = 80)')
eps = 0.01
qo, P = buckle(b80, eps)
w, F, E, asym, lam = indentation_curve(b80, qo)
F = -F / (b80.EI / b80.L**2)                 # resisting force, dimensionless
f_th, slope_th = theory.point_force_center(eps, S)
ia = np.where(asym > 0)[0]
sel = ia[(np.abs(w[ia]) < 0.5 * w[0])]
slope = np.polyfit(w[sel] / b80.L, F[sel], 1)[0]
barrier = (E.max() - E[0])
barrier_th = (theory.TAU2**2 - theory.TAU1**2) * b80.EI / b80.L**2 * (eps - theory.TAU2**2 * S) * b80.L
log(f'   midpoint, displacement control, eps=1%: peak force {F.max():.3f} (theory {f_th:.3f}, {(F.max()/f_th-1)*100:+.2f}%)')
log(f'   slope of asymmetric branch {slope:.2f} (Pandey et al.: {slope_th:.2f}, {(slope/slope_th-1)*100:+.2f}%)')
log(f'   energy barrier {barrier*1e3:.4f} mJ (two-mode estimate (P2-P1) dL: {barrier_th*1e3:.4f} mJ, {(barrier/barrier_th-1)*100:+.2f}%)')
out['indentation'] = dict(peak=F.max(), peak_theory=f_th, slope=slope, slope_theory=slope_th,
                          barrier_J=barrier, barrier_theory_J=barrier_th)
deltas = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
fc = {}
for e_ in (0.01, 0.03):
    qo_, _ = buckle(b80, e_)
    fc[e_] = np.array([critical_point_force(b80, qo_, dlt)[0] / (b80.EI / b80.L**2) for dlt in deltas])
    th = np.array([theory.point_force_offset(e_, dlt) for dlt in deltas])
    log(f'   force control, eps={e_}: delta, F_cr sim, Yan et al., diff%')
    for dlt, a, bb in zip(deltas, fc[e_], th):
        log(f'      {dlt:.2f}  {a:8.3f}  {bb:8.3f}  {(a/bb-1)*100:+.2f}')
out['offset_force'] = {str(k): v.tolist() for k, v in fc.items()}
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.9))
ax[0].plot(w / w[0], F, color=BLUE, label='simulation')
ww = np.linspace(-1, 1, 50)
ax[0].plot(ww, slope_th * ww * w[0] / b80.L, color=INK, lw=0.9, ls='--', label='theory')
ax[0].plot([0.75, 1.0], [f_th, f_th], color=INK, lw=0.9, ls='--')
ax[0].set_ylim(-15, 21); ax[0].axhline(0, color=MUTED, lw=0.4)
ax[0].set_xlabel('midpoint height $w/w_0$'); ax[0].set_ylabel('$FL^2/EI$')
ax[0].legend(loc='upper right'); ax[0].invert_xaxis()
ax[0].set_title('(a)', loc='left', fontsize=8)
dd = np.linspace(0.18, 0.5, 60)
for e_, c, mk in ((0.01, BLUE, 'o'), (0.03, ORANGE, 's')):
    ax[1].plot(dd, [theory.point_force_offset(e_, x) for x in dd], color=c, lw=1,
               label=rf'Yan, $\varepsilon={e_*100:.0f}\%$')
    ax[1].plot(deltas, fc[e_], mk, color=c, label=rf'sim., $\varepsilon={e_*100:.0f}\%$')
ax[1].set_xlabel(r'force position $\delta$'); ax[1].set_ylabel(r'$F_{\rm cr}L^2/EI$')
ax[1].set_ylim(0, 46); ax[1].legend(loc='upper center', ncol=2, columnspacing=0.6, handletextpad=0.3,
                                    handlelength=1.4, fontsize=6.0, bbox_to_anchor=(0.56, 1.02))
ax[1].set_title('(b)', loc='left', fontsize=8)
fig.tight_layout(w_pad=0.6)
save(fig, 'val_pointforce')

# ------------------------------------------------------------------ 4. dynamics
log('\n4. DYNAMICS')
om_th = theory.antisymmetric_frequency()
for N in (40, 80, 160):
    bb = Beam(N=N)
    qo_, _ = buckle(bb, 0.01)
    om = natural_frequencies(bb, qo_, nmodes=2) * bb.tstar
    log(f'   N={N}: lowest frequency of the arch {om[0]:.3f}/t* (theory, antisymmetric mode: {om_th:.3f}, {(om[0]/om_th-1)*100:+.2f}%)')
    out[f'freq_N{N}'] = om[0]
# free vibration with implicit Euler: period and numerical damping versus time step
qo, _ = buckle(beam, 0.01)
gH = grad_hess(beam, qo)[1]
f_ = beam.free
sm = 1 / np.sqrt(beam.m[f_])
lam_, vec_ = np.linalg.eigh(gH[np.ix_(f_, f_)] * sm[:, None] * sm[None, :])
mode = np.zeros(beam.ndof)
mode[f_] = vec_[:, 0] * sm
mode *= 1e-4 * beam.L / np.max(np.abs(mode))
w1 = np.sqrt(lam_[0])
quarter = 2 * (beam.N // 4 + 1) + 1
from beam import advance
for nper in (50, 200, 800):
    dt = 2 * np.pi / w1 / nper
    q_, u_ = qo + mode, np.zeros(beam.ndof)
    y = []
    for n in range(4 * nper):
        q_, u_, _ = advance(beam, q_, u_, dt)
        y.append(q_[quarter] - qo[quarter])
    y = np.array(y)
    zc = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
    tz = (zc + y[zc] / (y[zc] - y[zc + 1]) + 1) * dt
    period = 2 * np.mean(np.diff(tz))
    decay = np.max(np.abs(y[3 * nper:])) / np.max(np.abs(y[:nper]))
    log(f'   implicit Euler, {nper} steps/period: period error {(period*w1/(2*np.pi)-1)*100:+.2f}%, '
        f'amplitude after 3 periods {decay*100:.1f}% (numerical damping only)')
# critical slowing down (offset patch = limit point)
s_a, l_a = 0.35, 0.2
tr = trigger(beam, qo, s_a, l_a, rtol=1e-6)
dls = np.array([0.3, 0.1, 0.03, 0.01, 0.003, 0.001])
ts_ = []
for dlt in dls:
    k0c = beam.kappa0(s_a, l_a, tr['sign'] * (1 + dlt) * tr['kcrit'] / beam.L)
    r = snap_dynamics(beam, qo, k0c, q_start=tr['q'], zeta=0.0, dt_frac=1 / 8000)
    ts_.append(r['t_snap'])
ts_ = np.array(ts_)
expo = np.polyfit(np.log(dls[2:]), np.log(ts_[2:]), 1)[0]
log(f'   snap time from the equilibrium at the threshold, s_a=0.35 (no damping): ' +
    ', '.join(f'{a:g}:{b_:.3f}' for a, b_ in zip(dls, ts_)))
log(f'   fitted exponent for delta <= 0.03: {expo:.3f} (Gomez et al.: -1/4)')
out['slowing'] = dict(delta=dls.tolist(), t_snap=ts_.tolist(), exponent=expo)
# energy balance during a snap
k0c = beam.kappa0(s_a, l_a, tr['sign'] * 1.1 * tr['kcrit'] / beam.L)
bal = {}
for dtf in (1 / 1000, 1 / 4000, 1 / 16000):
    r = snap_dynamics(beam, qo, k0c, dt_frac=dtf, hold=0.3, record=True)
    t_, ym, ke, ee_, di = r['hist']
    E0 = energy(beam, qo, k0c)
    loss = E0 - (ke[-1] + ee_[-1] + di[-1])          # energy removed by the integrator
    bal[dtf] = (r['t_snap'], loss / (E0 - ee_.min()))
    log(f'   step to 1.1 kcrit, dt=t*/{1/dtf:.0f}: t_snap={r["t_snap"]:.4f} t*, '
        f'numerical energy loss = {bal[dtf][1]*100:.1f}% of the released elastic energy')
    if dtf == 1 / 4000:
        hist = r['hist']
out['energy_balance'] = {str(k): v for k, v in bal.items()}
for dtf in (1 / 4000, 1 / 16000):
    r = snap_dynamics(beam, qo, k0c, dt_frac=dtf, hold=0.3, record=True, scheme='trapezoid')
    t_, ym, ke, ee_, di = r['hist']
    loss = E0 - (ke[-1] + ee_[-1] + di[-1])
    log(f'   same with the trapezoidal rule, dt=t*/{1/dtf:.0f}: t_snap={r["t_snap"]:.4f} t*, '
        f'energy error = {loss/(E0-ee_.min())*100:+.2f}% of the released elastic energy')
    out[f'trapezoid_{1/dtf:.0f}'] = (r['t_snap'], loss / (E0 - ee_.min()))
    if dtf == 1 / 4000:
        hist_tr = r['hist']
fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.0))
ax[0].loglog(dls, ts_, 'o', color=BLUE, label='simulation')
ax[0].loglog(dls, ts_[-1] * (dls / dls[-1]) ** -0.25, color=INK, lw=0.9, ls='--', label=r'slope $-1/4$')
ax[0].set_yticks([0.2, 0.4, 0.8]); ax[0].set_yticklabels(['0.2', '0.4', '0.8']); ax[0].minorticks_off()
ax[0].set_ylim(0.13, 1.9)
ax[0].set_xlabel(r'overdrive $\delta$'); ax[0].set_ylabel(r'$t_{\rm snap}/t^*$'); ax[0].legend(loc='upper right')
ax[0].set_title('(a)', loc='left', fontsize=8)
t_, ym, ke, ee_, di = hist
E0 = energy(beam, qo, k0c)
ax[1].plot(t_, (ee_ - E0) * 1e3, color=BLUE, label='elastic')
ax[1].plot(t_, ke * 1e3, color=ORANGE, ls='--', label='kinetic')
ax[1].plot(t_, (ee_ - E0 + ke + di) * 1e3, color=INK, lw=0.9, label='sum, impl. Euler')
t2, ym2, ke2, ee2, di2 = hist_tr
ax[1].plot(t2, (ee2 - E0 + ke2 + di2) * 1e3, color=MUTED, lw=0.9, ls=':', label='sum, trapezoidal')
ax[1].set_ylim(-1.9, 3.4)
ax[1].set_xlabel('$t/t^*$'); ax[1].set_ylabel('energy change (mJ)')
ax[1].legend(loc='upper left', ncol=1, handlelength=1.8, labelspacing=0.25)
ax[1].set_title('(b)', loc='left', fontsize=8)
fig.tight_layout(w_pad=0.6)
save(fig, 'val_dynamics')

# ------------------------------------------------------------------ 5. convergence
log('\n5. MESH CONVERGENCE (eps = 1%, patch s_a = 0.35, l_a = 0.2)')
log('   N  half-clamp   w0/L      P L^2/EI   F_hold L^2/EI   kcrit L    W_act [mJ]')
conv = []
for half in (False, True):
    for N in (20, 40, 80, 160):
        bb = Beam(N=N, half_clamp=half)
        qo_, P_ = buckle(bb, 0.01)
        Fh = critical_point_force(bb, qo_, 0.5)[0] / (bb.EI / bb.L**2)
        tr_ = trigger(bb, qo_, 0.35, 0.2)
        conv.append((N, int(half), bb.ymid(qo_) / bb.L, P_ / (bb.EI / bb.L**2), Fh, tr_['kcrit'], tr_['W_act'] * 1e3))
        log('   %3d  %d         %.5f   %.3f     %.3f          %.4f    %.4f' % conv[-1])
out['convergence'] = conv
log(f'\ntotal time {time.time() - t_start:.0f} s')
with open('results/validation.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
with open('results/validation.json', 'w') as f:
    json.dump(out, f, indent=1, default=float)
