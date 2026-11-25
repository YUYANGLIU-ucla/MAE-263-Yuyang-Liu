"""
convergence.py -- time-step and mesh convergence of the selected design
(results/search.json).   Run after `python design.py search`:
    python convergence.py        (about 3 minutes)
Writes results/convergence.txt, results/convergence.json, figures/val_convergence.pdf/png.
MAE 263F (Fall 2026), Yuyang Liu.
"""
import json
import time
import numpy as np
from beam import Beam
from snap import buckle, trigger, snap_dynamics, settle_and_release
from design import evaluate, DELTA
from plotstyle import plt, save, BLUE, ORANGE, AQUA, INK, MUTED

t0 = time.time()
eps, s_a, l_a = json.load(open('results/search.json'))['x']
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(s)


log(f'selected design: eps = {eps:.5f}, s_a = {s_a:.4f}, l_a = {l_a:.4f}')

# ---------------------------------------------------------------- time step (N = 40)
beam = Beam(N=40)
qo = buckle(beam, eps)[0]
tr = trigger(beam, qo, s_a, l_a, sign=1.0)
k0 = beam.kappa0(s_a, l_a, tr['sign'] * (1 + DELTA) * tr['kcrit'] / beam.L)
q_on, q_off = settle_and_release(beam, tr['q'], k0)
w_on = beam.ymid(q_on)
fracs = np.array([1 / 250, 1 / 500, 1 / 1000, 1 / 2000, 1 / 4000, 1 / 8000, 1 / 16000])
ts = {}
for scheme in ('euler', 'trapezoid'):
    ts[scheme] = []
    for f in fracs:
        try:
            r = snap_dynamics(beam, qo, k0, w_target=w_on, dt_frac=f, scheme=scheme)
            ts[scheme].append(r['t_snap'] * beam.tstar * 1e3)
        except RuntimeError:            # Newton failed (no step halving in the trapezoidal step)
            ts[scheme].append(np.nan)
    ts[scheme] = np.array(ts[scheme])
log('\nTIME STEP (N = 40): t_snap [ms]')
log('   dt/t*      implicit Euler   trapezoidal')
for f, a, b in zip(fracs, ts['euler'], ts['trapezoid']):
    fmt = lambda v: 'Newton failed' if np.isnan(v) else f'{v:.4f}'
    log(f'   1/{1/f:<7.0f}  {fmt(a):<16} {fmt(b)}')
ref = ts['trapezoid'][-1]
for scheme in ('euler', 'trapezoid'):
    err = np.abs(ts[scheme][:-1] - ref) / ref
    log(f'   {scheme}: difference to trapezoidal at t*/16000: '
        + ', '.join('-' if np.isnan(e) else f'{e*100:.2f}%' for e in err))
    if scheme == 'euler':
        p = np.polyfit(np.log(fracs[1:-1]), np.log(np.abs(ts[scheme][1:-1] - ref)), 1)[0]
        log(f'   implicit Euler: observed order of convergence {p:.2f}')

# ---------------------------------------------------------------- mesh (dt = t*/4000)
log('\nMESH (dt = t*/4000)')
log('   N    kcrit L   W_act [mJ]  F_hold [N]  strain [%]  t_snap [ms]')
Ns = [20, 40, 80, 160]
rows = []
for N in Ns:
    r = evaluate(Beam(N=N), eps, s_a, l_a)
    rows.append((N, r['kcrit'], r['W_act'] * 1e3, r['F_hold'], r['strain'] * 100, r['t_snap'] * 1e3))
    log('   %3d  %.4f    %.4f      %.4f      %.3f       %.3f' % rows[-1])
rows = np.array(rows)
log(f'\ntotal time {time.time() - t0:.0f} s')
with open('results/convergence.txt', 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
json.dump(dict(dt_frac=fracs.tolist(), t_snap_euler=ts['euler'].tolist(),
               t_snap_trapezoid=ts['trapezoid'].tolist(), mesh=rows.tolist()),
          open('results/convergence.json', 'w'), indent=1)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.95))
ax[0].semilogx(fracs, ts['euler'], 'o-', color=BLUE, label='implicit Euler')
ax[0].semilogx(fracs, ts['trapezoid'], 's--', color=ORANGE, label='trapezoidal')
ax[0].invert_xaxis()
ax[0].set_xlabel(r'time step $\Delta t/t^*$'); ax[0].set_ylabel(r'$t_{\rm snap}$ (ms)')
ax[0].legend(loc='upper right')
ax[0].set_title('(a)', loc='left', fontsize=8)
names = (('kcrit', 1, r'$\bar\kappa_{\rm trig}$', BLUE, 'o', '-'),
         ('W', 2, r'$W_{\rm act}$', ORANGE, 's', '--'),
         ('F', 3, r'$F_{\rm hold}$', AQUA, '^', ':'),
         ('t', 5, r'$t_{\rm snap}$', INK, 'D', '-.'))
for _, j, lab, c, mk, ls in names:
    ax[1].semilogx(rows[:, 0], rows[:, j] / rows[-1, j], color=c, marker=mk, ls=ls, label=lab, base=2)
ax[1].axhline(1.0, color=MUTED, lw=0.5)
ax[1].set_xticks(Ns); ax[1].set_xticklabels([str(n) for n in Ns]); ax[1].minorticks_off()
ax[1].set_ylim(0.955, 1.075)
ax[1].set_xlabel('number of edges $N$'); ax[1].set_ylabel('value / value at $N=160$')
ax[1].legend(loc='upper right', ncol=2, fontsize=6.3, labelspacing=0.15, handlelength=1.8, columnspacing=0.6)
ax[1].set_title('(b)', loc='left', fontsize=8)
fig.tight_layout(w_pad=0.6)
save(fig, 'val_convergence')
