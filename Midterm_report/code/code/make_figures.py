"""
make_figures.py -- figures of the design study from results/grid.csv and
results/search.json.   Run:  python make_figures.py
MAE 263F (Fall 2026), Yuyang Liu.
"""
import json
import os
import numpy as np
from beam import Beam
from snap import buckle, trigger, snap_dynamics, settle_and_release
from design import F_REQ, T_MAX, STRAIN_LIM, DELTA, holding
import theory
from plotstyle import plt, save, BLUE, ORANGE, AQUA, INK, MUTED, BLUES

g = np.genfromtxt('results/grid.csv', delimiter=',', names=True)
eps_list = np.unique(g['eps'])
beam = Beam(N=40)
FS = beam.EI / beam.L**2


def table(l_a, name):
    """2-D array (eps rows, s_a columns) of one quantity for one patch length."""
    m = g[np.isclose(g['l_a'], l_a)]
    sa = np.unique(m['s_a'])
    A = np.full((len(eps_list), len(sa)), np.nan)
    for r in m:
        A[np.searchsorted(eps_list, r['eps']), np.searchsorted(sa, r['s_a'])] = r[name]
    return sa, A


# ---------------------------------------------------------------- maps for l_a = 0.2
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
for ax, (name, scale, label) in zip(axes, (('kcrit', 1, r'trigger curvature $\bar\kappa_{\rm trig}L$'),
                                           ('W_act', 1e3, r'actuator work $W_{\rm act}$ (mJ)'),
                                           ('t_snap', 1e3, r'closing time $t_{\rm snap}$ (ms)'))):
    sa, A = table(0.2, name)
    _, SW = table(0.2, 'switched')
    A = np.where(SW > 0, A * scale, np.nan)        # blank: the arch does not stay closed
    norm = None
    if name == 'W_act':
        import matplotlib.colors as mc
        norm = mc.LogNorm(vmin=np.nanmin(A), vmax=np.nanmax(A))
    x_edges = np.concatenate([sa - 0.0125, [sa[-1] + 0.0125]])
    y_edges = np.arange(len(eps_list) + 1)
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#dddddd')
    pc = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(A), cmap=cmap, norm=norm,
                       edgecolors='white', linewidth=0.3)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not np.isfinite(A[i, j]):
                ax.text(sa[j], i + 0.5, r'$\times$', ha='center', va='center', color=INK, fontsize=6)
    ax.set_yticks(np.arange(len(eps_list)) + 0.5)
    ax.set_yticklabels([f'{e*100:g}' for e in eps_list])
    ax.set_xlabel('patch position $s_a$')
    ax.set_ylabel(r'end shortening $\varepsilon$ (%)')
    ax.set_title(label, fontsize=7.5)
    cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.06)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.4)
fig.tight_layout(w_pad=0.8)
save(fig, 'design_maps')

# ---------------------------------------------------------------- effect of patch length
fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.0))
e0 = 0.015
for l_a, c, mk, ls in ((0.1, BLUE, 'o', '-'), (0.2, ORANGE, 's', '--'), (0.3, AQUA, '^', ':')):
    m = g[np.isclose(g['l_a'], l_a) & np.isclose(g['eps'], e0)]
    sw = m['switched'] > 0                         # gap where the arch does not stay closed
    ax[0].semilogy(m['s_a'], np.where(sw, m['W_act'] * 1e3, np.nan), color=c, marker=mk, ls=ls,
                   label=rf'$\ell_a={l_a}$')
    ax[1].plot(m['s_a'], np.where(sw, m['strain'] * 100, np.nan), color=c, marker=mk, ls=ls,
               label=rf'$\ell_a={l_a}$')
ax[0].set_xlabel('patch position $s_a$'); ax[0].set_ylabel(r'$W_{\rm act}$ (mJ)')
ax[0].set_ylim(0.012, 20)
ax[0].legend(loc='lower center', handlelength=2.2, labelspacing=0.15, ncol=1)
ax[0].set_title('(a)', loc='left', fontsize=8)
ax[1].axhline(STRAIN_LIM * 100, color=INK, lw=0.7, ls='--')
ax[1].annotate('limit', (0.5, STRAIN_LIM * 100), ha='right', va='bottom', fontsize=7, color=INK)
ax[1].set_xlabel('patch position $s_a$'); ax[1].set_ylabel('largest bending strain (%)')
ax[1].set_title('(b)', loc='left', fontsize=8)
ax[1].set_ylim(0.05, 1.8)
ax[1].legend(loc='lower center', ncol=3, handlelength=1.3, columnspacing=0.5, handletextpad=0.25,
             fontsize=5.8, bbox_to_anchor=(0.56, -0.02))
fig.tight_layout(w_pad=0.6)
save(fig, 'design_patch')

# ---------------------------------------------------------------- holding force and trade-off
feas = (g['switched'] > 0) & np.isfinite(g['t_snap']) & (g['t_snap'] <= T_MAX) & (g['strain'] <= STRAIN_LIM)
Fh = np.array([g[np.isclose(g['eps'], e)]['F_hold'][0] for e in eps_list])
Wmin = np.array([np.min(g[feas & np.isclose(g['eps'], e)]['W_act']) if np.any(feas & np.isclose(g['eps'], e)) else np.nan
                 for e in eps_list])
S = beam.h**2 / (12 * beam.L**2)
fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.0))
ee = np.linspace(0.004, 0.05, 80)
ax[0].plot(ee * 100, [theory.point_force_center(e, S)[0] * FS for e in ee], color=INK, lw=0.9, ls='--', label='theory')
ax[0].plot(eps_list * 100, Fh, 'o', color=BLUE, label='simulation')
ax[0].axhline(F_REQ, color=MUTED, lw=0.6)
ax[0].annotate(r'$F_{\rm req}$', (0.3, F_REQ), va='bottom', fontsize=7, color=INK)
ax[0].set_xlabel(r'end shortening $\varepsilon$ (%)'); ax[0].set_ylabel(r'holding force $F_{\rm hold}$ (N)')
ax[0].legend(loc='lower right'); ax[0].set_title('(a)', loc='left', fontsize=8)
ax[1].plot(Fh, Wmin * 1e3, 'o-', color=BLUE, label='best grid design')
if os.path.exists('results/search.json'):
    sr = json.load(open('results/search.json'))
    f40 = sr['final_N40']
    ax[1].plot(f40['F_hold'], f40['W_act'] * 1e3, '*', color=ORANGE, ms=8, label='search optimum')
ax[1].axvline(F_REQ, color=MUTED, lw=0.6)
ax[1].set_xlabel(r'holding force $F_{\rm hold}$ (N)'); ax[1].set_ylabel(r'least $W_{\rm act}$ (mJ)')
ax[1].set_ylim(-0.1, 3.4); ax[1].legend(loc='upper left'); ax[1].set_title('(b)', loc='left', fontsize=8)
fig.tight_layout(w_pad=0.6)
save(fig, 'design_tradeoff')

# ---------------------------------------------------------------- the selected design
if os.path.exists('results/search.json'):
    eps, s_a, l_a = sr['x']
    qo = buckle(beam, eps)[0]
    tr = trigger(beam, qo, s_a, l_a)
    k0 = beam.kappa0(s_a, l_a, tr['sign'] * (1 + DELTA) * tr['kcrit'] / beam.L)
    q_on, q_off = settle_and_release(beam, tr['q'], k0)
    w_on, w0 = beam.ymid(q_on), beam.ymid(qo)
    dyn = snap_dynamics(beam, qo, k0, w_target=w_on, record=True, store_every=20)
    ts_ms = dyn['t_snap'] * beam.tstar * 1e3
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.4), gridspec_kw={'height_ratios': [1, 0.95]})
    times = np.linspace(0, ts_ms, 6)
    tt = np.array([t for t, _ in dyn['shapes']]) * beam.tstar * 1e3
    ax[0].plot(qo[0::2] / beam.L, qo[1::2] * 1e3, color=BLUES[0], lw=1.4, label='0 ms')
    for c, tq in zip(BLUES[1:], times[1:]):
        i_ = int(np.argmin(np.abs(tt - tq)))
        q = dyn['shapes'][i_][1]
        ax[0].plot(q[0::2] / beam.L, q[1::2] * 1e3, color=c, lw=1.4, label=f'{tt[i_]:.1f} ms')
    ax[0].plot(q_off[0::2] / beam.L, q_off[1::2] * 1e3, color=INK, lw=1.0, ls='--', label='closed, off')
    ax[0].legend(loc='upper center', ncol=4, fontsize=6.2, handlelength=1.5, columnspacing=0.8,
                 labelspacing=0.2, bbox_to_anchor=(0.5, 1.02))
    wts = beam.patch_weights(s_a, l_a) > 0
    ax[0].plot(qo[0::2][1:-1][wts] / beam.L, qo[1::2][1:-1][wts] * 1e3 + 0.9, color=ORANGE, lw=2.5)
    ax[0].annotate('patch', (s_a, np.max(qo[1::2][1:-1][wts]) * 1e3 + 1.4), ha='center', va='bottom', color=INK, fontsize=7)
    ax[0].set_xlabel('$x/L$', labelpad=1); ax[0].set_ylabel('$y$ (mm)')
    ax[0].set_ylim(-1.3 * w0 * 1e3, 2.35 * w0 * 1e3)
    ax[0].set_title('(a)', loc='left', fontsize=8)
    # long runs without numerical damping (trapezoidal rule) for two damping ratios
    settle = {}
    for zeta, c, ls in ((0.02, BLUE, '-'), (0.10, ORANGE, '--')):
        d2 = snap_dynamics(beam, qo, k0, w_target=w_on, zeta=zeta, hold=1.3, record=True, scheme='trapezoid')
        t_ms = d2['hist'][0] * beam.tstar * 1e3
        y = d2['hist'][1]
        out_band = np.abs(y - w_on) > 0.1 * (w0 - w_on)
        t_set = t_ms[np.where(out_band)[0][-1]] if out_band.any() else 0.0
        settle[zeta] = (d2['t_snap'] * beam.tstar * 1e3, t_set, t_ms[-1])
        ax[1].plot(t_ms, y * 1e3, color=c, ls=ls, lw=1.1, label=rf'$\zeta={zeta:g}$')
    ax[1].axhline(w_on * 1e3, color=MUTED, lw=0.7, ls=':', label='settled, command on')
    ax[1].set_xlabel('time (ms)'); ax[1].set_ylabel('midpoint height (mm)')
    ax[1].legend(loc='upper right', ncol=3, handlelength=1.8, columnspacing=0.7, fontsize=6.3)
    ax[1].set_ylim(-1.35 * w0 * 1e3, 1.45 * w0 * 1e3)
    ax[1].set_title('(b)', loc='left', fontsize=8)
    fig.tight_layout(h_pad=0.7)
    save(fig, 'design_optimum')
    for z, (a_, b_, c_) in settle.items():
        print(f'zeta={z}: first arrival {a_:.2f} ms, settled within 10% of the stroke after {b_:.1f} ms (window {c_:.0f} ms)')
    json.dump({str(z): v for z, v in settle.items()}, open('results/optimum_dynamics.json', 'w'), indent=1)
print('figures written to figures/')
