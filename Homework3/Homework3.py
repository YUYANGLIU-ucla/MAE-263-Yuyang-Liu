"""MAE-263F HW3: 3D Discrete Elastic Rods with bending, twist and stretch.

python Homework3.py          # compute Q1--Q11 and save results
python Homework3.py --test   # geometry, equilibrium and time-step checks

DER force derivatives are the bundled course functions by M. Khalid Jawed.
The setup, bounded Newton solver, steady-state checks and sweeps are adapted
for this homework. No external course checkout is needed.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded
from der_functions import (computeTangent, assignOrthonormalAdaptedFrame,
                           computeSpaceParallel, computeTimeParallel,
                           computeMaterialDirectors, getKappa, getRefTwist,
                           getFs, getFb, getFt)


def eb_deflection(L=.2, E=10e6, g=.05, rho=1000., r0=.001):
    """Uniform self-weight cantilever: rho*g*L^4/(2*E*r0^2)."""
    return rho*g*L**4/(2*E*r0**2)


def setup(N=52, L=.2, E=10e6, g=9.81, natR=0., zero_curvature=False):
    if N < 4 or L <= 0 or E <= 0 or g < 0 or natR < 0:
        raise ValueError('Invalid geometry, stiffness or gravity')
    ne, ndof = N-1, 4*N-1
    nodes = np.zeros((N, 3))
    if natR == 0:
        dl = L/(N-2)  # ONLY the last N-2 edges are free.
        nodes[:, 0] = (np.arange(N)-1)*dl
        total_length = (N-1)*dl
    else:
        angles = np.arange(N)*L/(natR*ne)
        nodes[:, 0] = natR*np.cos(angles)
        nodes[:, 1] = natR*np.sin(angles)
        total_length = L  # prescribed arc length, as in the course example
    q = np.zeros(ndof)
    for c in range(3):
        q[c::4] = nodes[:, c]
    r0, rho, nu = .001, 1000., .5
    A, I, J = np.pi*r0**2, np.pi*r0**4/4, np.pi*r0**4/2
    EA, EI, GJ = E*A, E*I, E/(2*(1+nu))*J
    dm = rho*A*total_length/ne
    nodal_mass = np.full(N, dm)
    nodal_mass[[0, -1]] *= .5
    mass = np.zeros(ndof)
    for c in range(3):
        mass[c::4] = nodal_mass
    mass[3::4] = .5*dm*r0**2
    gravity = np.zeros(ndof)
    gravity[2::4] = -nodal_mass*g
    ref_len = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    voronoi = np.r_[ref_len[0]/2, (ref_len[:-1]+ref_len[1:])/2, ref_len[-1]/2]
    a1_first, _ = assignOrthonormalAdaptedFrame(computeTangent(q)[0])
    a1, a2 = computeSpaceParallel(a1_first, q)
    m1, m2 = computeMaterialDirectors(a1, a2, q[3::4])
    kappa_bar = getKappa(q, m1, m2)
    if zero_curvature:
        kappa_bar = np.zeros_like(kappa_bar)
    return dict(q=q, a1=a1, a2=a2, mass=mass, gravity=gravity,
                EA=EA, EI=EI, GJ=GJ, ref_len=ref_len, voronoi=voronoi,
                kappa_bar=kappa_bar, twist_bar=np.zeros(N),
                # Tighter than the example's 1e-3: small-gravity mesh errors
                # must not be obscured by the Newton residual tolerance.
                tol=EI/L**2*1e-6, N=N, total_length=total_length)


def forces(q, old_q, old_a1, old_twist, p):
    """Evaluate forces using frames transported from the accepted old state.

    getRefTwist mutates its argument: copy it so Newton trial states never
    overwrite the accepted branch of the reference twist.
    """
    a1, a2 = computeTimeParallel(old_a1, old_q, q)
    ref_twist = getRefTwist(a1, computeTangent(q), old_twist.copy())
    m1, m2 = computeMaterialDirectors(a1, a2, q[3::4])
    Fb, Jb = getFb(q, m1, m2, p['kappa_bar'], p['EI'], p['voronoi'])
    Ft, Jt = getFt(q, ref_twist, p['twist_bar'], p['GJ'], p['voronoi'])
    Fs, Js = getFs(q, p['EA'], p['ref_len'])
    return Fb+Ft+Fs, Jb+Jt+Js, a1, a2, ref_twist


def step(old_q, old_u, old_a1, old_twist, dt, p):
    """Implicit Euler / Newton on free DOFs; first seven DOFs stay fixed."""
    q = old_q.copy()
    diagonal = p['mass']/dt**2
    ndof = q.size
    for iteration in range(60):
        F, JF, a1, a2, twist = forces(q, old_q, old_a1, old_twist, p)
        residual = diagonal*(q-old_q) - p['mass']*old_u/dt - F - p['gravity']
        if np.linalg.norm(residual[7:]) < p['tol']:
            return q, (q-old_q)/dt, a1, a2, twist, iteration
        jacobian = -JF
        jacobian[np.diag_indices(ndof)] += diagonal
        # Each energy touches at most 3 adjacent nodes: half-bandwidth = 10.
        # This solves the SAME Newton system as np.linalg.solve, much faster.
        Jfree = jacobian[7:, 7:]
        n = Jfree.shape[0]
        band = np.zeros((21, n))
        for offset in range(-10, 11):
            columns = np.arange(max(0, offset), min(n, n+offset))
            band[10-offset, columns] = Jfree[columns-offset, columns]
        correction = solve_banded((10, 10), band, residual[7:])
        q[7:] -= correction
        if not np.isfinite(q).all():
            break
    raise RuntimeError('Newton failed to converge within 60 iterations')


def run(N=52, L=.2, E=10e6, g=9.81, natR=0., dt=.1,
        totalTime=10., zero_curvature=False, progress=False):
    """Return final q, initial q, histories and measured steady-state checks.

    A failed step is retried as two half steps (at most four subdivisions).
    No physical damping is added. All histories include t=0.
    """
    steps = round(totalTime/dt) if dt > 0 else 0
    if steps < 1 or not np.isclose(steps*dt, totalTime):
        raise ValueError('totalTime must be a positive integer multiple of dt')
    p = setup(N, L, E, g, natR, zero_curvature)
    q = p['q'].copy()
    initial = q.copy()
    u, twist = np.zeros_like(q), np.zeros(N)
    a1, a2 = p['a1'], p['a2']
    history = np.zeros((steps+1, q.size))
    history[0] = q
    max_iterations, subdivisions = 0, 0

    def advance(q, u, a1, twist, interval, depth=0):
        nonlocal max_iterations, subdivisions
        try:
            state = step(q, u, a1, twist, interval, p)
            max_iterations = max(max_iterations, state[-1])
            return state[:5]
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            if depth == 4:
                raise
            subdivisions += 1
            mid = advance(q, u, a1, twist, interval/2, depth+1)
            return advance(mid[0], mid[1], mid[2], mid[4], interval/2, depth+1)

    for i in range(1, steps+1):
        q, u, a1, a2, twist = advance(q, u, a1, twist, dt)
        history[i] = q
        if progress and i % max(1, steps//5) == 0:
            print(f'    simulated {i*dt:g}/{totalTime:g} s', flush=True)
    times = np.arange(steps+1)*dt
    tip_z = history[:, -1]
    distance = np.linalg.norm(history[:, -3:] - history[:, :3], axis=1)
    last_second = times >= max(0., totalTime-1.) - 1e-10
    # Max deviation from final over the ENTIRE last second, not just endpoints.
    z_change = np.max(np.abs(tip_z[last_second]-tip_z[-1]))
    distance_change = np.max(np.abs(distance[last_second]-distance[-1]))
    theta = history[:, 3::4]
    twist_change = np.max(np.abs(theta[last_second]-theta[-1]))
    positional_change = np.max(np.linalg.norm(
        (history[last_second]-history[-1])[:, np.arange(q.size) % 4 != 3]
        .reshape(-1, N, 3), axis=2))
    z_steady = bool(z_change <= max(1e-9, .001*abs(tip_z[-1])))
    distance_steady = bool(distance_change <= max(1e-9, .001*distance[-1]))
    twist_steady = bool(twist_change <= max(1e-6, .001*np.max(np.abs(theta[-1]))))
    steady = distance_steady if zero_curvature and g == 0 else z_steady
    if natR > 0 and not zero_curvature:
        steady = steady and twist_steady
    return dict(q=q, initial=initial, time=times, tip_z=tip_z,
                distance=distance, theta=theta, history=history,
                z_change=z_change, distance_change=distance_change,
                twist_change=twist_change, positional_change=positional_change,
                steady=steady, z_steady=z_steady, distance_steady=distance_steady,
                twist_steady=twist_steady, max_iterations=max_iterations,
                subdivisions=subdivisions, parameters=p)


def nearest(value, options):
    return 'abcd'[np.argmin(np.abs(np.array(options)-value))]


def tests():
    p = setup()
    np.testing.assert_allclose(p['q'][4:7], 0.)
    np.testing.assert_allclose(p['q'][-3:], [.2, 0., 0.])
    np.testing.assert_allclose(p['ref_len'], .004)
    np.testing.assert_allclose(p['mass'][0::4].sum(), 1000*np.pi*.001**2*.204)
    assert p['q'].size == 207 and p['q'][7:].size == 200
    assert np.all(p['kappa_bar'] == 0.)
    np.testing.assert_allclose(eb_deflection(), .004)
    F, JF, *_ = forces(p['q'], p['q'], p['a1'], np.zeros(52), p)
    np.testing.assert_allclose(F, 0., atol=1e-10)
    np.testing.assert_allclose(JF, JF.T, atol=1e-8)
    rows, cols = np.indices(JF.shape)
    assert np.max(np.abs(JF[np.abs(rows-cols)>10])) < 1e-12
    c = setup(N=50, natR=.02)
    F, *_ = forces(c['q'], c['q'], c['a1'], np.zeros(50), c)
    np.testing.assert_allclose(F, 0., atol=1e-9)
    result = run(N=12, g=0., totalTime=.2)
    np.testing.assert_allclose(result['q'], result['initial'], atol=1e-12)
    result = run(N=12, g=.05, totalTime=.2)
    np.testing.assert_array_equal(result['q'][:7], result['initial'][:7])
    assert result['tip_z'][-1] < 0 and np.isfinite(result['q']).all()
    # An independent dense solve checks the banded solver away from rest.
    p = result['parameters']
    q = result['q']
    F, JF, *_ = forces(q, p['q'], p['a1'], np.zeros(12), p)
    matrix = -JF
    matrix[np.diag_indices(q.size)] += p['mass']/.1**2
    matrix = matrix[7:, 7:]
    n = matrix.shape[0]
    band = np.zeros((21, n))
    for offset in range(-10, 11):
        columns = np.arange(max(0, offset), min(n, n+offset))
        band[10-offset, columns] = matrix[columns-offset, columns]
    rhs = np.linspace(-1., 1., n)
    np.testing.assert_allclose(solve_banded((10, 10), band, rhs),
                               np.linalg.solve(matrix, rhs), rtol=1e-7, atol=1e-6)
    print('PASS: free length, total mass, DOFs, EB formula, straight/curved rest forces,')
    print('      symmetric banded Jacobian, fixed clamp, and implicit time steps.')


def homework(output, part='all'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = {}
    report = []

    def say(text):
        print(text, flush=True)
        report.append(text)
        (output/'answers.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')

    def case(name, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in runs:
            print(f'Running {name}: {kwargs}', flush=True)
            r = run(progress=True, **kwargs)
            runs[key] = r
            np.savez_compressed(output/f'{name}.npz', q=r['q'], initial=r['initial'],
                                time=r['time'], history=r['history'])
            np.savetxt(output/f'{name}_history.csv',
                       np.column_stack((r['time'], r['tip_z'], r['distance'],
                                        np.max(np.abs(r['theta']), axis=1))),
                       delimiter=',', header='time_s,tip_z_m,end_distance_m,max_abs_theta_rad',
                       comments='')
            say(f'{name}: steady={r["steady"]}; last-second max changes: '
                f'z={r["z_change"]:.3g} m, distance={r["distance_change"]:.3g} m, '
                f'twist={r["twist_change"]:.3g} rad; max Newton iterations={r["max_iterations"]}; '
                f'half-step retries={r["subdivisions"]}.')
            if not r['steady']:
                say('WARNING: steady-state criterion not met; extend totalTime before using this result.')
        return runs[key]

    say('HW3: results from 3D DER simulations (dt=0.1 s, total time=10 s).')
    default = None
    if part in ('all', '1'):
        eb = eb_deflection()
        say(f'Q1: ({nearest(eb, [.0005,.001,.004,.032])}) EB={eb:.9g} m = {eb*1000:.6g} mm.')
        small = case('small_gravity', g=.05)
        delta = -small['q'][-1]
        say(f'Q2: ({nearest(delta,[.001,.0042,.016,.00026])}) DER={delta:.9g} m.')
        say('Q3: (a) first two nodes are fixed; the first edge is inside the clamp, so free length is (N-2)*dl.')
        mesh_rows = []
        for N in (22, 42, 82):
            r = case(f'mesh_N{N}', N=N, g=.05)
            delta = -r['q'][-1]
            mesh_rows.append([N, delta, eb, abs(delta-eb)/eb])
            say(f'Q4: N={N}, DER={delta:.9g} m, EB={eb:.9g} m, difference={abs(delta-eb)/eb:.3%}.')
        say('Q4: (d) slightly above EB; the mesh error decreases as free edges are doubled.')
        np.savetxt(output/'mesh_sweep.csv', mesh_rows, delimiter=',',
                   header='N,DER_m,EB_m,relative_difference', comments='')
        stiff_rows = []
        for E_mpa in (10, 30, 100, 300, 1000):
            r = case(f'stiffness_{E_mpa}MPa', E=E_mpa*1e6)
            if E_mpa == 10:
                default = r
            delta, eb = -r['q'][-1], eb_deflection(E=E_mpa*1e6, g=9.81)
            error = abs(delta-eb)/eb
            stiff_rows.append([E_mpa, delta, eb, error])
            say(f'Q5: E={E_mpa} MPa, DER={delta:.9g} m, EB={eb:.9g} m, e={error:.3%}.')
        passing = [row[0] for row in stiff_rows if row[-1] < .2]
        if passing:
            value = min(passing)
            label = {10:'a',30:'b',100:'c',1000:'d'}.get(value, 'not a listed choice')
            say(f'Q5: ({label}) smallest tested E with e<20%: {value} MPa.')
        else:
            say('Q5: no tested E meets e<20%; inspect convergence.')
        np.savetxt(output/'stiffness_sweep.csv', stiff_rows, delimiter=',',
                   header='E_MPa,DER_m,EB_m,relative_difference', comments='')
        gravity_rows = []
        for g in (.1, .3, 1., 3., 9.81):
            r = default if g == 9.81 else case(f'gravity_{g:g}', g=g)
            delta, eb = -r['q'][-1], eb_deflection(g=g)
            error = abs(delta-eb)/eb
            gravity_rows.append([g, delta, eb, error])
            say(f'Q6: g={g:g} m/s^2, DER={delta:.9g} m, EB={eb:.9g} m, e={error:.3%}.')
        failing = [row[0] for row in gravity_rows if row[-1] > .2]
        if failing:
            value = min(failing)
            label = {.1:'a',.3:'b',1.:'c',3.:'d'}.get(value, 'not a listed choice')
            say(f'Q6: ({label}) smallest tested gravity with e>20%: {value:g} m/s^2.')
        else:
            say('Q6: no tested gravity exceeds 20%; inspect convergence.')
        np.savetxt(output/'gravity_sweep.csv', gravity_rows, delimiter=',',
                   header='g_m_per_s2,DER_m,EB_m,relative_difference', comments='')
        delta = -default['q'][-1]
        say(f'Q7: ({nearest(delta,[.02,.18,.06,.78])}) DER={delta:.9g} m; EB={eb_deflection(g=9.81):.9g} m.')
        say('Q8: (a) linear EB assumes small slopes and near-straight geometry; DER includes large rotations.')

    if part in ('all', '2'):
        if default is None:
            default = case('straight_default', E=10e6)
        curved = case('curved_gravity', N=50, natR=.02)
        z = curved['q'][-1]
        say(f'Q9: ({nearest(z,[-.01,-.18,-.04,-.10])}) curved rod tip z={z:.9g} m.')
        say(f'Q10: max |theta|: straight={np.max(np.abs(default["q"][3::4])):.9g} rad; '
            f'curved={np.max(np.abs(curved["q"][3::4])):.9g} rad.')
        say('Q10: (b) straight rod bends in a plane with zero twist angles; curved rod develops nonzero angles.')
        opened = case('zero_natural_curvature', N=50, natR=.02, g=0., zero_curvature=True)
        say(f'Q11: (d) removal of natural curvature opens the rod. End distance: '
            f'initial={opened["distance"][0]:.9g} m, final={opened["distance"][-1]:.9g} m; '
            f'initial discrete contour length={opened["parameters"]["ref_len"].sum():.9g} m.')
        plots(output, default, curved, opened)
    say(f'Results saved to {output.resolve()}')


def plots(output, straight, curved, opened):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    for i, (r, title) in enumerate(((straight,'Straight rod'), (curved,'Natural curved rod'),
                                    (opened,'Zero natural curvature, g=0'))):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        for key, style, label in [('initial','--','Initial'), ('q','-','Final')]:
            ax.plot(r[key][0::4], r[key][1::4], r[key][2::4], style, label=label)
        ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title=title)
        xyz = np.r_[r['initial'][np.arange(r['q'].size)%4!=3].reshape(-1,3),
                    r['q'][np.arange(r['q'].size)%4!=3].reshape(-1,3)]
        center = (xyz.max(axis=0)+xyz.min(axis=0))/2
        radius = max(np.ptp(xyz,axis=0).max()/2, .01)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1,1,1))
        ax.legend(fontsize=8)
    fig.savefig(output/'rod_shapes.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for r, label in [(straight,'Straight'),(curved,'Curved')]:
        axes[0].plot(r['time'], r['tip_z'], label=label)
        axes[1].plot(np.arange(len(r['q'][3::4])), r['q'][3::4], label=label)
    axes[0].set(xlabel='Time (s)', ylabel='Tip z (m)', title='Steady-state histories')
    axes[1].set(xlabel='Edge index', ylabel='Twist angle (rad)', title='Q10: final twist angles')
    axes[2].plot(opened['time'], opened['distance'])
    axes[2].set(xlabel='Time (s)', ylabel='End-to-end distance (m)', title='Q11: rod opening')
    for ax in axes:
        ax.grid(alpha=.3)
    axes[0].legend()
    axes[1].legend()
    fig.savefig(output/'histories_and_twist.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--part', choices=['all','1','2'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    if args.test:
        tests()
    else:
        homework(args.output_dir, args.part)


if __name__ == '__main__':
    main()
