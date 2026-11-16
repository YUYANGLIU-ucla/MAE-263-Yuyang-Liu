"""MAE-263F HW4: triangle topology, hinge energy and gradient checks.

Run: python Homework4.py          (Q1--Q11 and saved results)
     python Homework4.py --test   (independent verification)

The companion shell derivative function is copied unchanged from the public
course repository; see THIRD_PARTY_NOTICES.md. Other routines below are the
homework-specific implementation and independent checks.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from gradEb_hessEb_Shell import getTheta, gradEb_hessEb_Shell


Q = np.array([0., 0., 0., 1., 0., 0., .5, 1., 0., .5, -1., 1.])
EPS = .01


def plate_mesh():
    """9 nodes; each unit square split along its lower-left/upper-right diagonal."""
    nodes = np.array([[i, j, 0.] for j in range(3) for i in range(3)])
    triangles = []
    for j in range(2):
        for i in range(2):
            a = 3*j+i
            triangles.extend(((a, a+1, a+4), (a, a+4, a+3)))
    return nodes, np.array(triangles, dtype=int)


def getEdgesAndHinges(elements):
    """Count unique undirected edges and shared edges without course dependencies.

    Return edges (n,2), hinges (m,4) = [edge node0, node1, opposite2, opposite3].
    A boundary edge has one adjacent triangle, a hinge has exactly two.
    """
    adjacency = {}
    elements = np.asarray(elements)
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError('elements must be an (n,3) integer array')
    if not np.issubdtype(elements.dtype, np.integer) or np.any(elements < 0):
        raise ValueError('Triangle indices must be nonnegative integers')
    seen = set()
    for triangle in elements:
        if len(set(triangle)) != 3:
            raise ValueError('Triangle has a repeated node')
        key = tuple(sorted(triangle))
        if key in seen:
            raise ValueError('Duplicate triangle')
        seen.add(key)
        for a, b, opposite in (triangle, np.roll(triangle, -1), np.roll(triangle, -2)):
            edge = tuple(sorted((int(a), int(b))))
            adjacency.setdefault(edge, []).append(int(opposite))
    if any(len(opposites) > 2 for opposites in adjacency.values()):
        raise ValueError('Nonmanifold edge: more than two adjacent triangles')
    edges = sorted(adjacency)
    hinges = [(*edge, *adjacency[edge]) for edge in edges if len(adjacency[edge]) == 2]
    return (np.asarray(edges, dtype=int).reshape(-1, 2),
            np.asarray(hinges, dtype=int).reshape(-1, 4))


def Eb(q, thetaBar=0., kb=1.):
    """Bending energy in joules. The returned dF is +grad(E), not physical force."""
    return .5*kb*(getTheta(np.asarray(q, dtype=float))-thetaBar)**2


def perturbed_energies(q, eps=EPS):
    """Each perturbation starts from the ORIGINAL q; never accumulate changes."""
    if eps <= 0:
        raise ValueError('eps must be positive')
    result = np.zeros(12)
    for i in range(12):
        trial = np.asarray(q, dtype=float).copy()
        trial[i] += eps
        result[i] = Eb(trial)
    return result


def independent_angle_gradient(q):
    """Independent analytical chain rule through the two triangle normals.

    This does NOT call the course gradTheta/gradient function. It is an extra
    check of the sign, ordering and magnitude returned by the supplied code.
    """
    x0, x1, x2, x3 = np.asarray(q, dtype=float).reshape(4, 3)
    e, u, v = x1-x0, x2-x0, x3-x0
    a, b = np.cross(e, u), np.cross(v, e)
    if min(np.linalg.norm(e), np.linalg.norm(a), np.linalg.norm(b)) < 1e-12:
        raise ValueError('Degenerate hinge')
    h = e/np.linalg.norm(e)
    da = -np.cross(h, a)/np.dot(a, a)
    db = np.cross(h, b)/np.dot(b, b)
    ge = np.cross(u, da) + np.cross(db, v)
    gu, gv = np.cross(da, e), np.cross(e, db)
    return np.r_[-ge-gu-gv, ge, gu, gv]


def central_gradient(q, step=1e-6):
    eye = np.eye(12)*step
    return np.array([(Eb(q+d)-Eb(q-d))/(2*step) for d in eye])


def closest(value, options):
    """Nearest multiple-choice numeric answer, never using rounded intermediate values."""
    distance = [np.linalg.norm(np.asarray(value)-np.asarray(option)) for option in options]
    return 'abcd'[int(np.argmin(distance))]


def grouped(v):
    return '[ ' + ' | '.join(', '.join(f'{x:.9f}' for x in row)
                            for row in np.asarray(v).reshape(4, 3)) + ' ]'


def compute():
    nodes, elements = plate_mesh()
    edges, hinges = getEdgesAndHinges(elements)
    angle = getTheta(Q)
    base = Eb(Q)
    perturbed = perturbed_energies(Q)
    # REQUIRED forward difference, with eps=0.01 m and full precision energies.
    fd = (perturbed-base)/EPS
    gradient, hessian = gradEb_hessEb_Shell(Q, None, None, None, 0., 1.)
    choices = {}
    choices[1] = closest([len(edges),len(hinges)], [[12,4],[16,16],[16,8],[20,8]])
    choices[2] = closest([angle,base], [[np.pi/4,.308],[-np.pi/4,.308],
                                      [-np.pi/4,.617],[np.pi/2,1.234]])
    energy_options = {
        3: [[.30843,.30649,.30254],[.30843,.31042,.31432],
            [.30254,.30649,.30843],[.30843,.31238,.31234]],
        4: [[.30843,.30843,.31633],[.30843,.31042,.31432],
            [.61685,.61298,.60508],[.30843,.30649,.30254]],
        5: [[.30843,.30649,.30254],[.30843,.30843,.30062],
            [.30843,.30843,.31633],[.61685,.61685,.63266]],
        6: [[.30843,.30453,.30449],[.30843,.31238,.31234],
            [.30843,.30843,.31633],[.31234,.31238,.30843]],
    }
    for node in range(4):
        choices[node+3] = closest(perturbed[3*node:3*node+3], energy_options[node+3])
    choices[7] = closest(fd, [
        [0,.194,.589,0,.194,.589,0,0,-.790,0,-.396,-.392],
        [0,-.00194,-.00589,0,-.00194,-.00589,0,0,.00790,0,.00396,.00392],
        [0,-.194,-.589,0,-.194,-.589,0,.396,.392,0,0,.790],
        [0,-.194,-.589,0,-.194,-.589,0,0,.790,0,.396,.392]])
    choices[8] = closest(gradient, [
        [0,-.196,-.589,0,-.196,-.589,0,0,.785,0,.393,.393],
        [0,.196,.589,0,.196,.589,0,0,-.785,0,-.393,-.393],
        [0,-.393,-1.178,0,-.393,-1.178,0,0,1.571,0,.785,.785],
        [0,0,.785,0,-.196,-.589,0,-.196,-.589,0,.393,.393]])
    # Q9--Q11 are conceptual, not inferred from the hinge calculations.
    choices.update({9:'c', 10:'d', 11:'a'})
    return dict(nodes=nodes, elements=elements, edges=edges, hinges=hinges,
                angle=angle, base=base, perturbed=perturbed, fd=fd,
                gradient=gradient, hessian=hessian, choices=choices)


def test():
    r = compute()
    assert r['nodes'].shape == (9,3) and r['elements'].shape == (8,3)
    assert len(r['edges']) == 16 and len(r['hinges']) == 8
    assert 9-16+8 == 1  # Euler characteristic of a disk
    assert 3*8 == 2*8+8  # interior edges counted twice, boundary once
    edges, hinges = getEdgesAndHinges(r['elements'][::-1, ::-1])
    np.testing.assert_array_equal(edges, r['edges'])
    np.testing.assert_array_equal(hinges[:,:2], r['hinges'][:,:2])
    # Signed normal construction: n0=(0,0,1), n1=(0,1,1), so theta=-pi/4.
    np.testing.assert_allclose(r['angle'], -np.pi/4, atol=1e-14)
    np.testing.assert_allclose(r['base'], np.pi**2/32, atol=1e-14)
    np.testing.assert_allclose(getTheta(*Q.reshape(4,3)), r['angle'], atol=1e-14)
    flat = Q.copy()
    flat[11] = 0.
    np.testing.assert_allclose(getTheta(flat), 0., atol=1e-14)
    mirror = Q.copy()
    mirror[11] = -1.
    np.testing.assert_allclose(getTheta(mirror), np.pi/4, atol=1e-14)
    np.testing.assert_allclose(r['gradient'], central_gradient(Q), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(r['gradient'],
                               r['angle']*independent_angle_gradient(Q), atol=1e-12)
    np.testing.assert_allclose(r['hessian'], r['hessian'].T, atol=1e-12)
    # Check the energy Hessian by differencing gradients, independently of hessTheta.
    hess_fd = np.column_stack([
        (gradEb_hessEb_Shell(Q+d,None,None,None,0.,1.)[0]
         -gradEb_hessEb_Shell(Q-d,None,None,None,0.,1.)[0])/2e-6
        for d in np.eye(12)*1e-6])
    np.testing.assert_allclose(r['hessian'], hess_fd, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r['gradient'].reshape(4,3).sum(axis=0), 0., atol=1e-12)
    translation = np.tile([1.2,-.7,2.1],4)
    np.testing.assert_allclose(Eb(Q+translation), r['base'], atol=1e-13)
    for i in range(12):
        trial = Q.copy()
        trial[i] += EPS
        np.testing.assert_allclose(r['perturbed'][i], Eb(trial), atol=1e-14)
    errors = [np.linalg.norm((perturbed_energies(Q,e)-r['base'])/e-r['gradient'])
              for e in (.01,.005,.0025)]
    assert errors[0] > errors[1] > errors[2]
    # Additional bent configurations, not merely the homework's symmetric hinge.
    rng = np.random.default_rng(2634)
    for _ in range(8):
        q = Q+rng.normal(0.,.05,12)
        grad, _ = gradEb_hessEb_Shell(q,None,None,None,0.,1.)
        np.testing.assert_allclose(grad,central_gradient(q),rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(grad,getTheta(q)*independent_angle_gradient(q),atol=1e-10)
    # Undamped oscillator m=k=1, dt=0.2: implicit Euler loses energy;
    # Newmark average acceleration (beta=.25, gamma=.5) preserves it.
    x_ie, v_ie, x_nm, v_nm, dt = 1., 0., 1., 0., .2
    energies_ie, energies_nm = [.5], [.5]
    for _ in range(100):
        x_ie, v_ie = (x_ie+dt*v_ie)/(1+dt**2), (v_ie-dt*x_ie)/(1+dt**2)
        acceleration = -x_nm
        x_next = (x_nm+dt*v_nm+.25*dt**2*acceleration)/(1+.25*dt**2)
        v_nm = v_nm+.5*dt*(acceleration-x_next)
        x_nm = x_next
        energies_ie.append(.5*(x_ie*x_ie+v_ie*v_ie))
        energies_nm.append(.5*(x_nm*x_nm+v_nm*v_nm))
    assert np.all(np.diff(energies_ie)<0.)
    np.testing.assert_allclose(energies_nm,.5,atol=1e-13)
    print('PASS: mesh topology, signed hinge angle and energy, all 12 perturbations.')
    print('PASS: analytical gradient vs central differences and independent chain rule.')
    print('PASS: Hessian, symmetry, translation invariance, finite-difference convergence.')
    print('PASS: implicit-Euler energy decay and Newmark average-acceleration energy conservation.')


def save_results(r, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = ['MAE-263F Homework 4', 'Options follow the PDF order; check text if Gradescope reorders choices.', '']
    choices = r['choices']
    lines.append(f'Q1: ({choices[1]}) {len(r["edges"])} edges and {len(r["hinges"])} hinges; 8 boundary edges.')
    lines.append(f'Q2: ({choices[2]}) theta={r["angle"]:.12f} rad = -pi/4; Eb={r["base"]:.12f} J.')
    for node in range(4):
        values = ', '.join(f'{e:.12f}' for e in r['perturbed'][3*node:3*node+3])
        lines.append(f'Q{node+3}: ({choices[node+3]}) node x{node}, separate +0.01 m x/y/z perturbations: {values} J.')
    lines.extend([f'Q7: ({choices[7]}) forward-difference energy gradient (N), grouped by node:',
                  grouped(r['fd']),
                  f'Q8: ({choices[8]}) analytical ENERGY gradient dF (N), NOT physical force:',
                  grouped(r['gradient']),
                  f'Max |forward difference - analytical| = {np.max(np.abs(r["fd"]-r["gradient"])):.9g} N.',
                  'Physical bending force = -dF. Forward differences at eps=0.01 have truncation error.',
                  f'Q9: ({choices[9]}) total energy decreases: implicit Euler introduces numerical damping.',
                  f'Q10: ({choices[10]}) beta=0.25, gamma=0.5: average-acceleration Newmark conserves energy for an undamped linear oscillator.',
                  f'Q11: ({choices[11]}) corrector needed when the contact active set changes:',
                  '     activate a free node penetrating the ground; release a contact node requiring a downward/adhesive ground reaction.',
                  '', 'Answer sequence: '+', '.join(choices[i].upper() for i in range(1,12))])
    text = '\n'.join(lines)+'\n'
    print(text)
    (output/'answers.txt').write_text(text,encoding='utf-8')
    for name in ('nodes','elements','edges','hinges','hessian'):
        np.savetxt(output/f'{name}.csv',r[name],delimiter=',',fmt='%.17g')
    with (output/'energy_gradient_check.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['DOF_0_based','node','coordinate','base_energy_J','perturbed_energy_J',
                         'forward_gradient_N','analytical_gradient_N','difference_N'])
        for i in range(12):
            writer.writerow([i,i//3,'xyz'[i%3],r['base'],r['perturbed'][i],
                             r['fd'][i],r['gradient'][i],r['fd'][i]-r['gradient'][i]])
    print(f'Results saved to {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test',action='store_true',help='independent analytical and numerical checks')
    parser.add_argument('--answers',action='store_true',help='same as the default complete run')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    if args.test:
        test()
    else:
        save_results(compute(),args.output_dir)


if __name__ == '__main__':
    main()
