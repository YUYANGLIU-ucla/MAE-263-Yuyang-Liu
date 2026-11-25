"""
theory.py -- closed-form / semi-analytical reference results used for validation.
MAE 263F (Fall 2026), Yuyang Liu.
"""
import numpy as np


def _ellip(k, n=20001):
    """Complete elliptic integrals K(k), E(k) by direct numerical integration."""
    th = np.linspace(0, np.pi / 2, n)
    s = np.sqrt(1 - (k * np.sin(th)) ** 2)
    dth = th[1] - th[0]
    trap = lambda y: dth * (np.sum(y) - 0.5 * (y[0] + y[-1]))      # trapezoidal rule
    return trap(1 / s), trap(s)


def elastica_clamped(eps):
    """
    Exact inextensible elastica for a clamped-clamped strip with end shortening eps.
    The first buckling mode is made of four quarter-waves (inflection points at L/4,
    3L/4), so with modulus k = sin(alpha/2), alpha = largest slope angle:
        eps = 2 (1 - E(k)/K(k)),   w0/L = k / K(k),   P L^2/EI = 16 K(k)^2.
    Returns (w0/L, P L^2/EI).
    """
    lo, hi = 1e-6, 0.999
    for _ in range(60):                      # bisection for k
        k = 0.5 * (lo + hi)
        K, E = _ellip(k)
        if 2 * (1 - E / K) < eps:
            lo = k
        else:
            hi = k
    return k / K, 16 * K**2


def small_eps(eps):
    """Leading-order results: w0/L = (2/pi) sqrt(eps), P L^2/EI = 4 pi^2."""
    return 2 / np.pi * np.sqrt(eps), 4 * np.pi**2


# second buckling eigenvalue of a clamped-clamped column: tau/2 = tan(tau/2)
TAU2 = 8.986818916
TAU1 = 2 * np.pi


def point_force_center(eps, S):
    """
    Midpoint point force on a clamped arch, Pandey et al., EPL 105, 24001 (2014), eq. (3):
        F_cr L^2/EI = 207.75 w0/L = 129.53 (eps - 80.76 S)^(1/2),
    with stretchability S = h^2/(12 L^2).  At F_cr the arch switches to the asymmetric
    mode; on the asymmetric branch F L^2/EI = 207.75 * (w/L).  The constants are the
    ones printed in the paper.  Returns (F_cr L^2/EI, slope).
    """
    fcr = 129.53 * np.sqrt(eps - 80.76 * S)
    slope = 207.75
    return fcr, slope


def point_force_offset(eps, delta, n=20001):
    """
    Critical point force at position delta (Yan, Yu & Mehta, TAML 2019, two-mode model,
    axial compressibility neglected):
        F_cr L^2/EI = sqrt(eps) * max_eta 1/sqrt( w1^2/(n1^2 (eta^2-n1^2)^2)
                                                 + w2^2/(n2^2 (eta^2-n2^2)^2) ),
    n1 = 2 pi, n2 = 2.86 pi, 2 pi < eta < 2.86 pi, with the mode shapes evaluated at delta.
    """
    n1, n2 = TAU1, TAU2
    w1 = 1 - np.cos(n1 * delta)
    w2 = 1 - np.cos(n2 * delta) - 2 / n2 * (n2 * delta - np.sin(n2 * delta))
    eta = np.linspace(n1, n2, n)[1:-1]
    den = w1**2 / (n1**2 * (eta**2 - n1**2) ** 2) + w2**2 / (n2**2 * (eta**2 - n2**2) ** 2)
    return np.sqrt(eps) * np.max(1 / np.sqrt(den))


def antisymmetric_frequency(P=4 * np.pi**2):
    """
    Lowest antisymmetric vibration frequency (in units of 1/t*) of a clamped-clamped
    beam carrying the axial compressive load P (in units of EI/L^2).  From
    w'''' + P w'' = omega^2 w with w = A sinh(a x) + B sin(b x) on -1/2 < x < 1/2 and
    w = w' = 0 at x = 1/2:   a cosh(a/2) sin(b/2) - b sinh(a/2) cos(b/2) = 0,
    a^2 = (-P + sqrt(P^2 + 4 omega^2))/2,  b^2 = (P + sqrt(P^2 + 4 omega^2))/2.
    For the buckled arch P = 4 pi^2, and this mode does not change the arc length to
    first order, so its frequency does not depend on the end shortening.
    """
    def f(om):
        r = np.sqrt(P**2 + 4 * om**2)
        a, b = np.sqrt((-P + r) / 2), np.sqrt((P + r) / 2)
        return a * np.cosh(a / 2) * np.sin(b / 2) - b * np.sinh(a / 2) * np.cos(b / 2)
    oms = np.linspace(1.0, 120.0, 12000)
    vals = f(oms)
    i = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0][0]
    lo, hi = oms[i], oms[i + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sign(f(mid)) == np.sign(f(lo)):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
