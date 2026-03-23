"""
Radiative transfer solver for CO line emission in a circumstellar envelope.

Phase A: Large Velocity Gradient (Sobolev / escape probability) approximation.

The LVG method computes the probability that a photon emitted in a line
escapes the local region without being reabsorbed, based on the local
velocity gradient. This decouples the radiative transfer from the global
problem and allows fast, shell-by-shell solution of statistical equilibrium.
"""
import numpy as np
from mcrt.constants import (
    h_planck, c_light, k_boltz, T_CMB, m_H
)


def sobolev_tau(n_l, n_u, g_l, g_u, A_ul, nu, dvdr):
    """
    Sobolev (LVG) optical depth for a transition.

    tau = (A_ul * c^3 / (8 * pi * nu^3)) * (n_l * g_u / g_l - n_u) / |dv/dr|

    Parameters
    ----------
    n_l, n_u : float
        Number densities of lower and upper levels [cm^-3].
    g_l, g_u : float
        Statistical weights.
    A_ul : float
        Einstein A coefficient [s^-1].
    nu : float
        Line frequency [Hz].
    dvdr : float
        Absolute velocity gradient |dv/dr| [s^-1].

    Returns
    -------
    tau : float
        Sobolev optical depth (can be negative for inversion).
    """
    if dvdr < 1e-10:
        dvdr = 1e-10  # prevent division by zero
    tau = (A_ul * c_light**3 / (8.0 * np.pi * nu**3)) * \
          (n_l * g_u / g_l - n_u) / dvdr
    return tau


def escape_probability(tau):
    """
    Escape probability for the LVG (Sobolev) approximation.

    beta(tau) = (1 - exp(-tau)) / tau   for tau != 0
    beta(0) = 1

    For negative tau (population inversion), use the same formula
    which gives beta > 1 (amplification).
    """
    if abs(tau) < 1e-6:
        # Taylor expansion: beta ~ 1 - tau/2 + tau^2/6
        return 1.0 - tau / 2.0 + tau**2 / 6.0
    return (1.0 - np.exp(-tau)) / tau


def mean_intensity_LVG(tau, beta, S_ul, J_ext):
    """
    Mean intensity in the LVG approximation.

    J_bar = (1 - beta) * S_ul + beta * J_ext

    where S_ul is the line source function and J_ext is the external
    radiation field (CMB + dust).

    Parameters
    ----------
    tau : float
        Sobolev optical depth.
    beta : float
        Escape probability.
    S_ul : float
        Source function [erg/s/cm^2/Hz/sr].
    J_ext : float
        External mean intensity [erg/s/cm^2/Hz/sr].

    Returns
    -------
    J_bar : float
        Angle-averaged mean intensity.
    """
    return (1.0 - beta) * S_ul + beta * J_ext


def source_function(n_u, n_l, g_u, g_l, nu):
    """
    Line source function S_ul = (2h*nu^3/c^2) / (n_l*g_u/(n_u*g_l) - 1).
    """
    if n_u < 1e-30:
        return 0.0
    ratio = (n_l * g_u) / (n_u * g_l)
    if ratio <= 1.0:
        # Population inversion -- set source function to large value
        return 1e10 * planck_Bnu(nu, 1e4)
    return (2.0 * h_planck * nu**3 / c_light**2) / (ratio - 1.0)


def planck_Bnu(nu, T):
    """Planck function B_nu(T) [erg/s/cm^2/Hz/sr]."""
    if T < 0.01:
        return 0.0
    x = h_planck * nu / (k_boltz * T)
    if x > 500:
        return 0.0
    return (2.0 * h_planck * nu**3 / c_light**2) / (np.exp(x) - 1.0)


def solve_statistical_equilibrium(n_CO, n_H2, T_gas, T_dust, dvdr,
                                  mol_data, J_ext_func=None,
                                  max_iter=200, tol=1e-4):
    """
    Solve statistical equilibrium for CO level populations in one shell
    using the LVG (escape probability) method.

    Iterates:
    1. Compute Sobolev tau and escape probability from current populations.
    2. Build rate matrix including radiative (with escape prob) and collisional rates.
    3. Solve for new populations.
    4. Repeat until converged.

    Parameters
    ----------
    n_CO : float
        Total CO number density [cm^-3].
    n_H2 : float
        H2 number density [cm^-3].
    T_gas : float
        Gas kinetic temperature [K].
    T_dust : float
        Dust temperature [K] (for background radiation).
    dvdr : float
        Absolute velocity gradient |dv/dr| [s^-1].
    mol_data : MolecularData
        Molecular line data.
    J_ext_func : callable or None
        Function(nu) returning external radiation field J_ext.
        If None, uses CMB only.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on fractional population change.

    Returns
    -------
    pops : array (n_levels,)
        Fractional level populations (sum = 1).
    J_bar : array (n_trans,)
        Mean intensity for each transition.
    converged : bool
        Whether the iteration converged.
    """
    nl = mol_data.n_levels
    nt = mol_data.n_trans

    # Initial guess: LTE populations
    pops = mol_data.LTE_populations(T_gas)
    J_bar = np.zeros(nt)

    # Get collisional rates at this temperature
    gamma_ul_all = mol_data.gamma_ul(T_gas)
    gamma_lu_all = mol_data.gamma_lu(T_gas)

    for iteration in range(max_iter):
        pops_old = pops.copy()

        # Compute Sobolev tau, escape probability, and J_bar for each transition
        for t in range(nt):
            u = mol_data.upper[t]
            l = mol_data.lower[t]
            nu = mol_data.nu[t]
            A_ul = mol_data.A[t]

            n_u = pops[u] * n_CO
            n_l = pops[l] * n_CO

            tau = sobolev_tau(n_l, n_u, mol_data.g[l], mol_data.g[u],
                              A_ul, nu, dvdr)
            beta = escape_probability(tau)

            # External radiation: CMB + dust emission
            J_cmb = planck_Bnu(nu, T_CMB)
            J_dust = planck_Bnu(nu, T_dust)
            # Simple dilution: dust fills a small solid angle, parametrised
            # as a dilution factor W_dust that decreases with distance
            # For the LVG approximation, use CMB as primary background
            J_bg = J_cmb + 0.0 * J_dust  # dust coupling handled via heating term

            if J_ext_func is not None:
                J_bg = J_ext_func(nu)

            S = source_function(n_u, n_l, mol_data.g[u], mol_data.g[l], nu)
            J_bar[t] = mean_intensity_LVG(tau, beta, S, J_bg)

        # Build and solve rate equations
        # Rate matrix M: M[i,j] = rate from level j to level i
        # Diagonal: M[i,i] = -sum of rates out of level i
        M = np.zeros((nl, nl))

        # Radiative rates (only J+1 -> J dipole transitions)
        for t in range(nt):
            u = mol_data.upper[t]
            l = mol_data.lower[t]
            nu = mol_data.nu[t]
            A_ul = mol_data.A[t]
            B_ul = mol_data.B_ul[t]
            B_lu = mol_data.B_lu[t]

            n_u = pops[u] * n_CO
            n_l = pops[l] * n_CO
            tau = sobolev_tau(n_l, n_u, mol_data.g[l], mol_data.g[u],
                              A_ul, nu, dvdr)
            beta = escape_probability(tau)

            # Effective radiative rates (modified by escape probability)
            # Rate u -> l: A_ul * beta + B_ul * J_bar (stimulated emission)
            # Rate l -> u: B_lu * J_bar (absorption)
            R_ul = A_ul * beta + B_ul * J_bar[t]
            R_lu = B_lu * J_bar[t]

            M[l, u] += R_ul   # u -> l
            M[u, l] += R_lu   # l -> u

        # Collisional rates
        for idx, (u, l) in enumerate(mol_data.col_pairs):
            if u >= nl or l >= nl:
                continue
            C_ul = gamma_ul_all[idx] * n_H2
            C_lu = gamma_lu_all[idx] * n_H2
            M[l, u] += C_ul   # u -> l
            M[u, l] += C_lu   # l -> u

        # Fill diagonal: M[i,i] = -sum of rates out of level i
        for i in range(nl):
            M[i, i] = -np.sum(M[:, i]) + M[i, i]  # subtract self from sum

        # Replace last equation with conservation: sum(n_i) = 1
        M[-1, :] = 1.0
        rhs = np.zeros(nl)
        rhs[-1] = 1.0

        # Solve
        try:
            pops = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Singular matrix -- fall back to LTE
            pops = mol_data.LTE_populations(T_gas)
            break

        # Ensure non-negative populations
        pops = np.clip(pops, 0.0, None)
        total = pops.sum()
        if total > 0:
            pops /= total
        else:
            pops = mol_data.LTE_populations(T_gas)

        # Check convergence
        max_change = np.max(np.abs(pops - pops_old) / (pops_old + 1e-30))
        if max_change < tol:
            return pops, J_bar, True

    return pops, J_bar, False


def solve_all_shells(env, mol_data, isotope='12CO'):
    """
    Solve LVG statistical equilibrium for all shells.

    Parameters
    ----------
    env : Envelope
        The envelope model with current T_gas.
    mol_data : MolecularData
        Molecular data for this isotopologue.
    isotope : str
        '12CO' or '13CO' to select the right density.

    Returns
    -------
    all_pops : array (n_shells, n_levels)
        Level populations at each shell.
    all_Jbar : array (n_shells, n_trans)
        Mean intensities at each shell.
    """
    ns = env.n_shells
    nl = mol_data.n_levels
    nt = mol_data.n_trans

    all_pops = np.zeros((ns, nl))
    all_Jbar = np.zeros((ns, nt))

    for i in range(ns):
        if isotope == '12CO':
            n_CO = env.n_12CO[i]
        else:
            n_CO = env.n_13CO[i]

        pops, Jbar, converged = solve_statistical_equilibrium(
            n_CO=n_CO,
            n_H2=env.n_H2[i],
            T_gas=env.T_gas[i],
            T_dust=env.T_dust[i],
            dvdr=env.dvdr[i],
            mol_data=mol_data,
        )
        all_pops[i] = pops
        all_Jbar[i] = Jbar

    return all_pops, all_Jbar
