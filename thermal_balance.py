"""
Thermal balance for the IRC+10216 envelope.

Computes heating and cooling rates and solves for T(r) by integrating
the energy equation outward from the inner boundary.

Heating:
    - Dust-gas frictional (drift) heating
    - CO vibrational (IR pumping) heating via 4.67 micron band
    - Grain photoelectric heating (outer envelope)

Cooling:
    - Adiabatic expansion cooling (PdV work)
    - CO rotational line cooling (from level populations)

Note on CO cooling: In the inner envelope, the LVG approximation
overestimates CO cooling because it does not include the strong
stellar/dust radiation field at mm wavelengths that partially
thermalizes the CO lines. We limit the CO cooling contribution
to prevent numerical instability. The full MC (Phase B) handles
this self-consistently.
"""
import numpy as np
from scipy.integrate import solve_ivp
from constants import (
    M_dot, L_star, c_light, k_boltz, m_H, mu, T_CMB,
    h_planck, G0_ISRF, epsilon_pe, T_star, R_star, X_12CO
)


def gamma_H2(T):
    """
    Adiabatic index for molecular hydrogen.

    At T < 80 K only translational DOF are excited -> gamma = 5/3.
    At 80 < T < 300 K rotational DOF are excited -> gamma = 7/5.
    Smooth transition between regimes.
    """
    if T < 60:
        return 5.0 / 3.0
    elif T > 300:
        return 7.0 / 5.0
    else:
        f = (T - 60.0) / 240.0
        return (1.0 - f) * 5.0 / 3.0 + f * 7.0 / 5.0


def planck_Bnu(nu, T):
    """Planck function B_nu(T) [erg/s/cm^2/Hz/sr]."""
    if T < 0.01:
        return 0.0
    x = h_planck * nu / (k_boltz * T)
    if x > 500.0:
        return 0.0
    return (2.0 * h_planck * nu**3 / c_light**2) / (np.exp(x) - 1.0)


def heating_dustgas(r, Q_dg):
    """
    Dust-gas frictional (drift) heating rate per unit volume [erg/cm^3/s].

    H_dg = Q * M_dot * L_star / (16 * pi^2 * r^4 * c)
    """
    return Q_dg * M_dot * L_star / (16.0 * np.pi**2 * r**4 * c_light)


def heating_IR_pumping(r, n_CO, n_H2, T_gas, T_dust):
    """
    IR vibrational pumping heating via CO v=1 <- v=0 at 4.67 micron.
    """
    nu_10 = 6.42e13
    h_nu = h_planck * nu_10
    A_10 = 34.0

    B_01 = A_10 * c_light**2 / (2.0 * h_planck * nu_10**3)

    W_star = (R_star / r)**2 / 4.0
    J_star = W_star * planck_Bnu(nu_10, T_star)
    J_dust = planck_Bnu(nu_10, T_dust)
    J_bar = J_star + J_dust

    if T_gas > 30.0:
        k_10 = 1.5e-15 * T_gas**1.5 * np.exp(-3080.0 / max(T_gas, 50.0))
    else:
        k_10 = 1.0e-20

    pump_rate = B_01 * J_bar
    col_deexcite = k_10 * n_H2
    denom = pump_rate + A_10 + col_deexcite
    x1 = pump_rate / denom if denom > 0 else 0.0

    return n_CO * col_deexcite * x1 * h_nu


def heating_photoelectric(r, n_H2, tau_UV):
    """
    Grain photoelectric heating rate per unit volume [erg/cm^3/s].

    H_pe = epsilon * 4e-26 * n_H * G0 * exp(-tau_UV)
    """
    n_H = 2.0 * n_H2
    return epsilon_pe * 4.0e-26 * n_H * G0_ISRF * np.exp(-tau_UV)


def cooling_CO_line(n_CO, level_pops, J_bar, mol_data):
    """
    CO rotational line cooling rate per unit volume [erg/cm^3/s].

    Lambda_CO = n_CO * sum_transitions [(n_u*A - (n_l*B_lu - n_u*B_ul)*J_bar) * h*nu]
    """
    cooling = 0.0
    for t in range(mol_data.n_trans):
        u = mol_data.upper[t]
        l = mol_data.lower[t]
        nu = mol_data.nu[t]
        A_ul = mol_data.A[t]
        B_ul = mol_data.B_ul[t]
        B_lu = mol_data.B_lu[t]

        n_u = level_pops[u]
        n_l = level_pops[l]

        Jbar = J_bar[t] if J_bar is not None else 0.0
        net_rate = (n_u * A_ul - (n_l * B_lu - n_u * B_ul) * Jbar) * h_planck * nu
        cooling += net_rate

    return n_CO * cooling


def _dTdr_shell(r, T, i, env, level_pops_12, level_pops_13,
                J_bar_12, J_bar_13, mol_12, mol_13, co_cooling_frac):
    """
    Temperature gradient dT/dr at shell i.

    The energy equation for steady-state spherical wind:
    dT/dr = -(gamma-1)*T*(2/r + dvdr/v) + (H - L_CO) / (n*k*v)
    """
    T = max(T, T_CMB)

    v = env.v[i]
    dvdr = env.dvdr[i]
    n_H2 = env.n_H2[i]
    T_dust = env.T_dust[i]
    n_total = n_H2 * 1.2

    gam = gamma_H2(T)

    # Heating rates
    H_dg = heating_dustgas(r, env.Q_dg[i])
    H_IR = heating_IR_pumping(r, env.n_12CO[i], n_H2, T, T_dust)
    H_pe = heating_photoelectric(r, n_H2, env.tau_UV[i])
    H_total = H_dg + H_IR + H_pe

    # CO line cooling
    lp12 = level_pops_12[i] if level_pops_12 is not None else mol_12.LTE_populations(T)
    lp13 = level_pops_13[i] if level_pops_13 is not None else mol_13.LTE_populations(T)
    jb12 = J_bar_12[i] if J_bar_12 is not None else None
    jb13 = J_bar_13[i] if J_bar_13 is not None else None

    L_12CO = cooling_CO_line(env.n_12CO[i], lp12, jb12, mol_12)
    L_13CO = cooling_CO_line(env.n_13CO[i], lp13, jb13, mol_13)
    L_CO_raw = max(L_12CO + L_13CO, 0.0)

    # Limit CO cooling: in the LVG approximation, CO cooling is overestimated
    # in the inner envelope because the stellar/dust radiation field at mm
    # wavelengths is not included. Limit L_CO to a fraction of the adiabatic
    # cooling rate to prevent numerical instability.
    L_adiabatic = abs(n_total * k_boltz * T * (gam - 1.0) * (2.0 * v / r + dvdr))
    L_CO = min(L_CO_raw, co_cooling_frac * L_adiabatic)

    # Temperature gradient
    adiabatic_term = -(gam - 1.0) * T * (2.0 / r + dvdr / v)
    source_term = (H_total - L_CO) / (n_total * k_boltz * v)

    return adiabatic_term + source_term


def solve_temperature(env, level_pops_12, level_pops_13,
                      J_bar_12, J_bar_13, mol_12, mol_13,
                      T_inner=2000.0, damping=0.5, co_cooling_frac=0.3):
    """
    Integrate the temperature ODE outward from the inner boundary.

    Uses sub-stepped RK4 integration for stability.

    Parameters
    ----------
    env : Envelope
    level_pops_12, level_pops_13 : arrays or None
    J_bar_12, J_bar_13 : arrays or None
    mol_12, mol_13 : MolecularData
    T_inner : float
        Inner boundary temperature [K].
    damping : float
        Damping factor for updating T_gas.
    co_cooling_frac : float
        Maximum CO cooling as fraction of adiabatic cooling (default 0.3).
        Accounts for radiation field effects not captured by LVG.

    Returns
    -------
    T_new : array (n_shells,)
        New temperature profile.
    """
    n = env.n_shells
    T_new = np.zeros(n)
    T_new[0] = T_inner

    for i in range(n - 1):
        r_start = env.r[i]
        r_end = env.r[i + 1]
        T = T_new[i]

        # Sub-stepped RK4 integration between shells
        n_sub = 10
        dr = (r_end - r_start) / n_sub

        for _ in range(n_sub):
            r = r_start + _ * dr

            k1 = dr * _dTdr_shell(r, T, i, env,
                                  level_pops_12, level_pops_13,
                                  J_bar_12, J_bar_13,
                                  mol_12, mol_13, co_cooling_frac)
            k2 = dr * _dTdr_shell(r + 0.5 * dr, T + 0.5 * k1, i, env,
                                  level_pops_12, level_pops_13,
                                  J_bar_12, J_bar_13,
                                  mol_12, mol_13, co_cooling_frac)
            k3 = dr * _dTdr_shell(r + 0.5 * dr, T + 0.5 * k2, i, env,
                                  level_pops_12, level_pops_13,
                                  J_bar_12, J_bar_13,
                                  mol_12, mol_13, co_cooling_frac)
            k4 = dr * _dTdr_shell(r + dr, T + k3, i, env,
                                  level_pops_12, level_pops_13,
                                  J_bar_12, J_bar_13,
                                  mol_12, mol_13, co_cooling_frac)

            T = T + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            T = max(T, T_CMB)

        T_new[i + 1] = T

    # Clamp to physical range
    T_new = np.maximum(T_new, T_CMB)

    # Apply damping
    T_updated = (1.0 - damping) * env.T_gas + damping * T_new
    T_updated = np.maximum(T_updated, T_CMB)

    return T_updated
