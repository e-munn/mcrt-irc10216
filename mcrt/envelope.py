"""
Envelope model for IRC+10216.

Provides the radial grid and precomputed physical quantities:
velocity, density, dust temperature, initial gas temperature guess.
"""
import numpy as np
from mcrt.constants import (
    r_inner, r_outer, n_shells, M_dot, v_inf, r0, v_turb,
    m_H, mu, X_12CO, X_13CO, T_dust_inner, r_dust, dust_T_power,
    Q_inner, Q_outer, r_Q_break, T_CMB, k_boltz, c_light,
    L_star, sigma_UV, G0_ISRF
)


class Envelope:
    """Spherically symmetric circumstellar envelope."""

    def __init__(self, n_shells=n_shells, r_min=r_inner, r_max=r_outer):
        self.n_shells = n_shells
        # Logarithmic radial grid -- cell centres
        self.r = np.logspace(np.log10(r_min), np.log10(r_max), n_shells)
        # Cell boundaries (half-way in log space)
        log_r = np.log10(self.r)
        d_log_r = np.diff(log_r)
        self.r_edges = np.zeros(n_shells + 1)
        self.r_edges[0] = 10.0 ** (log_r[0] - 0.5 * d_log_r[0])
        self.r_edges[-1] = 10.0 ** (log_r[-1] + 0.5 * d_log_r[-1])
        for i in range(1, n_shells):
            self.r_edges[i] = 10.0 ** (0.5 * (log_r[i - 1] + log_r[i]))
        self.dr = np.diff(self.r_edges)

        # Velocity field: v(r) = v_inf * sqrt(1 - r0/r)
        ratio = np.clip(r0 / self.r, 0.0, 0.999)
        self.v = v_inf * np.sqrt(1.0 - ratio)

        # Velocity gradient dv/dr
        self.dvdr = v_inf * 0.5 * (r0 / self.r**2) / np.sqrt(1.0 - ratio)

        # Effective line width (thermal + turbulent, placeholder, updated per T)
        self.delta_v = np.full(n_shells, v_turb)

        # H2 number density from mass continuity: n_H2 = M_dot / (4*pi*r^2*v*mu*m_H)
        self.n_H2 = M_dot / (4.0 * np.pi * self.r**2 * self.v * mu * m_H)

        # CO number densities
        self.n_12CO = X_12CO * self.n_H2
        self.n_13CO = X_13CO * self.n_H2

        # Dust temperature: T_d(r) = T_dust_inner * (r / r_dust)^(-dust_T_power)
        self.T_dust = np.where(
            self.r >= r_dust,
            T_dust_inner * (self.r / r_dust) ** (-dust_T_power),
            T_dust_inner
        )

        # Dust-gas coupling coefficient Q(r)
        self.Q_dg = np.where(self.r < r_Q_break, Q_inner, Q_outer)

        # UV optical depth (integrated inward from outer boundary)
        # tau_UV(r) = integral from r to r_outer of sigma_UV * n_H * dr
        # n_H ~ 2 * n_H2 (each H2 provides 2 H nuclei)
        self.tau_UV = self._compute_tau_UV()

        # Initial gas temperature guess: power-law, clamped to T_CMB minimum
        self.T_gas = np.maximum(
            2000.0 * (self.r / r_inner) ** (-0.8),
            T_CMB
        )

    def _compute_tau_UV(self):
        """Compute UV optical depth from outer boundary inward."""
        n_H = 2.0 * self.n_H2  # total H nuclei
        tau = np.zeros(self.n_shells)
        # Integrate from outside in
        tau[-1] = 0.0
        for i in range(self.n_shells - 2, -1, -1):
            dr = self.r[i + 1] - self.r[i]
            tau[i] = tau[i + 1] + sigma_UV * 0.5 * (n_H[i] + n_H[i + 1]) * dr
        return tau

    def update_delta_v(self, T_gas, m_mol=28.0 * m_H):
        """Update thermal line width: delta_v = sqrt(v_turb^2 + 2kT/m)."""
        v_th = np.sqrt(2.0 * k_boltz * T_gas / m_mol)
        self.delta_v = np.sqrt(v_turb**2 + v_th**2)

    def dust_emission_Bnu(self, nu):
        """Planck function B_nu(T_dust) at frequency nu for each shell [erg/s/cm^2/Hz/sr]."""
        x = np.clip(h_planck * nu / (k_boltz * self.T_dust), 0.0, 500.0)
        return (2.0 * h_planck * nu**3 / c_light**2) / (np.exp(x) - 1.0)

    def cmb_Bnu(self, nu):
        """Planck function B_nu(T_CMB) at frequency nu [erg/s/cm^2/Hz/sr]."""
        x = np.clip(h_planck * nu / (k_boltz * T_CMB), 0.0, 500.0)
        return (2.0 * h_planck * nu**3 / c_light**2) / (np.exp(x) - 1.0)


# Import h_planck at module level for use in methods
from mcrt.constants import h_planck
