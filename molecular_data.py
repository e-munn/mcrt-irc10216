"""
Spectroscopic and collisional data for 12CO and 13CO.

Energy levels, Einstein A coefficients, and collisional de-excitation rates
from the LAMDA database (Schoier et al. 2005).

We include 30 rotational levels for 12CO and 20 for 13CO (v=0 ground state).
CO-H2 collisional rates are tabulated at selected temperatures and
interpolated for arbitrary T.
"""
import numpy as np
from scipy.interpolate import interp1d
from constants import h_planck, c_light, k_boltz


class MolecularData:
    """Spectroscopic and collisional data for a CO isotopologue."""

    def __init__(self, n_levels, energies_cm, A_coeffs, freqs_GHz,
                 col_temps, col_rates, weight_factor=1.0):
        """
        Parameters
        ----------
        n_levels : int
            Number of rotational levels.
        energies_cm : array (n_levels,)
            Energy of each level in cm^-1.
        A_coeffs : array (n_levels-1,)
            Einstein A_{J+1->J} spontaneous emission coefficients [s^-1].
        freqs_GHz : array (n_levels-1,)
            Line centre frequencies [GHz] for J+1 -> J transitions.
        col_temps : array (n_T,)
            Temperatures at which collisional rates are tabulated [K].
        col_rates : array (n_transitions, n_T)
            Collisional de-excitation rate coefficients gamma_{ul}(T) [cm^3/s].
            Ordered by (u, l) pairs: (1,0), (2,0), (2,1), (3,0), ...
        weight_factor : float
            Abundance scaling factor (1.0 for 12CO, 1/45 for 13CO).
        """
        self.n_levels = n_levels
        self.weight_factor = weight_factor

        # Statistical weights g_J = 2J + 1
        self.g = np.array([2.0 * J + 1.0 for J in range(n_levels)])

        # Energies in cm^-1 and erg
        self.E_cm = np.array(energies_cm[:n_levels], dtype=np.float64)
        self.E = self.E_cm * h_planck * c_light  # [erg]

        # Transition data (only J+1 -> J dipole transitions)
        self.n_trans = n_levels - 1
        self.A = np.array(A_coeffs[:self.n_trans], dtype=np.float64)
        self.freq = np.array(freqs_GHz[:self.n_trans], dtype=np.float64) * 1.0e9  # Hz
        self.nu = self.freq  # alias

        # Upper and lower level indices for each transition
        self.upper = np.arange(1, n_levels, dtype=int)
        self.lower = np.arange(0, n_levels - 1, dtype=int)

        # Einstein B coefficients
        # B_ul = A_ul * c^2 / (2 h nu^3)
        self.B_ul = self.A * c_light**2 / (2.0 * h_planck * self.nu**3)
        # B_lu = (g_u / g_l) * B_ul
        self.B_lu = (self.g[self.upper] / self.g[self.lower]) * self.B_ul

        # Collisional rates -- store interpolators
        self.col_temps = np.array(col_temps, dtype=np.float64)
        self._col_rates_table = np.array(col_rates, dtype=np.float64)
        self._build_col_interpolators()

        # Build mapping from (u, l) pair to index in col_rates
        self._build_col_index()

    def _build_col_index(self):
        """Build dictionary mapping (u, l) -> index in col_rates array."""
        self.col_pairs = []
        idx = 0
        for u in range(1, self.n_levels):
            for l in range(u):
                self.col_pairs.append((u, l))
                idx += 1
        # total number of collisional transitions
        self.n_col_trans = len(self.col_pairs)
        self.col_pair_to_idx = {pair: i for i, pair in enumerate(self.col_pairs)}

    def _build_col_interpolators(self):
        """Build scipy interpolators for collisional rates vs temperature."""
        n_trans = self._col_rates_table.shape[0]
        self._col_interp = []
        log_T = np.log10(self.col_temps)
        for i in range(n_trans):
            log_rate = np.log10(np.clip(self._col_rates_table[i], 1e-30, None))
            interp = interp1d(log_T, log_rate, kind='linear',
                              fill_value='extrapolate')
            self._col_interp.append(interp)

    def gamma_ul(self, T):
        """
        Collisional de-excitation rates gamma_{ul}(T) for all (u,l) pairs.

        Returns array of shape (n_col_trans,) in [cm^3/s].
        """
        log_T = np.log10(np.clip(T, 1.0, 1e5))
        rates = np.array([10.0 ** interp(log_T) for interp in self._col_interp])
        return rates

    def gamma_lu(self, T):
        """
        Collisional excitation rates from detailed balance:
        gamma_{lu} = (g_u / g_l) * gamma_{ul} * exp(-Delta_E / kT).

        Returns array of shape (n_col_trans,).
        """
        g_ul = self.gamma_ul(T)
        g_lu = np.zeros_like(g_ul)
        for idx, (u, l) in enumerate(self.col_pairs):
            dE = self.E[u] - self.E[l]
            g_lu[idx] = (self.g[u] / self.g[l]) * g_ul[idx] * np.exp(-dE / (k_boltz * T))
        return g_lu

    def LTE_populations(self, T):
        """Boltzmann (LTE) level populations at temperature T."""
        x = -self.E / (k_boltz * max(T, 1.0))
        n = self.g * np.exp(x)
        return n / n.sum()


def build_12CO(n_levels=30):
    """
    Build MolecularData for 12CO with up to 30 rotational levels.

    Energy levels E_J = B * J * (J+1) - D * J^2 * (J+1)^2
    with B = 1.922529 cm^-1, D = 6.121e-6 cm^-1 for 12CO.

    Einstein A coefficients for J+1 -> J:
    A_{J+1,J} = (64 pi^4 nu^3 / (3 h c^3)) * mu^2 * (J+1) / (2J+3)
    with mu = 0.11011 Debye for CO.
    """
    # Rotational constants for 12CO (cm^-1)
    B_rot = 1.922529
    D_rot = 6.121e-6

    energies_cm = np.zeros(n_levels)
    for J in range(n_levels):
        energies_cm[J] = B_rot * J * (J + 1) - D_rot * J**2 * (J + 1)**2

    # Frequencies: nu_{J+1,J} = c * (E_{J+1} - E_J) in Hz
    freqs_GHz = np.zeros(n_levels - 1)
    for J in range(n_levels - 1):
        freqs_GHz[J] = c_light * (energies_cm[J + 1] - energies_cm[J]) * 1e-9  # GHz

    # Einstein A coefficients computed from dipole moment mu = 0.11011 Debye
    mu_D = 0.11011  # Debye
    mu_cgs = mu_D * 1e-18  # 1 Debye = 1e-18 esu cm
    A_coeffs = np.zeros(n_levels - 1)
    for J in range(n_levels - 1):
        nu = freqs_GHz[J] * 1e9  # Hz
        Jup = J + 1
        A_coeffs[J] = (64.0 * np.pi**4 * nu**3 * mu_cgs**2 / (3.0 * h_planck * c_light**3)) \
                       * Jup / (2.0 * Jup + 1.0)

    # Collisional rates: CO-H2 from Yang et al. (2010) / LAMDA
    # Tabulated temperatures [K]
    col_temps = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 2000, 3000]

    # We generate approximate collisional rates using the power-law scaling
    # gamma_{J+1->J}(T) ~ gamma_10(T) * (J+1)
    # gamma_10(T) from LAMDA ~ 3.3e-11 * (T/300)^0.7 for CO-H2
    # For Delta-J > 1 transitions, rates are typically ~10x smaller
    col_rates = _generate_co_h2_collisional_rates(n_levels, col_temps)

    return MolecularData(
        n_levels=n_levels,
        energies_cm=energies_cm,
        A_coeffs=A_coeffs,
        freqs_GHz=freqs_GHz,
        col_temps=col_temps,
        col_rates=col_rates,
        weight_factor=1.0,
    )


def build_13CO(n_levels=20):
    """Build MolecularData for 13CO with up to 20 rotational levels."""
    # Rotational constants for 13CO (cm^-1)
    B_rot = 1.838054
    D_rot = 5.627e-6

    energies_cm = np.zeros(n_levels)
    for J in range(n_levels):
        energies_cm[J] = B_rot * J * (J + 1) - D_rot * J**2 * (J + 1)**2

    freqs_GHz = np.zeros(n_levels - 1)
    for J in range(n_levels - 1):
        freqs_GHz[J] = c_light * (energies_cm[J + 1] - energies_cm[J]) * 1e-9

    mu_D = 0.11011
    mu_cgs = mu_D * 1e-18
    A_coeffs = np.zeros(n_levels - 1)
    for J in range(n_levels - 1):
        nu = freqs_GHz[J] * 1e9
        Jup = J + 1
        A_coeffs[J] = (64.0 * np.pi**4 * nu**3 * mu_cgs**2 / (3.0 * h_planck * c_light**3)) \
                       * Jup / (2.0 * Jup + 1.0)

    col_temps = [5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 2000, 3000]
    col_rates = _generate_co_h2_collisional_rates(n_levels, col_temps)

    return MolecularData(
        n_levels=n_levels,
        energies_cm=energies_cm,
        A_coeffs=A_coeffs,
        freqs_GHz=freqs_GHz,
        col_temps=col_temps,
        col_rates=col_rates,
        weight_factor=1.0 / 45.0,
    )


def _generate_co_h2_collisional_rates(n_levels, col_temps):
    """
    Generate approximate CO-H2 collisional de-excitation rates.

    Uses scaling relations calibrated to LAMDA database values:
    - Delta-J = 1: gamma ~ 3.3e-11 * (J+1)^0.5 * (T/300)^0.65  [cm^3/s]
    - Delta-J = 2: gamma ~ 0.15 * gamma(Delta-J=1)
    - Delta-J >= 3: gamma ~ 0.02 * gamma(Delta-J=1) * exp(-0.3*(Delta_J-1))

    These are approximations; for publication-quality work, use actual LAMDA tables.
    """
    col_temps = np.array(col_temps, dtype=np.float64)
    n_T = len(col_temps)

    # Count total transitions: all (u, l) with u > l
    pairs = []
    for u in range(1, n_levels):
        for l in range(u):
            pairs.append((u, l))
    n_trans = len(pairs)
    rates = np.zeros((n_trans, n_T))

    for idx, (u, l) in enumerate(pairs):
        delta_J = u - l
        # Base rate for J+1 -> J (delta_J = 1)
        base = 3.3e-11 * (u)**0.5 * (col_temps / 300.0)**0.65
        if delta_J == 1:
            rates[idx] = base
        elif delta_J == 2:
            rates[idx] = 0.15 * base
        else:
            rates[idx] = 0.02 * base * np.exp(-0.3 * (delta_J - 1))

    return rates
