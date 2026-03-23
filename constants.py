"""
Physical constants (CGS) and IRC+10216 envelope parameters.

References:
    Crosas & Menten 1997, ApJ 483, 913
    Crosas & Menten 1996, ASP Conf. Series 93, 98
"""
import numpy as np

# --------------- Physical constants (CGS) ---------------
c_light = 2.99792458e10       # speed of light [cm/s]
h_planck = 6.62607015e-27     # Planck constant [erg s]
k_boltz = 1.380649e-16        # Boltzmann constant [erg/K]
m_H = 1.6735575e-24           # hydrogen atom mass [g]
m_H2 = 2.0 * m_H              # H2 mass [g]
G_grav = 6.67430e-8           # gravitational constant [cm^3 g^-1 s^-2]
M_sun = 1.989e33              # solar mass [g]
L_sun = 3.828e33              # solar luminosity [erg/s]
au = 1.496e13                 # astronomical unit [cm]
pc = 3.0857e18                # parsec [cm]
yr = 3.1557e7                 # year [s]
sigma_SB = 5.6704e-5          # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
T_CMB = 2.725                 # CMB temperature [K]

# --------------- IRC+10216 parameters ---------------
D_pc = 150.0                  # distance [pc]
D = D_pc * pc                 # distance [cm]
M_dot_Msun_yr = 3.25e-5       # mass-loss rate [M_sun/yr]
M_dot = M_dot_Msun_yr * M_sun / yr  # mass-loss rate [g/s]
v_inf = 14.0e5                # terminal velocity [cm/s]
r0_arcsec = 0.133             # r0 in arcsec (dust condensation radius scale)
r0 = r0_arcsec * D * np.pi / (180.0 * 3600.0)  # r0 [cm] ~ 2.9e14 cm
v_turb = 1.0e5                # turbulent velocity [cm/s]
L_star = 1.0e4 * L_sun        # stellar luminosity [erg/s]
T_star = 2330.0               # effective temperature of star [K]
R_star = np.sqrt(L_star / (4.0 * np.pi * sigma_SB * T_star**4))  # stellar radius [cm]

# Dust parameters
r_dust_arcsec = 0.2           # dust inner radius [arcsec]
r_dust = r_dust_arcsec * D * np.pi / (180.0 * 3600.0)  # [cm] ~ 4.36e14 cm
T_dust_inner = 650.0          # dust temperature at inner radius [K]
dust_T_power = 0.4            # T_d ~ r^(-0.4)

# Dust-gas coupling efficiency
Q_inner = 2.5e-2              # r < 1e16 cm
Q_outer = 1.8e-2              # r > 1e16 cm
r_Q_break = 1.0e16            # transition radius [cm]

# CO abundances
X_CO = 6.0e-4                 # CO/H2 abundance ratio
ratio_12C_13C = 45.0          # 12C/13C isotope ratio
X_12CO = X_CO                 # 12CO/H2 (CO is dominantly 12CO)
X_13CO = X_CO / ratio_12C_13C  # 13CO/H2

# Envelope radial grid
r_inner = 2.0e15              # inner radius [cm]
r_outer = 3.0e17              # outer radius [cm]
n_shells = 60                 # number of radial shells

# Mean molecular weight (mostly H2 with ~20% He by number)
mu = 2.33                     # mean molecular weight [m_H]

# Photoelectric heating parameters
G0_ISRF = 1.0                 # Habing field scaling factor
epsilon_pe = 0.05             # photoelectric heating efficiency
sigma_UV = 2.0e-21            # UV dust cross section per H nucleus [cm^2]
