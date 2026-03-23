#!/usr/bin/env python3
"""Compare cooling formula with escape-probability estimate."""
import numpy as np
from constants import m_H, h_planck, k_boltz, c_light, T_CMB
from envelope import Envelope
from molecular_data import build_12CO
from radiative_transfer import sobolev_tau, escape_probability, planck_Bnu, solve_statistical_equilibrium

env = Envelope(n_shells=60)
mol = build_12CO(n_levels=30)

i = 0
T = env.T_gas[i]
n_CO = env.n_12CO[i]
n_H2 = env.n_H2[i]
dvdr = env.dvdr[i]
T_dust = env.T_dust[i]

# Solve SE
pops, Jbar, conv = solve_statistical_equilibrium(n_CO, n_H2, T, T_dust, dvdr, mol)
lte = mol.LTE_populations(T)

print(f"SE converged: {conv}")
print(f"\n{'J':>3} {'x_LTE':>10} {'x_SE':>10} {'ratio':>8}")
for j in range(min(20, mol.n_levels)):
    r = pops[j] / (lte[j] + 1e-30)
    print(f"{j:3d} {lte[j]:10.5f} {pops[j]:10.5f} {r:8.3f}")

# Compute cooling two ways
print("\nCooling comparison per transition:")
print(f"{'t':>3} {'J':>3} {'beta':>8} {'Jbar':>12} {'S':>12} "
      f"{'full_formula':>14} {'beta*nA*hnu':>14}")

cool_full = 0.0
cool_beta = 0.0
for t in range(min(20, mol.n_trans)):
    u = mol.upper[t]
    l = mol.lower[t]
    nu = mol.nu[t]
    A = mol.A[t]
    B_ul = mol.B_ul[t]
    B_lu = mol.B_lu[t]

    x_u = pops[u]
    x_l = pops[l]

    # Sobolev tau with SE pops
    tau = sobolev_tau(x_l * n_CO, x_u * n_CO, mol.g[l], mol.g[u], A, nu, dvdr)
    beta = escape_probability(tau)

    # Source function
    ratio = (x_l * mol.g[u]) / (x_u * mol.g[l]) if x_u > 1e-30 else 1e30
    if ratio > 1:
        S = (2 * h_planck * nu**3 / c_light**2) / (ratio - 1)
    else:
        S = 0.0

    # Full formula cooling
    full = n_CO * (x_u * A - (x_l * B_lu - x_u * B_ul) * Jbar[t]) * h_planck * nu

    # Escape probability cooling
    J_ext = planck_Bnu(nu, T_CMB)
    beta_cool = n_CO * beta * (x_u * A + (x_u * B_ul - x_l * B_lu) * J_ext) * h_planck * nu

    cool_full += full
    cool_beta += beta_cool

    print(f"{t:3d} {u:3d} {beta:8.4f} {Jbar[t]:12.4e} {S:12.4e} "
          f"{full:14.4e} {beta_cool:14.4e}")

print(f"\nTotal full formula: {cool_full:.4e}")
print(f"Total beta estimate: {cool_beta:.4e}")
