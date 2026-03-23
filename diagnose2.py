#!/usr/bin/env python3
"""Check escape probabilities and optical depths at inner shell."""
import numpy as np
from constants import m_H, h_planck, k_boltz, c_light
from envelope import Envelope
from molecular_data import build_12CO
from radiative_transfer import sobolev_tau, escape_probability

env = Envelope(n_shells=60)
mol = build_12CO(n_levels=30)

i = 0  # inner shell
T = env.T_gas[i]
n_CO = env.n_12CO[i]
dvdr = env.dvdr[i]

# LTE populations at T=2000 K
pops = mol.LTE_populations(T)

print(f"r = {env.r[i]:.2e} cm, T = {T:.0f} K, n_CO = {n_CO:.1f} cm^-3")
print(f"|dv/dr| = {dvdr:.3e} s^-1")
print(f"\n{'J+1->J':>8} {'nu[GHz]':>10} {'A[s-1]':>10} {'n_l':>8} {'n_u':>8} "
      f"{'tau':>10} {'beta':>8} {'beta*A*n_u*hnu':>16}")

total_cooling = 0.0
for t in range(min(mol.n_trans, 25)):
    u = mol.upper[t]
    l = mol.lower[t]
    nu = mol.nu[t]
    A = mol.A[t]
    n_l = pops[l] * n_CO
    n_u = pops[u] * n_CO

    tau = sobolev_tau(n_l, n_u, mol.g[l], mol.g[u], A, nu, dvdr)
    beta = escape_probability(tau)
    contrib = beta * n_u * A * h_planck * nu

    total_cooling += contrib
    print(f"{u:3d}->{l:<3d} {nu/1e9:10.3f} {A:10.3e} {n_l:8.3f} {n_u:8.3f} "
          f"{tau:10.2f} {beta:8.5f} {contrib:16.3e}")

print(f"\nTotal LVG cooling (approx): {total_cooling:.3e} erg/cm^3/s")
print(f"n_CO * total: already included above")
