#!/usr/bin/env python3
"""Diagnostic script to check heating/cooling balance at inner shells."""
import numpy as np
from constants import m_H, h_planck, k_boltz, T_CMB
from envelope import Envelope
from molecular_data import build_12CO, build_13CO
from radiative_transfer import solve_statistical_equilibrium
from thermal_balance import (
    heating_dustgas, heating_IR_pumping, heating_photoelectric,
    cooling_CO_line, gamma_H2
)

env = Envelope(n_shells=60)
mol_12 = build_12CO(n_levels=30)
mol_13 = build_13CO(n_levels=20)

print("Shell diagnostics (inner 10 shells):")
print(f"{'i':>3} {'r[cm]':>10} {'T_gas':>8} {'T_dust':>7} {'n_H2':>10} "
      f"{'H_dg':>10} {'H_IR':>10} {'H_pe':>10} {'L_ad':>10} {'L_CO12':>10} "
      f"{'dTdr':>10}")

for i in range(min(15, env.n_shells)):
    r = env.r[i]
    T = env.T_gas[i]
    v = env.v[i]
    dvdr = env.dvdr[i]
    n_H2 = env.n_H2[i]
    T_dust = env.T_dust[i]

    # Solve LVG for level pops
    pops, Jbar, _ = solve_statistical_equilibrium(
        n_CO=env.n_12CO[i], n_H2=n_H2, T_gas=T,
        T_dust=T_dust, dvdr=dvdr, mol_data=mol_12
    )

    # Heating
    H_dg = heating_dustgas(r, env.Q_dg[i])
    H_IR = heating_IR_pumping(r, env.n_12CO[i], n_H2, T, T_dust)
    H_pe = heating_photoelectric(r, n_H2, env.tau_UV[i])

    # Adiabatic cooling
    gam = gamma_H2(T)
    n_total = n_H2 * 1.2
    div_v = 2.0 * v / r + dvdr
    L_ad = n_total * k_boltz * T * (gam - 1.0) * div_v

    # CO cooling
    L_CO = cooling_CO_line(env.n_12CO[i], pops, Jbar, mol_12)

    # dT/dr
    H_total = H_dg + H_IR + H_pe
    dTdr = (-(gam - 1.0) * T * (2.0/r + dvdr/v) + (H_total - L_CO) / (n_total * k_boltz * v))

    print(f"{i:3d} {r:10.2e} {T:8.1f} {T_dust:7.1f} {n_H2:10.2e} "
          f"{H_dg:10.2e} {H_IR:10.2e} {H_pe:10.2e} {L_ad:10.2e} {L_CO:10.2e} "
          f"{dTdr:10.2e}")
