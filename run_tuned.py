#!/usr/bin/env python3
"""
Tuned run: adjust parameters to better match Crosas & Menten Figure 1.

Changes from default:
- co_cooling_frac: 0.3 -> 0.15 (flatter inner slope)
- G0_ISRF: 1.0 -> 1.7 (stronger ISRF for outer envelope heating)
- epsilon_pe: 0.05 -> 0.10 (higher PE efficiency -- typical for small grains)
- sigma_UV: 2e-21 -> 1.5e-21 (slightly less UV shielding -> more PE heating)
"""
import numpy as np
import constants
from envelope import Envelope
from molecular_data import build_12CO, build_13CO
from radiative_transfer import solve_all_shells
from thermal_balance import solve_temperature
from plotting import plot_temperature_profile, plot_convergence

# Override constants for tuning
constants.G0_ISRF = 1.7
constants.epsilon_pe = 0.10
constants.sigma_UV = 1.5e-21

# Re-import after override so thermal_balance picks up the changes
import importlib
import thermal_balance
importlib.reload(thermal_balance)
from thermal_balance import solve_temperature

def run_tuned():
    n_shells = 60
    env = Envelope(n_shells=n_shells)

    mol_12 = build_12CO(n_levels=30)
    mol_13 = build_13CO(n_levels=20)

    print("=" * 60)
    print("IRC+10216 LVG -- TUNED parameters")
    print("=" * 60)
    print(f"G0 = {constants.G0_ISRF}, epsilon_pe = {constants.epsilon_pe}, "
          f"sigma_UV = {constants.sigma_UV:.1e}")
    print(f"co_cooling_frac = 0.15")
    print("-" * 60)

    from constants import m_H
    convergence_history = []

    for iteration in range(50):
        damping = min(0.3 + 0.02 * iteration, 0.8)

        env.update_delta_v(env.T_gas, m_mol=28.0 * m_H)

        pops_12, Jbar_12 = solve_all_shells(env, mol_12, isotope='12CO')
        pops_13, Jbar_13 = solve_all_shells(env, mol_13, isotope='13CO')

        T_new = solve_temperature(
            env, pops_12, pops_13, Jbar_12, Jbar_13,
            mol_12, mol_13, T_inner=2000.0, damping=damping,
            co_cooling_frac=0.15
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            frac_change = np.abs(T_new - env.T_gas) / env.T_gas
        max_change = np.nanmax(frac_change)
        convergence_history.append(max_change)

        print(f"Iter {iteration + 1:3d}: max dT/T = {max_change:.4f}, "
              f"T_inner = {T_new[0]:.1f} K, T_outer = {T_new[-1]:.2f} K")

        env.T_gas = T_new

        if max_change < 0.005:
            print(f"\nConverged after {iteration + 1} iterations!")
            break

    print("-" * 60)
    print("Final temperature profile (tuned):")
    indices = np.linspace(0, n_shells - 1, 12, dtype=int)
    print(f"  {'r [cm]':>12s}  {'T_gas [K]':>10s}  {'T_dust [K]':>10s}")
    for i in indices:
        print(f"  {env.r[i]:12.3e}  {env.T_gas[i]:10.2f}  {env.T_dust[i]:10.2f}")

    # Save tuned plot
    plot_temperature_profile(env.r, env.T_gas, T_dust=env.T_dust,
                             filename='figure1_tuned.pdf',
                             title='IRC+10216 $T_K(r)$ -- Tuned LVG Model')

    np.savetxt('temperature_profile_tuned.dat',
               np.column_stack([env.r, env.T_gas, env.T_dust, env.n_H2]),
               header='r[cm]  T_gas[K]  T_dust[K]  n_H2[cm^-3]',
               fmt='%14.6e')
    print("\nSaved tuned data to temperature_profile_tuned.dat")

    return env


if __name__ == '__main__':
    run_tuned()
