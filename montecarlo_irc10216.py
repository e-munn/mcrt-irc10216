#!/usr/bin/env python3
"""
Monte Carlo Radiative Transfer Model for IRC+10216.

Phase A: LVG (Sobolev escape probability) approximation.

Reproduces the solid-line kinetic temperature profile T_K(r) from
Figure 1 of Crosas & Menten (1996, ASP Conf. Series 93, 98).

The code self-consistently solves:
1. CO level populations via LVG statistical equilibrium
2. Thermal balance: dust-gas heating + photoelectric heating vs
   adiabatic cooling + CO line cooling

Usage:
    python montecarlo_irc10216.py
"""
import sys
import numpy as np

from mcrt.constants import T_CMB, m_H, r_inner
from mcrt.envelope import Envelope
from mcrt.molecular_data import build_12CO, build_13CO
from mcrt.radiative_transfer import solve_all_shells
from mcrt.thermal_balance import solve_temperature
from mcrt.plotting import plot_temperature_profile, plot_convergence


def run_model(max_iter=50, tol=0.01, damping_initial=0.3,
              n_shells=60, verbose=True):
    """
    Run the full LVG thermal balance model.

    Parameters
    ----------
    max_iter : int
        Maximum number of global iterations.
    tol : float
        Convergence tolerance on max fractional temperature change.
    damping_initial : float
        Initial damping factor for temperature updates.
    n_shells : int
        Number of radial shells.
    verbose : bool
        Print progress information.

    Returns
    -------
    env : Envelope
        Final envelope model with converged T_gas.
    """
    if verbose:
        print("=" * 60)
        print("IRC+10216 LVG Thermal Balance Model")
        print("=" * 60)

    # Initialise envelope
    env = Envelope(n_shells=n_shells)
    if verbose:
        print(f"Grid: {n_shells} shells, r = [{env.r[0]:.2e}, {env.r[-1]:.2e}] cm")
        print(f"Inner T_gas = {env.T_gas[0]:.1f} K, Outer T_gas = {env.T_gas[-1]:.1f} K")

    # Build molecular data
    mol_12 = build_12CO(n_levels=30)
    mol_13 = build_13CO(n_levels=20)
    if verbose:
        print(f"12CO: {mol_12.n_levels} levels, {mol_12.n_trans} transitions")
        print(f"13CO: {mol_13.n_levels} levels, {mol_13.n_trans} transitions")
        print("-" * 60)

    convergence_history = []

    for iteration in range(max_iter):
        # Adaptive damping: start cautious, increase as we converge
        damping = min(damping_initial + 0.02 * iteration, 0.8)

        # Update thermal line widths
        env.update_delta_v(env.T_gas, m_mol=28.0 * m_H)

        # Step 1: Solve LVG for level populations at current T(r)
        pops_12, Jbar_12 = solve_all_shells(env, mol_12, isotope='12CO')
        pops_13, Jbar_13 = solve_all_shells(env, mol_13, isotope='13CO')

        # Step 2: Solve thermal balance for new T(r)
        T_new = solve_temperature(
            env, pops_12, pops_13, Jbar_12, Jbar_13,
            mol_12, mol_13, T_inner=2000.0, damping=damping
        )

        # Check convergence
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_change = np.abs(T_new - env.T_gas) / env.T_gas
        max_change = np.nanmax(frac_change)
        convergence_history.append(max_change)

        if verbose:
            print(f"Iter {iteration + 1:3d}: max dT/T = {max_change:.4f}, "
                  f"T_inner = {T_new[0]:.1f} K, T_outer = {T_new[-1]:.2f} K, "
                  f"damping = {damping:.2f}")

        # Update temperature
        env.T_gas = T_new

        if max_change < tol:
            if verbose:
                print(f"\nConverged after {iteration + 1} iterations!")
            break
    else:
        if verbose:
            print(f"\nDid not converge after {max_iter} iterations "
                  f"(max dT/T = {max_change:.4f})")

    if verbose:
        print("-" * 60)
        print("Final temperature profile:")
        # Print a few representative points
        indices = np.linspace(0, n_shells - 1, 10, dtype=int)
        print(f"  {'r [cm]':>12s}  {'T_gas [K]':>10s}  {'T_dust [K]':>10s}  "
              f"{'n_H2 [cm^-3]':>14s}")
        for i in indices:
            print(f"  {env.r[i]:12.3e}  {env.T_gas[i]:10.2f}  {env.T_dust[i]:10.2f}  "
                  f"{env.n_H2[i]:14.3e}")

    return env, convergence_history


def main():
    """Main entry point."""
    env, conv_hist = run_model(max_iter=50, tol=0.005, damping_initial=0.3,
                               n_shells=60, verbose=True)

    # Plot results
    plot_temperature_profile(env.r, env.T_gas, T_dust=env.T_dust,
                             filename='results/figure1.pdf')

    if len(conv_hist) > 1:
        plot_convergence(range(1, len(conv_hist) + 1), conv_hist,
                         filename='results/convergence.pdf')

    # Also save data to text file
    np.savetxt('results/temperature_profile.dat',
               np.column_stack([env.r, env.T_gas, env.T_dust, env.n_H2]),
               header='r[cm]  T_gas[K]  T_dust[K]  n_H2[cm^-3]',
               fmt='%14.6e')
    print("\nSaved temperature data to temperature_profile.dat")

    return env


if __name__ == '__main__':
    main()
