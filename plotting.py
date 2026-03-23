"""
Plotting module for IRC+10216 temperature profile.

Reproduces Figure 1 from Crosas & Menten (1996):
log-log plot of T_K (K) vs radial offset (cm).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_temperature_profile(r, T_gas, T_dust=None, filename='figure1.pdf',
                             title='IRC+10216 Kinetic Temperature Profile'):
    """
    Create a log-log plot of T_K vs r, matching Figure 1 axes.

    Parameters
    ----------
    r : array
        Radial positions [cm].
    T_gas : array
        Gas kinetic temperature [K].
    T_dust : array or None
        Dust temperature [K] (plotted as dashed line if provided).
    filename : str
        Output filename.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.loglog(r, T_gas, 'k-', linewidth=2, label=r'$T_{\rm K}$ (gas)')

    if T_dust is not None:
        ax.loglog(r, T_dust, 'r--', linewidth=1.5, label=r'$T_{\rm dust}$')

    # Axes matching Figure 1
    ax.set_xlim(1e15, 1e18)
    ax.set_ylim(1.0, 3000.0)
    ax.set_xlabel(r'$r$ [cm]', fontsize=14)
    ax.set_ylabel(r'$T$ [K]', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {filename}")


def plot_convergence(iterations, max_delta_T, filename='convergence.pdf'):
    """Plot convergence history of the temperature iteration."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(iterations, max_delta_T, 'bo-', markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'max $|\Delta T / T|$', fontsize=12)
    ax.set_title('Temperature Convergence', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved convergence plot to {filename}")
