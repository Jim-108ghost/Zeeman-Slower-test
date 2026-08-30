# -*- coding: utf-8 -*-
"""Generate the 245 -> 35 m/s Yb-171 field for tapered average intensity."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


def trapezoid_beam_radius(
    x,
    x_left=-0.300,
    r_left=4e-3,
    x_right=0.320,
    r_right=8e-3,
):
    x_array = np.asarray(x, dtype=float)
    x_clipped = np.clip(x_array, x_left, x_right)
    radius = r_left + (r_right - r_left) * (
        (x_clipped - x_left) / (x_right - x_left)
    )
    return float(radius) if x_array.ndim == 0 else radius


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_csv = (
        project_root / "inputs" / "Yb171_tapered_average_s_245mps_B_vs_z.csv"
    )
    output_plot = (
        project_root / "inputs" / "Yb171_tapered_average_s_245mps_design.png"
    )

    hbar = 1.054571817e-34
    h = 2.0 * np.pi * hbar
    c = 299792458.0
    elementary_charge = 1.602176634e-19
    electron_mass = 9.1093837139e-31
    mass_171 = 171.0 * 1.66053906660e-27
    gamma_399 = 2.0 * np.pi * 28.9e6
    wavelength_399 = 398.9e-9
    k_399 = 2.0 * np.pi / wavelength_399
    mu_eff = -elementary_charge * hbar / (2.0 * electron_mass)

    slower_length_m = 0.185
    initial_velocity_m_s = 245.0
    final_velocity_m_s = 35.0
    laser_detuning_rad_s = 2.0 * np.pi * (-600.0e6)
    laser_power_W = 35e-3

    z_m = np.linspace(0.0, slower_length_m, 2001)
    radius_m = trapezoid_beam_radius(z_m)
    saturation_intensity = (
        np.pi * h * c * gamma_399 / (3.0 * wavelength_399**3)
    )
    intensity_axis_W_m2 = 2.0 * laser_power_W / (np.pi * radius_m**2)
    # Preserve the original transverse-average convention while allowing the
    # beam radius, and therefore the average saturation, to vary with x.
    s_average = 0.5 * intensity_axis_W_m2 / saturation_intensity
    eta_available = s_average / (1.0 + s_average)

    a_max_m_s2 = hbar * k_399 * gamma_399 / (2.0 * mass_171)
    available_integral_m = np.trapezoid(eta_available, z_m)
    force_scale = (
        initial_velocity_m_s**2 - final_velocity_m_s**2
    ) / (2.0 * a_max_m_s2 * available_integral_m)
    if not 0.0 < force_scale < 1.0:
        raise RuntimeError(
            "Requested velocities exceed the average-intensity force: "
            f"scale={force_scale:.9f}"
        )

    eta_design = force_scale * eta_available
    eta_integral = cumulative_trapezoid(eta_design, z_m, initial=0.0)
    velocity_m_s = np.sqrt(
        np.maximum(
            initial_velocity_m_s**2 - 2.0 * a_max_m_s2 * eta_integral,
            0.0,
        )
    )

    detuning_term_squared = np.maximum(
        s_average / eta_design - (1.0 + s_average),
        0.0,
    )
    delta_eff_rad_s = -0.5 * gamma_399 * np.sqrt(detuning_term_squared)
    B_T = (hbar / mu_eff) * (
        laser_detuning_rad_s + k_399 * velocity_m_s - delta_eff_rad_s
    )

    output = pd.DataFrame(
        {
            "z_m": z_m,
            "z_cm": z_m * 1e2,
            "B_T": B_T,
            "B_mT": B_T * 1e3,
            "beam_radius_mm": radius_m * 1e3,
            "s_average": s_average,
            "eta_available": eta_available,
            "force_scale": np.full_like(z_m, force_scale),
            "force_margin_fraction": np.full_like(z_m, 1.0 - force_scale),
            "eta_design": eta_design,
            "v_design_m_s": velocity_m_s,
            "delta_eff_MHz": delta_eff_rad_s / (2.0 * np.pi * 1e6),
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(z_m * 1e3, B_T * 1e3)
    axes[0].set_ylabel("B (mT)")
    axes[0].set_title("Yb-171 average-s design: 245 to 35 m/s")
    axes[1].plot(z_m * 1e3, velocity_m_s)
    axes[1].set_ylabel("v (m/s)")
    axes[2].plot(z_m * 1e3, s_average, label="average s(x)")
    axes[2].plot(z_m * 1e3, eta_available, label="available force fraction")
    axes[2].plot(z_m * 1e3, eta_design, label="designed force fraction")
    axes[2].set_xlabel("x (mm)")
    axes[2].set_ylabel("dimensionless")
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=200)
    plt.close(fig)

    print(f"force scale            : {force_scale:.9f}")
    print(f"force margin           : {(1.0-force_scale)*100:.6f} %")
    print(f"designed velocity      : {velocity_m_s[0]:.6f} -> {velocity_m_s[-1]:.6f} m/s")
    print(f"average s              : {s_average[0]:.6f} -> {s_average[-1]:.6f}")
    print(f"effective detuning     : {delta_eff_rad_s[0]/(2*np.pi*1e6):.6f} -> {delta_eff_rad_s[-1]/(2*np.pi*1e6):.6f} MHz")
    print(f"magnetic field         : {B_T[0]*1e3:.6f} -> {B_T[-1]*1e3:.6f} mT")
    print(output_csv)
    print(output_plot)


if __name__ == "__main__":
    main()
