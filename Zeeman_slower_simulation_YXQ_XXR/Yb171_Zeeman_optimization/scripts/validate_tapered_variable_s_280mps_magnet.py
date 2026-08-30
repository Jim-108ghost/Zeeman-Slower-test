# -*- coding: utf-8 -*-
"""Validate the run-11 field, magnet coordinates and on-axis slowing."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


def beam_radius_m(z_m):
    z_clipped = np.clip(np.asarray(z_m), -0.300, 0.320)
    return 4e-3 + 4e-3 * (z_clipped + 0.300) / 0.620


def simulate(B_func, initial_velocity_m_s):
    pi = np.pi
    hbar = 1.054571817e-34
    h = 2.0 * pi * hbar
    c = 299792458.0
    elementary_charge = 1.602176634e-19
    electron_mass = 9.1093837139e-31
    mass = 171.0 * 1.66053906660e-27
    gamma = 2.0 * pi * 28.9e6
    wavelength = 398.9e-9
    k = 2.0 * pi / wavelength
    mu_eff = -elementary_charge * hbar / (2.0 * electron_mass)
    delta_laser = 2.0 * pi * (-600e6)
    laser_power_W = 35e-3
    slower_length_m = 0.185
    saturation_intensity = pi * h * c * gamma / (3.0 * wavelength**3)

    def equation(_time, state):
        z_m, velocity_m_s = state
        radius_m = beam_radius_m(z_m)
        s_local = 2.0 * laser_power_W / (pi * radius_m**2) / saturation_intensity
        detuning = (
            delta_laser
            + k * velocity_m_s
            - mu_eff * B_func(z_m) / hbar
        )
        scattering_rate = (
            gamma
            / 2.0
            * s_local
            / (1.0 + s_local + (2.0 * detuning / gamma) ** 2)
        )
        return [velocity_m_s, -hbar * k * scattering_rate / mass]

    def at_exit(_time, state):
        return state[0] - slower_length_m

    at_exit.terminal = True
    at_exit.direction = 1
    solution = solve_ivp(
        equation,
        (0.0, 0.02),
        (0.0, initial_velocity_m_s),
        events=at_exit,
        rtol=5e-9,
        atol=1e-10,
        max_step=2e-6,
    )
    return {
        "initial_velocity_m_s": initial_velocity_m_s,
        "final_position_mm": solution.y[0, -1] * 1e3,
        "final_velocity_m_s": solution.y[1, -1],
        "travel_time_ms": solution.t[-1] * 1e3,
        "exited_185mm": bool(solution.t_events[0].size),
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "runs" / "11_tapered_variable_s_280mps_185mm"
    target = pd.read_csv(
        project_root / "inputs" / "Yb171_tapered_variable_s_280mps_B_vs_z.csv"
    )
    field = pd.read_csv(
        run_dir / "Yb171_tapered_variable_s_280mps_185mm_field.csv"
    )
    config = pd.read_csv(
        run_dir / "Yb171_tapered_variable_s_280mps_185mm_magnet_config.csv"
    )
    individual = pd.read_csv(
        run_dir / "Yb171_tapered_variable_s_280mps_185mm_individual_magnets.csv"
    )
    extended = pd.read_csv(
        run_dir / "Yb171_tapered_variable_s_280mps_185mm_field_to_400mm.csv"
    )

    expected_x_mm = np.array(
        [
            9.467, 22.400, 35.333, 48.267, 61.200,
            74.133, 87.067, 100.000, 112.933, 125.867,
            138.800, 151.733, 164.667, 177.600, 190.533,
        ]
    )
    total_magnets = int(config["total_magnets_pair"].sum())
    checks = {
        "target_has_2001_rows": len(target) == 2001,
        "target_is_0_to_185mm": np.isclose(target.z_m.iloc[0], 0.0)
        and np.isclose(target.z_m.iloc[-1], 0.185),
        "fifteen_fixed_longitudinal_positions": len(config) == 15
        and np.allclose(config.x_mm, expected_x_mm, atol=1e-9),
        "total_magnets_is_200": total_magnets == 200,
        "individual_file_has_200_rows": len(individual) == 200,
        "first_four_stacks_negative": np.all(config.orientation.iloc[:4] == -1),
        "remaining_stacks_positive": np.all(config.orientation.iloc[4:] == 1),
        "extended_field_is_minus20_to_400mm": np.isclose(extended.z_mm.iloc[0], -20.0)
        and np.isclose(extended.z_mm.iloc[-1], 400.0),
    }

    # Notebook/EXP DATA compatibility file: use the exact z_m/B_T schema that
    # the simulation cell expects, while retaining the variable-s metadata.
    active_field = field.loc[
        (field.z_mm >= 0.0) & (field.z_mm <= 185.0)
    ].copy()
    exp_compatible = pd.DataFrame(
        {
            "z_m": active_field.z_mm.to_numpy(dtype=float) * 1e-3,
            "z_cm": active_field.z_mm.to_numpy(dtype=float) * 1e-1,
            "B_T": active_field.B_sim_mT.to_numpy(dtype=float) * 1e-3,
            "B_mT": active_field.B_sim_mT.to_numpy(dtype=float),
        }
    )
    metadata_columns = [
        "beam_radius_mm",
        "s_local",
        "eta_available",
        "force_scale",
        "force_margin_fraction",
        "eta_design",
        "v_design_m_s",
        "delta_eff_MHz",
    ]
    for column in metadata_columns:
        exp_compatible[column] = np.interp(
            exp_compatible.z_m,
            target.z_m,
            target[column],
        )
    exp_compatible_path = (
        project_root
        / "inputs"
        / "Yb171_tapered_variable_s_280mps_magnet_B_vs_z.csv"
    )
    exp_compatible.to_csv(exp_compatible_path, index=False)
    checks["exp_compatible_field_is_0_to_185mm"] = (
        np.isclose(exp_compatible.z_m.iloc[0], 0.0)
        and np.isclose(exp_compatible.z_m.iloc[-1], 0.185)
        and np.all(np.diff(exp_compatible.z_m) > 0.0)
    )

    target_B = PchipInterpolator(target.z_m, target.B_T)
    magnet_B = PchipInterpolator(field.z_mm * 1e-3, field.B_sim_mT * 1e-3)
    rows = []
    for field_name, B_func in (("designed_target", target_B), ("small_magnets", magnet_B)):
        for initial_velocity in (280.0, 285.0, 286.0):
            rows.append(
                {
                    "field_source": field_name,
                    **simulate(B_func, initial_velocity),
                }
            )
    validation = pd.DataFrame(rows)
    validation_path = (
        run_dir / "Yb171_tapered_variable_s_280mps_185mm_trajectory_validation.csv"
    )
    validation.to_csv(validation_path, index=False)

    magnet_280_final = validation.loc[
        (validation.field_source == "small_magnets")
        & np.isclose(validation.initial_velocity_m_s, 280.0),
        "final_velocity_m_s",
    ].iloc[0]
    checks["280mps_atom_exits_below_50mps"] = 0.0 < magnet_280_final <= 50.0
    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{name}: {passed}")
    print(validation.to_string(index=False))
    print(validation_path)
    print(exp_compatible_path)
    if failed:
        raise RuntimeError("Validation failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
