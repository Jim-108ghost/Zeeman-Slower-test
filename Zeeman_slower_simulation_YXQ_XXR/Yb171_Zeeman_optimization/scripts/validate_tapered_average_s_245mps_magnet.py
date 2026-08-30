# -*- coding: utf-8 -*-
"""Validate the 245 m/s average-s target and its small-magnet realization."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


def beam_radius_m(z_m):
    z_clipped = np.clip(np.asarray(z_m), -0.300, 0.320)
    return 4e-3 + 4e-3 * (z_clipped + 0.300) / 0.620


def make_simulator(B_func):
    hbar = 1.054571817e-34
    h = 2.0 * np.pi * hbar
    c = 299792458.0
    elementary_charge = 1.602176634e-19
    electron_mass = 9.1093837139e-31
    mass = 171.0 * 1.66053906660e-27
    gamma = 2.0 * np.pi * 28.9e6
    wavelength = 398.9e-9
    k = 2.0 * np.pi / wavelength
    mu_eff = -elementary_charge * hbar / (2.0 * electron_mass)
    delta_laser = 2.0 * np.pi * (-600e6)
    laser_power_W = 35e-3
    slower_length_m = 0.185
    saturation_intensity = np.pi * h * c * gamma / (3.0 * wavelength**3)

    def simulate(initial_velocity_m_s):
        def equation(_time, state):
            z_m, velocity_m_s = state
            radius_m = beam_radius_m(z_m)
            intensity_axis = 2.0 * laser_power_W / (np.pi * radius_m**2)
            s_average = 0.5 * intensity_axis / saturation_intensity
            detuning = (
                delta_laser
                + k * velocity_m_s
                - mu_eff * B_func(z_m) / hbar
            )
            scattering_rate = (
                gamma
                / 2.0
                * s_average
                / (1.0 + s_average + (2.0 * detuning / gamma) ** 2)
            )
            return [velocity_m_s, -hbar * k * scattering_rate / mass]

        def at_exit(_time, state):
            return state[0] - slower_length_m

        def at_zero_velocity(_time, state):
            return state[1]

        at_exit.terminal = True
        at_exit.direction = 1
        at_zero_velocity.terminal = True
        at_zero_velocity.direction = -1
        solution = solve_ivp(
            equation,
            (0.0, 0.02),
            (0.0, float(initial_velocity_m_s)),
            events=(at_exit, at_zero_velocity),
            max_step=0.02 / 999.0,
        )
        return {
            "initial_velocity_m_s": float(initial_velocity_m_s),
            "final_position_mm": solution.y[0, -1] * 1e3,
            "final_velocity_m_s": solution.y[1, -1],
            "travel_time_ms": solution.t[-1] * 1e3,
            "exited_185mm": bool(solution.t_events[0].size),
        }

    return simulate


def main():
    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "runs" / "12_tapered_average_s_245mps_185mm"
    prefix = "Yb171_tapered_average_s_245mps_185mm"
    target = pd.read_csv(
        project_root / "inputs" / "Yb171_tapered_average_s_245mps_B_vs_z.csv"
    )
    field = pd.read_csv(run_dir / f"{prefix}_field.csv")
    config = pd.read_csv(run_dir / f"{prefix}_magnet_config.csv")
    individual = pd.read_csv(run_dir / f"{prefix}_individual_magnets.csv")

    expected_x_mm = np.array(
        [
            9.467, 22.400, 35.333, 48.267, 61.200,
            74.133, 87.067, 100.000, 112.933, 125.867,
            138.800, 151.733, 164.667, 177.600, 190.533,
        ]
    )
    total_magnets = int(config["total_magnets_pair"].sum())
    checks = {
        "target_is_0_to_185mm": np.isclose(target.z_m.iloc[0], 0.0)
        and np.isclose(target.z_m.iloc[-1], 0.185),
        "fifteen_fixed_longitudinal_positions": len(config) == 15
        and np.allclose(config.x_mm, expected_x_mm, atol=1e-9),
        "total_magnets_not_over_200": total_magnets <= 200,
        "individual_rows_match_total": len(individual) == total_magnets,
    }

    active = field.loc[(field.z_mm >= 0.0) & (field.z_mm <= 185.0)].copy()
    exp_compatible = pd.DataFrame(
        {
            "z_m": active.z_mm.to_numpy(dtype=float) * 1e-3,
            "z_cm": active.z_mm.to_numpy(dtype=float) * 1e-1,
            "B_T": active.B_sim_mT.to_numpy(dtype=float) * 1e-3,
            "B_mT": active.B_sim_mT.to_numpy(dtype=float),
        }
    )
    for column in (
        "beam_radius_mm",
        "s_average",
        "eta_available",
        "force_scale",
        "force_margin_fraction",
        "eta_design",
        "v_design_m_s",
        "delta_eff_MHz",
    ):
        exp_compatible[column] = np.interp(
            exp_compatible.z_m, target.z_m, target[column]
        )
    exp_path = (
        project_root
        / "inputs"
        / "Yb171_tapered_average_s_245mps_magnet_B_vs_z.csv"
    )
    exp_compatible.to_csv(exp_path, index=False)

    target_B = PchipInterpolator(target.z_m, target.B_T)
    magnet_B = PchipInterpolator(field.z_mm * 1e-3, field.B_sim_mT * 1e-3)
    rows = []
    for source, B_func in (("ideal", target_B), ("small_magnets", magnet_B)):
        simulate = make_simulator(B_func)
        for velocity in (240.0, 243.0, 245.0, 246.0):
            rows.append({"field_source": source, **simulate(velocity)})
    trajectory = pd.DataFrame(rows)
    trajectory_path = run_dir / f"{prefix}_trajectory_validation.csv"
    trajectory.to_csv(trajectory_path, index=False)

    atom_source = (
        project_root
        / "runs"
        / "11_tapered_variable_s_280mps_185mm"
        / "Yb171_tapered_variable_s_280mps_tapered_atomic_EXP_DATA_5000atoms.csv"
    )
    atoms = pd.read_csv(atom_source)
    initial_velocity = atoms.initial_velocity_m_s.to_numpy(dtype=float)
    simulate_magnets = make_simulator(magnet_B)
    final_velocity = np.empty(initial_velocity.size)
    for index, velocity in enumerate(initial_velocity):
        final_velocity[index] = simulate_magnets(velocity)["final_velocity_m_s"]
    success = (final_velocity > 0.0) & (final_velocity <= 50.0)
    design_window = (initial_velocity > 0.0) & (initial_velocity <= 245.0)
    initially_below = (initial_velocity > 0.0) & (initial_velocity <= 50.0)
    newly_slowed = success & ~initially_below

    atom_output = pd.DataFrame(
        {
            "atom_index": np.arange(initial_velocity.size),
            "initial_velocity_m_s": initial_velocity,
            "final_velocity_m_s": final_velocity,
            "in_0_to_245mps_design_window": design_window,
            "initially_below_50mps": initially_below,
            "success_final_0_to_50mps": success,
        }
    )
    atom_output_path = run_dir / f"{prefix}_EXP_DATA_5000atoms.csv"
    atom_output.to_csv(atom_output_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "n_samples": initial_velocity.size,
                "laser_power_mW": 35.0,
                "design_input_velocity_m_s": 245.0,
                "MOT_capture_velocity_m_s": 50.0,
                "all_atoms_success_count": int(success.sum()),
                "all_atoms_success_percent": float(success.mean() * 100.0),
                "design_window_atom_count": int(design_window.sum()),
                "design_window_fraction_percent": float(design_window.mean() * 100.0),
                "design_window_success_count": int((success & design_window).sum()),
                "design_window_success_percent": float(success[design_window].mean() * 100.0),
                "initially_below_MOT_count": int(initially_below.sum()),
                "newly_slowed_count": int(newly_slowed.sum()),
                "newly_slowed_percent_of_all": float(newly_slowed.mean() * 100.0),
                "mean_initial_velocity_m_s": float(initial_velocity.mean()),
                "mean_final_velocity_m_s": float(final_velocity.mean()),
            }
        ]
    )
    summary_path = run_dir / f"{prefix}_EXP_DATA_summary.csv"
    summary.to_csv(summary_path, index=False)

    magnet_245 = trajectory.loc[
        (trajectory.field_source == "small_magnets")
        & np.isclose(trajectory.initial_velocity_m_s, 245.0),
        "final_velocity_m_s",
    ].iloc[0]
    checks["245mps_magnet_atom_exits_below_50mps"] = 0.0 < magnet_245 <= 50.0
    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{name}: {passed}")
    print(trajectory.to_string(index=False))
    print(summary.to_string(index=False))
    for path in (trajectory_path, exp_path, atom_output_path, summary_path):
        print(path)
    if failed:
        raise RuntimeError("Validation failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
