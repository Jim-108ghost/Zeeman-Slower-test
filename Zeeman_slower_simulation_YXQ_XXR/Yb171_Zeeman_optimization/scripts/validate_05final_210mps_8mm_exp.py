# -*- coding: utf-8 -*-
"""Validate the run-05 200-magnet field with the original 8 mm average beam."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


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
    beam_radius_m = 8e-3
    slower_length_m = 0.185
    saturation_intensity = np.pi * h * c * gamma / (3.0 * wavelength**3)
    intensity_axis = 2.0 * laser_power_W / (np.pi * beam_radius_m**2)
    s_average = 0.5 * intensity_axis / saturation_intensity

    def simulate(initial_velocity_m_s):
        def equation(_time, state):
            z_m, velocity_m_s = state
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
        return (
            float(solution.y[0, -1]),
            float(solution.y[1, -1]),
            bool(solution.t_events[0].size),
        )

    return simulate, s_average


def main():
    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "runs" / "05_max200_startopt_final"
    ideal_data = pd.read_csv(project_root / "inputs" / "Yb171_ideal_B_field.csv")
    magnet_data = pd.read_csv(
        run_dir / "Yb171_zeeman_15stacks_max200_startopt_final_field.csv"
    )
    ideal_B = PchipInterpolator(ideal_data.z_m, ideal_data.B_T)
    magnet_B = PchipInterpolator(
        magnet_data.z_mm.to_numpy(dtype=float) * 1e-3,
        magnet_data.B_sim_mT.to_numpy(dtype=float) * 1e-3,
    )

    atom_source = (
        project_root
        / "runs"
        / "11_tapered_variable_s_280mps_185mm"
        / "Yb171_tapered_variable_s_280mps_tapered_atomic_EXP_DATA_5000atoms.csv"
    )
    source_atoms = pd.read_csv(atom_source)
    initial_velocity = source_atoms.initial_velocity_m_s.to_numpy(dtype=float)
    design_window = (initial_velocity > 0.0) & (initial_velocity <= 210.0)
    initially_below = (initial_velocity > 0.0) & (initial_velocity <= 50.0)

    atom_output = pd.DataFrame(
        {
            "atom_index": np.arange(initial_velocity.size),
            "initial_velocity_m_s": initial_velocity,
            "in_0_to_210mps_design_window": design_window,
            "initially_below_50mps": initially_below,
        }
    )
    summaries = []
    for field_name, B_func in (("ideal", ideal_B), ("run05_200_magnets", magnet_B)):
        simulate, s_average = make_simulator(B_func)
        final_position = np.empty(initial_velocity.size)
        final_velocity = np.empty(initial_velocity.size)
        exited = np.zeros(initial_velocity.size, dtype=bool)
        for index, velocity in enumerate(initial_velocity):
            final_position[index], final_velocity[index], exited[index] = simulate(
                velocity
            )
        success = (final_velocity > 0.0) & (final_velocity <= 50.0)
        newly_slowed = success & ~initially_below
        captured_and_exited = success & exited
        atom_output[f"{field_name}_final_position_mm"] = final_position * 1e3
        atom_output[f"{field_name}_final_velocity_m_s"] = final_velocity
        atom_output[f"{field_name}_exited_185mm"] = exited
        atom_output[f"{field_name}_success_final_0_to_50mps"] = success
        summaries.append(
            {
                "field_source": field_name,
                "source_atom_file": str(atom_source),
                "saturation_model": "constant_transverse_average_s",
                "s_average": s_average,
                "n_samples": initial_velocity.size,
                "laser_power_mW": 35.0,
                "beam_radius_mm": 8.0,
                "slower_length_mm": 185.0,
                "design_input_velocity_m_s": 210.0,
                "design_final_velocity_m_s": 35.0,
                "MOT_capture_velocity_m_s": 50.0,
                "all_atoms_success_count": int(success.sum()),
                "all_atoms_success_percent": float(success.mean() * 100.0),
                "captured_and_exited_count": int(captured_and_exited.sum()),
                "captured_and_exited_percent": float(
                    captured_and_exited.mean() * 100.0
                ),
                "design_window_atom_count": int(design_window.sum()),
                "design_window_fraction_percent": float(design_window.mean() * 100.0),
                "design_window_success_count": int((success & design_window).sum()),
                "design_window_success_percent": float(
                    success[design_window].mean() * 100.0
                ),
                "initially_below_MOT_count": int(initially_below.sum()),
                "newly_slowed_count": int(newly_slowed.sum()),
                "newly_slowed_percent_of_all": float(newly_slowed.mean() * 100.0),
                "mean_initial_velocity_m_s": float(initial_velocity.mean()),
                "mean_final_velocity_m_s": float(final_velocity.mean()),
            }
        )

    active = magnet_data.loc[
        (magnet_data.z_mm >= 0.0) & (magnet_data.z_mm <= 185.0)
    ].copy()
    residual = active.residual_mT.to_numpy(dtype=float)
    z_mm = active.z_mm.to_numpy(dtype=float)
    error_rows = []
    for region, mask in (
        ("whole_0_185mm", (z_mm >= 0.0) & (z_mm <= 185.0)),
        ("start_0_30mm", (z_mm >= 0.0) & (z_mm <= 30.0)),
        ("middle_30_155mm", (z_mm > 30.0) & (z_mm < 155.0)),
        ("end_155_185mm", (z_mm >= 155.0) & (z_mm <= 185.0)),
    ):
        local = residual[mask]
        local_z = z_mm[mask]
        maximum_index = int(np.argmax(np.abs(local)))
        error_rows.append(
            {
                "region": region,
                "rms_error_mT": float(np.sqrt(np.mean(local**2))),
                "mae_mT": float(np.mean(np.abs(local))),
                "max_abs_error_mT": float(np.max(np.abs(local))),
                "max_abs_error_position_mm": float(local_z[maximum_index]),
            }
        )

    summary = pd.DataFrame(summaries)
    errors = pd.DataFrame(error_rows)
    atom_path = run_dir / "Yb171_05final_210mps_8mm_EXP_DATA_5000atoms.csv"
    summary_path = run_dir / "Yb171_05final_210mps_8mm_EXP_DATA_summary.csv"
    error_path = run_dir / "Yb171_05final_field_error_summary.csv"
    atom_output.to_csv(atom_path, index=False)
    summary.to_csv(summary_path, index=False)
    errors.to_csv(error_path, index=False)
    print("Field error summary")
    print(errors.to_string(index=False))
    print("\n5000-atom summary")
    print(summary.to_string(index=False))
    for path in (error_path, summary_path, atom_path):
        print(path)


if __name__ == "__main__":
    main()
