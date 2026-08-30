# -*- coding: utf-8 -*-
"""Test the accepted 192-magnet field with the 8 mm average-beam model."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from validate_05final_210mps_8mm_exp import make_simulator


PREFIX = "Yb171_210mps_8mm_ordered_192magnets"


def main():
    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "runs" / "16_210mps_8mm_ordered_192magnets"
    magnet_data = pd.read_csv(run_dir / f"{PREFIX}_field.csv")
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

    simulate, s_average = make_simulator(magnet_B)
    final_position = np.empty(initial_velocity.size)
    final_velocity = np.empty(initial_velocity.size)
    exited = np.zeros(initial_velocity.size, dtype=bool)
    for index, velocity in enumerate(initial_velocity):
        final_position[index], final_velocity[index], exited[index] = simulate(
            velocity
        )
        if (index + 1) % 500 == 0:
            print(f"simulated {index + 1}/{initial_velocity.size} atoms", flush=True)

    success = (final_velocity > 0.0) & (final_velocity <= 50.0)
    newly_slowed = success & ~initially_below
    captured_and_exited = success & exited
    summary = pd.DataFrame(
        [
            {
                "field_source": "ordered_192_magnets",
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
        ]
    )

    atom_output = pd.DataFrame(
        {
            "atom_index": np.arange(initial_velocity.size),
            "initial_velocity_m_s": initial_velocity,
            "in_0_to_210mps_design_window": design_window,
            "initially_below_50mps": initially_below,
            "final_position_mm": final_position * 1e3,
            "final_velocity_m_s": final_velocity,
            "exited_185mm": exited,
            "success_final_0_to_50mps": success,
            "captured_and_exited": captured_and_exited,
        }
    )

    # Deterministic single-atom checks around the 210 m/s design boundary.
    single_rows = []
    for velocity in (50.0, 100.0, 150.0, 200.0, 205.0, 210.0, 211.0, 215.0):
        position, final_speed, did_exit = simulate(velocity)
        single_rows.append(
            {
                "initial_velocity_m_s": velocity,
                "final_position_mm": position * 1e3,
                "final_velocity_m_s": final_speed,
                "exited_185mm": did_exit,
                "success_final_0_to_50mps": 0.0 < final_speed <= 50.0,
            }
        )
    singles = pd.DataFrame(single_rows)

    # Put the already validated ideal-field reference next to this new result.
    baseline_path = (
        project_root
        / "runs"
        / "05_max200_startopt_final"
        / "Yb171_05final_210mps_8mm_EXP_DATA_summary.csv"
    )
    baseline = pd.read_csv(baseline_path)
    baseline = baseline.loc[baseline.field_source == "ideal"].copy()
    comparison = pd.concat([baseline, summary], ignore_index=True)

    atom_path = run_dir / f"{PREFIX}_EXP_DATA_5000atoms.csv"
    summary_path = run_dir / f"{PREFIX}_EXP_DATA_summary.csv"
    comparison_path = run_dir / f"{PREFIX}_EXP_DATA_vs_ideal.csv"
    single_path = run_dir / f"{PREFIX}_single_atom_validation.csv"
    atom_output.to_csv(atom_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    singles.to_csv(single_path, index=False)

    print("\nSingle-atom validation")
    print(singles.to_string(index=False))
    print("\n5000-atom comparison")
    print(
        comparison[
            [
                "field_source",
                "all_atoms_success_count",
                "all_atoms_success_percent",
                "captured_and_exited_count",
                "captured_and_exited_percent",
                "design_window_success_count",
                "design_window_success_percent",
                "newly_slowed_count",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:")
    for path in (summary_path, comparison_path, single_path, atom_path):
        print(path)


if __name__ == "__main__":
    main()
