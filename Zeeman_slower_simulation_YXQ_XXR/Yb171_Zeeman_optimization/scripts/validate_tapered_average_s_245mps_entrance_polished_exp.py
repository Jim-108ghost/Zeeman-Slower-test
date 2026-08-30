# -*- coding: utf-8 -*-
"""Run 5000-atom validation for the run-13 entrance-polished magnet field."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from validate_tapered_average_s_245mps_magnet import make_simulator


def main():
    project_root = Path(__file__).resolve().parent.parent
    run_dir = (
        project_root / "runs" / "13_tapered_average_s_245mps_entrance_polished"
    )
    prefix = "Yb171_tapered_average_s_245mps_185mm_entrance_polished"
    target = pd.read_csv(
        project_root / "inputs" / "Yb171_tapered_average_s_245mps_B_vs_z.csv"
    )
    field = pd.read_csv(run_dir / f"{prefix}_field.csv")
    magnet_B = PchipInterpolator(
        field.z_mm.to_numpy(dtype=float) * 1e-3,
        field.B_sim_mT.to_numpy(dtype=float) * 1e-3,
    )

    atom_source = (
        project_root
        / "runs"
        / "11_tapered_variable_s_280mps_185mm"
        / "Yb171_tapered_variable_s_280mps_tapered_atomic_EXP_DATA_5000atoms.csv"
    )
    source_atoms = pd.read_csv(atom_source)
    initial_velocity = source_atoms.initial_velocity_m_s.to_numpy(dtype=float)
    simulate = make_simulator(magnet_B)
    final_velocity = np.empty(initial_velocity.size)
    final_position = np.empty(initial_velocity.size)
    for index, velocity in enumerate(initial_velocity):
        result = simulate(velocity)
        final_velocity[index] = result["final_velocity_m_s"]
        final_position[index] = result["final_position_mm"]

    success = (final_velocity > 0.0) & (final_velocity <= 50.0)
    design_window = (initial_velocity > 0.0) & (initial_velocity <= 245.0)
    initially_below = (initial_velocity > 0.0) & (initial_velocity <= 50.0)
    newly_slowed = success & ~initially_below

    atom_output = pd.DataFrame(
        {
            "atom_index": np.arange(initial_velocity.size),
            "initial_velocity_m_s": initial_velocity,
            "final_position_mm": final_position,
            "final_velocity_m_s": final_velocity,
            "in_0_to_245mps_design_window": design_window,
            "initially_below_50mps": initially_below,
            "success_final_0_to_50mps": success,
        }
    )
    atoms_path = run_dir / f"{prefix}_EXP_DATA_5000atoms.csv"
    atom_output.to_csv(atoms_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "source_atom_file": str(atom_source),
                "field_source": "run13_158_small_magnets",
                "saturation_model": "position_dependent_transverse_average_s(x)",
                "n_samples": initial_velocity.size,
                "laser_power_mW": 35.0,
                "slower_length_mm": 185.0,
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

    active = field.loc[(field.z_mm >= 0.0) & (field.z_mm <= 185.0)].copy()
    compatible = pd.DataFrame(
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
        compatible[column] = np.interp(
            compatible.z_m, target.z_m, target[column]
        )
    compatible_path = (
        project_root
        / "inputs"
        / "Yb171_tapered_average_s_245mps_entrance_polished_magnet_B_vs_z.csv"
    )
    compatible.to_csv(compatible_path, index=False)

    print(summary.to_string(index=False))
    print(atoms_path)
    print(summary_path)
    print(compatible_path)


if __name__ == "__main__":
    main()
