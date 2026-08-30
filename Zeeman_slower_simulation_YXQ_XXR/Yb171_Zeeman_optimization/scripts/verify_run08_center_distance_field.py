# -*- coding: utf-8 -*-
"""Verify run-08 geometry and field directly from magnet centre distances."""

from pathlib import Path

import numpy as np
import pandas as pd

from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import build_config
from yb171_zeeman_slower_zero_start_reasonable import OrderedCountOptimizer


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    run_dir = project_root / "runs" / "08_single_atom_complete_185mm"
    config = pd.read_csv(
        run_dir / "Yb171_single_atom_complete_185mm_magnet_config.csv"
    )
    individual = pd.read_csv(
        run_dir / "Yb171_single_atom_complete_185mm_individual_magnet_positions.csv"
    )
    saved_field = pd.read_csv(
        run_dir / "Yb171_single_atom_complete_185mm_field.csv"
    )

    cfg = build_config(scripts_dir)
    cfg.csv_path = str(
        project_root / "inputs" / "Yb171_single_atom_B_vs_z_current_185mm.csv"
    )
    cfg.L_m = 0.185
    cfg.target_mode = "crop"
    cfg.fit_left_extension_m = 0.0
    cfg.fit_right_extension_m = 0.0
    cfg.extension_fit_weight = 1.0
    optimizer = OrderedCountOptimizer(cfg)

    n = config.n_magnets_each_side.to_numpy(dtype=int)
    d_stack_center_m = config.d_stack_center_mm.to_numpy(dtype=float) * 1e-3
    z_m = saved_field.z_mm.to_numpy(dtype=float) * 1e-3
    recalculated_B_mT = (
        optimizer.simulated_field_at_z(
            {"n": n, "d_m": d_stack_center_m}, z_m
        )
        * 1e3
    )
    saved_B_mT = saved_field.B_sim_mT.to_numpy(dtype=float)
    field_difference_mT = recalculated_B_mT - saved_B_mT

    thickness_mm = float(individual.magnet_thickness_mm.iloc[0])
    half_thickness_mm = thickness_mm / 2.0
    pitch_mm = (cfg.magnet_thickness_m + cfg.magnet_gap_m) * 1e3
    geometry_rows = []
    maximum_center_mismatch_mm = 0.0
    for row in config.itertuples(index=False):
        offsets_mm = (
            np.arange(row.n_magnets_each_side, dtype=float)
            - 0.5 * (row.n_magnets_each_side - 1)
        ) * pitch_mm
        rebuilt_upper_centers_mm = row.d_stack_center_mm + offsets_mm
        recorded_upper_centers_mm = np.sort(
            individual.loc[
                (individual.stack_index == row.stack_index)
                & (individual.stack_side == "upper"),
                "transverse_z_mm",
            ].to_numpy(dtype=float)
        )
        center_mismatch_mm = float(
            np.max(np.abs(rebuilt_upper_centers_mm - recorded_upper_centers_mm))
        )
        maximum_center_mismatch_mm = max(
            maximum_center_mismatch_mm, center_mismatch_mm
        )
        nearest_center_mm = float(rebuilt_upper_centers_mm.min())
        inner_surface_mm = nearest_center_mm - half_thickness_mm
        geometry_rows.append(
            {
                "stack_index": int(row.stack_index),
                "x_mm": float(row.x_mm),
                "n_magnets_each_side": int(row.n_magnets_each_side),
                "d_stack_center_mm": float(row.d_stack_center_mm),
                "rebuilt_nearest_magnet_center_mm": nearest_center_mm,
                "half_thickness_subtracted_mm": half_thickness_mm,
                "verified_inner_surface_distance_mm": inner_surface_mm,
                "rebuilt_vs_recorded_center_max_difference_mm": center_mismatch_mm,
            }
        )

    # Deliberately shift every physical magnet 1.5 mm inward, equivalent to
    # mistakenly treating an inner-face coordinate as a magnet-centre coordinate.
    wrong_inward_B_mT = (
        optimizer.simulated_field_at_z(
            {"n": n, "d_m": d_stack_center_m - half_thickness_mm * 1e-3},
            z_m,
        )
        * 1e3
    )
    wrong_shift_difference_mT = wrong_inward_B_mT - saved_B_mT

    geometry = pd.DataFrame(geometry_rows)
    comparison = pd.DataFrame(
        {
            "z_mm": saved_field.z_mm,
            "saved_B_mT": saved_B_mT,
            "recalculated_from_centers_B_mT": recalculated_B_mT,
            "center_recalculation_minus_saved_mT": field_difference_mT,
            "wrong_centers_shifted_inward_1p5mm_B_mT": wrong_inward_B_mT,
            "wrong_shift_minus_saved_mT": wrong_shift_difference_mT,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "maximum_rebuilt_center_mismatch_mm": maximum_center_mismatch_mm,
                "center_based_field_rms_difference_mT": float(
                    np.sqrt(np.mean(field_difference_mT**2))
                ),
                "center_based_field_max_abs_difference_mT": float(
                    np.max(np.abs(field_difference_mT))
                ),
                "wrong_inward_1p5mm_field_rms_difference_mT": float(
                    np.sqrt(np.mean(wrong_shift_difference_mT**2))
                ),
                "wrong_inward_1p5mm_field_max_abs_difference_mT": float(
                    np.max(np.abs(wrong_shift_difference_mT))
                ),
            }
        ]
    )

    geometry_path = run_dir / "Yb171_single_atom_complete_185mm_center_geometry_check.csv"
    comparison_path = run_dir / "Yb171_single_atom_complete_185mm_center_field_check.csv"
    summary_path = run_dir / "Yb171_single_atom_complete_185mm_center_check_summary.csv"
    geometry.to_csv(geometry_path, index=False, float_format="%.12f")
    comparison.to_csv(comparison_path, index=False, float_format="%.12e")
    summary.to_csv(summary_path, index=False, float_format="%.12e")

    print("Geometry reconstructed from stack-centre distances")
    print(geometry.to_string(index=False))
    print("\nVerification summary")
    print(summary.to_string(index=False))
    print("\nSaved:")
    for path in (geometry_path, comparison_path, summary_path):
        print(path)


if __name__ == "__main__":
    main()
