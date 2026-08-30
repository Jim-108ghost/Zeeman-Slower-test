# -*- coding: utf-8 -*-
"""Export verified inner-face distances for the run-08 magnet layout."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "runs" / "08_single_atom_complete_185mm"
    config_path = run_dir / "Yb171_single_atom_complete_185mm_magnet_config.csv"
    individual_path = (
        run_dir / "Yb171_single_atom_complete_185mm_individual_magnet_positions.csv"
    )
    output_path = (
        run_dir / "Yb171_single_atom_complete_185mm_surface_to_axis.csv"
    )

    config = pd.read_csv(config_path)
    individual = pd.read_csv(individual_path)
    thickness_values = individual.magnet_thickness_mm.unique()
    if thickness_values.size != 1:
        raise RuntimeError("Magnet thickness is not uniform")
    thickness_mm = float(thickness_values[0])
    half_thickness_mm = 0.5 * thickness_mm

    rows = []
    for cfg_row in config.itertuples(index=False):
        group = individual.loc[individual.stack_index == cfg_row.stack_index]
        upper = group.loc[group.stack_side == "upper"]
        lower = group.loc[group.stack_side == "lower"]
        if len(upper) != cfg_row.n_magnets_each_side or len(lower) != len(upper):
            raise RuntimeError(f"Stack {cfg_row.stack_index}: magnet count mismatch")

        upper_nearest_center_y_mm = float(upper.transverse_z_mm.min())
        lower_nearest_center_y_mm = float(lower.transverse_z_mm.max())
        config_nearest_center_mm = float(cfg_row.nearest_magnet_center_mm)
        if not np.isclose(
            upper_nearest_center_y_mm, config_nearest_center_mm, atol=1e-10
        ):
            raise RuntimeError(
                f"Stack {cfg_row.stack_index}: config/individual centre mismatch"
            )
        if not np.isclose(
            lower_nearest_center_y_mm, -config_nearest_center_mm, atol=1e-10
        ):
            raise RuntimeError(
                f"Stack {cfg_row.stack_index}: upper/lower centres are not mirrored"
            )

        # The inner face is toward y=0. Thus the positive-y (upper) stack uses
        # centre - t/2, while the negative-y (lower) stack uses centre + t/2.
        upper_lower_surface_y_mm = upper_nearest_center_y_mm - half_thickness_mm
        lower_upper_surface_y_mm = lower_nearest_center_y_mm + half_thickness_mm
        surface_distance_mm = config_nearest_center_mm - half_thickness_mm
        if not np.isclose(
            upper_lower_surface_y_mm, surface_distance_mm, atol=1e-10
        ) or not np.isclose(
            lower_upper_surface_y_mm, -surface_distance_mm, atol=1e-10
        ):
            raise RuntimeError(f"Stack {cfg_row.stack_index}: face-distance check failed")

        rows.append(
            {
                "stack_index": int(cfg_row.stack_index),
                "x_mm": float(cfg_row.x_mm),
                "n_magnets_each_side": int(cfg_row.n_magnets_each_side),
                "magnet_thickness_mm": thickness_mm,
                "half_thickness_correction_mm": half_thickness_mm,
                "nearest_magnet_center_distance_mm": config_nearest_center_mm,
                "upper_stack_lower_surface_y_mm": upper_lower_surface_y_mm,
                "lower_stack_upper_surface_y_mm": lower_upper_surface_y_mm,
                "inner_surface_distance_to_axis_mm": surface_distance_mm,
                "distance_formula": "nearest center distance - 1.5 mm",
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(output_path, index=False, float_format="%.9f")
    print(output.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
