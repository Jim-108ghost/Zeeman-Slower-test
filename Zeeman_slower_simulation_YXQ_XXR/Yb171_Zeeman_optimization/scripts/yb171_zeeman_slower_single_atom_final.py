# -*- coding: utf-8 -*-
"""Export the final small-magnet array against Yb171_single_atom_B_vs_z.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from yb171_zeeman_slower_optimizer_15stacks import (
    PermanentMagnetZeemanSlower,
)
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import (
    build_config,
)
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt_final import (
    build_result,
)


OUTPUT_PREFIX = "Yb171_single_atom_small_magnets_final"
START_REGION_MM = 30.0
START_ERROR_DEADBAND_MT = 0.1


def individual_magnet_dataframe(optimizer, result):
    """Return the center coordinates of all physical disk magnets."""
    rows = []
    pitch_m = (
        optimizer.cfg.magnet_thickness_m + optimizer.cfg.magnet_gap_m
    )
    magnet_id = 1

    for stack_i, (x_m, n_each_side, d_m, orientation) in enumerate(
        zip(
            optimizer.x_stack_m,
            result["n"],
            result["d_m"],
            optimizer.orientation,
        ),
        start=1,
    ):
        n_each_side = int(n_each_side)
        offsets_m = (
            np.arange(n_each_side, dtype=float)
            - 0.5 * (n_each_side - 1)
        ) * pitch_m
        positive_centers_m = d_m + offsets_m

        for side_name, side_sign in (("upper", 1), ("lower", -1)):
            for local_i, positive_center_m in enumerate(
                positive_centers_m,
                start=1,
            ):
                rows.append(
                    {
                        "magnet_id": magnet_id,
                        "stack_index": stack_i,
                        "stack_side": side_name,
                        "magnet_index_on_side": local_i,
                        "longitudinal_x_mm": x_m * 1e3,
                        "transverse_z_mm": side_sign
                        * positive_center_m
                        * 1e3,
                        "distance_to_axis_mm": positive_center_m * 1e3,
                        "moment_orientation": int(orientation),
                        "magnet_diameter_mm": (
                            optimizer.cfg.magnet_diameter_m * 1e3
                        ),
                        "magnet_thickness_mm": (
                            optimizer.cfg.magnet_thickness_m * 1e3
                        ),
                    }
                )
                magnet_id += 1

    return pd.DataFrame(rows)


def save_layout_plot(individual_df, output_path, tube_outer_diameter_mm):
    """Draw a longitudinal/transverse side view of every disk magnet."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"upper": "tab:red", "lower": "tab:blue"}

    for row in individual_df.itertuples(index=False):
        rectangle = Rectangle(
            (
                row.longitudinal_x_mm - 0.5 * row.magnet_diameter_mm,
                row.transverse_z_mm - 0.5 * row.magnet_thickness_mm,
            ),
            row.magnet_diameter_mm,
            row.magnet_thickness_mm,
            facecolor=colors[row.stack_side],
            edgecolor="black",
            linewidth=0.25,
            alpha=0.8,
        )
        ax.add_patch(rectangle)

    tube_radius_mm = 0.5 * tube_outer_diameter_mm
    ax.axhline(0.0, color="black", linewidth=1.0, label="atomic-beam axis")
    ax.axhline(
        tube_radius_mm,
        color="0.5",
        linestyle="--",
        linewidth=1.0,
        label="tube outer surface",
    )
    ax.axhline(
        -tube_radius_mm,
        color="0.5",
        linestyle="--",
        linewidth=1.0,
    )
    ax.scatter([], [], marker="s", color=colors["upper"], label="upper magnets")
    ax.scatter([], [], marker="s", color=colors["lower"], label="lower magnets")
    ax.set_xlim(-5.0, 200.0)
    max_distance = individual_df["distance_to_axis_mm"].max() + 8.0
    ax.set_ylim(-max_distance, max_distance)
    ax.set_xlabel("Longitudinal position x (mm)")
    ax.set_ylabel("Transverse magnet-center position z (mm)")
    ax.set_title("Yb171 single-atom target: individual 8 mm x 3 mm magnets")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    input_csv = project_root / "inputs" / "Yb171_single_atom_B_vs_z.csv"
    output_dir = project_root / "runs" / "06_single_atom_target_final"

    # Use the same magnet dimensions, fixed longitudinal positions, and
    # <=200-magnet constraints as 05_final, but read the single-atom CSV
    # directly. Its z_m and B_T columns already use SI units.
    cfg = build_config(scripts_dir)
    cfg.csv_path = str(input_csv)
    cfg.fit_left_extension_m = 0.0
    cfg.fit_right_extension_m = 0.0
    cfg.extension_fit_weight = 1.0

    optimizer = PermanentMagnetZeemanSlower(cfg)
    result = build_result(optimizer)
    optimizer.print_summary(result)

    paths = optimizer.save_results(
        result,
        output_dir=output_dir,
        prefix=OUTPUT_PREFIX,
    )

    # Keep the physical target/simulated fields untouched. For acceptance
    # reporting only, treat up to 0.1 mT of residual in the entrance 30 mm
    # as zero, as requested.
    field_path = paths[1]
    field_df = pd.read_csv(field_path)
    raw_residual_mT = field_df["residual_mT"].to_numpy(dtype=float)
    start_mask = (
        (field_df["z_mm"].to_numpy(dtype=float) >= 0.0)
        & (field_df["z_mm"].to_numpy(dtype=float) <= START_REGION_MM)
    )
    acceptance_residual_mT = raw_residual_mT.copy()
    acceptance_residual_mT[start_mask] = np.sign(
        raw_residual_mT[start_mask]
    ) * np.maximum(
        np.abs(raw_residual_mT[start_mask]) - START_ERROR_DEADBAND_MT,
        0.0,
    )
    field_df[
        "acceptance_residual_with_start_0p1mT_deadband_mT"
    ] = acceptance_residual_mT
    field_df.to_csv(field_path, index=False)
    figures = optimizer.plot_results(
        result,
        output_dir=output_dir,
        prefix=OUTPUT_PREFIX,
        show=False,
    )

    individual_path = (
        output_dir / f"{OUTPUT_PREFIX}_individual_magnet_positions.csv"
    )
    individual_df = individual_magnet_dataframe(optimizer, result)
    individual_df.to_csv(individual_path, index=False)

    layout_path = output_dir / f"{OUTPUT_PREFIX}_magnet_layout.png"
    save_layout_plot(
        individual_df,
        layout_path,
        tube_outer_diameter_mm=cfg.tube_outer_diameter_m * 1e3,
    )

    summary_path = output_dir / f"{OUTPUT_PREFIX}_accuracy_summary.csv"
    pd.DataFrame(
        [
            {
                "target_csv": input_csv.name,
                "active_length_mm": cfg.L_m * 1e3,
                "magnet_diameter_mm": cfg.magnet_diameter_m * 1e3,
                "magnet_thickness_mm": cfg.magnet_thickness_m * 1e3,
                "total_magnets": result["total_magnets"],
                "active_rms_error_mT": result["rms_error_T"] * 1e3,
                "active_mae_mT": result["mae_T"] * 1e3,
                "active_max_abs_error_mT": (
                    result["max_abs_error_T"] * 1e3
                ),
                "start_30mm_rms_error_mT": (
                    result["start_30mm_rms_error_T"] * 1e3
                ),
                "start_30mm_max_abs_error_mT": (
                    result["start_30mm_max_abs_error_T"] * 1e3
                ),
                "start_error_deadband_mT": START_ERROR_DEADBAND_MT,
                "start_30mm_acceptance_rms_error_mT": float(
                    np.sqrt(
                        np.mean(acceptance_residual_mT[start_mask] ** 2)
                    )
                ),
                "start_30mm_acceptance_max_abs_error_mT": float(
                    np.max(np.abs(acceptance_residual_mT[start_mask]))
                ),
                "middle_rms_error_mT": (
                    result["middle_rms_error_T"] * 1e3
                ),
                "end_30mm_rms_error_mT": (
                    result["end_30mm_rms_error_T"] * 1e3
                ),
                "end_30mm_max_abs_error_mT": (
                    result["end_30mm_max_abs_error_T"] * 1e3
                ),
            }
        ]
    ).to_csv(summary_path, index=False)

    print("\nSaved:")
    for path in (
        *paths,
        *figures,
        individual_path,
        layout_path,
        summary_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
