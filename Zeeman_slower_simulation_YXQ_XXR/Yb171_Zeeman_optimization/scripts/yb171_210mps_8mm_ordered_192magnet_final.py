# -*- coding: utf-8 -*-
"""Save the accepted 192-magnet, nondecreasing-count 210 -> 35 m/s design."""

from pathlib import Path

import numpy as np
import pandas as pd

from yb171_tapered_variable_s_magnet_185mm import (
    save_accuracy_summary,
    save_counts_plot,
    save_stack_configuration,
)
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import build_config
from yb171_zeeman_slower_zero_start_reasonable import (
    OrderedCountOptimizer,
    balanced_distance_polish,
    individual_magnet_dataframe,
    save_layout_plot,
)


OUTPUT_PREFIX = "Yb171_210mps_8mm_ordered_192magnets"
FIXED_COUNTS_EACH_SIDE = np.array(
    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 8, 8], dtype=int
)


def regional_metrics(optimizer, result):
    z_mm = optimizer.z_m * 1e3
    error_mT = (result["B_sim_T"] - optimizer.B_target_T) * 1e3
    rows = []
    for region, mask in (
        ("whole_0_185mm", (z_mm >= 0.0) & (z_mm <= 185.0)),
        ("start_0_30mm", (z_mm >= 0.0) & (z_mm <= 30.0)),
        ("middle_30_155mm", (z_mm > 30.0) & (z_mm < 155.0)),
        ("end_155_185mm", (z_mm >= 155.0) & (z_mm <= 185.0)),
    ):
        local_error = error_mT[mask]
        local_z = z_mm[mask]
        worst = int(np.argmax(np.abs(local_error)))
        rows.append(
            {
                "region": region,
                "rms_error_mT": float(np.sqrt(np.mean(local_error**2))),
                "mae_mT": float(np.mean(np.abs(local_error))),
                "max_abs_error_mT": float(np.max(np.abs(local_error))),
                "max_abs_error_position_mm": float(local_z[worst]),
            }
        )
    return pd.DataFrame(rows)


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    source_csv = project_root / "inputs" / "Yb171_ideal_B_field.csv"
    prepared_csv = project_root / "inputs" / "Yb171_target_B_extended.csv"
    output_dir = project_root / "runs" / "16_210mps_8mm_ordered_192magnets"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(scripts_dir)
    cfg.csv_path = str(prepared_csv)
    cfg.L_m = 0.185
    cfg.fit_left_extension_m = 0.013
    cfg.fit_right_extension_m = 0.013
    cfg.extension_fit_weight = 0.02
    cfg.extension_fit_decay_m = 1.0
    cfg.start_fit_weight = 30.0
    cfg.end_fit_weight = 30.0
    cfg.max_abs_error_weight = 0.10

    optimizer = OrderedCountOptimizer(cfg)
    result = optimizer._optimize_d_for_fixed_counts(FIXED_COUNTS_EACH_SIDE)
    if result is None:
        raise RuntimeError("The accepted fixed-count layout is geometrically invalid")
    result.update(
        {
            "a": np.nan,
            "counts_refined": False,
            "total_magnets": int(2 * np.sum(FIXED_COUNTS_EACH_SIDE)),
        }
    )
    result = balanced_distance_polish(optimizer, result)

    if result["total_magnets"] > cfg.max_total_magnets:
        raise RuntimeError("The accepted design exceeds 200 physical magnets")
    if not optimizer._counts_are_reasonable(result["n"]):
        raise RuntimeError("Magnet counts are not nondecreasing")

    optimizer.print_summary(result)
    paths = list(
        optimizer.save_results(result, output_dir=output_dir, prefix=OUTPUT_PREFIX)
    )
    figures = list(
        optimizer.plot_results(
            result, output_dir=output_dir, prefix=OUTPUT_PREFIX, show=False
        )
    )

    config_path = output_dir / f"{OUTPUT_PREFIX}_magnet_config.csv"
    save_stack_configuration(optimizer, result, config_path)
    individual_path = output_dir / f"{OUTPUT_PREFIX}_individual_magnets.csv"
    individual = individual_magnet_dataframe(optimizer, result)
    individual.to_csv(individual_path, index=False)
    layout_path = output_dir / f"{OUTPUT_PREFIX}_magnet_layout.png"
    save_layout_plot(
        individual,
        layout_path,
        tube_outer_diameter_mm=cfg.tube_outer_diameter_m * 1e3,
        title="210 to 35 m/s: ordered 192-magnet layout",
    )
    counts_path = output_dir / f"{OUTPUT_PREFIX}_counts_vs_field.png"
    save_counts_plot(optimizer, result, counts_path)
    accuracy_path = output_dir / f"{OUTPUT_PREFIX}_accuracy_summary.csv"
    save_accuracy_summary(optimizer, result, source_csv, prepared_csv, accuracy_path)
    regional_path = output_dir / f"{OUTPUT_PREFIX}_regional_error_summary.csv"
    regional = regional_metrics(optimizer, result)
    regional.to_csv(regional_path, index=False)

    # Notebook/EXP-DATA-ready copy, with explicit SI and plotting columns.
    field = pd.read_csv(output_dir / f"{OUTPUT_PREFIX}_field.csv")
    import_field = pd.DataFrame(
        {
            "z_m": field["z_mm"].to_numpy(dtype=float) * 1e-3,
            "z_cm": field["z_mm"].to_numpy(dtype=float) * 0.1,
            "B_T": field["B_sim_mT"].to_numpy(dtype=float) * 1e-3,
            "B_mT": field["B_sim_mT"].to_numpy(dtype=float),
        }
    )
    import_path = (
        project_root / "inputs" / f"{OUTPUT_PREFIX}_B_vs_z.csv"
    )
    import_field.to_csv(import_path, index=False)

    print("\nRegional field error")
    print(regional.to_string(index=False))
    print("\nSaved:")
    for path in (
        *paths,
        *figures,
        individual_path,
        layout_path,
        counts_path,
        accuracy_path,
        regional_path,
        import_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
