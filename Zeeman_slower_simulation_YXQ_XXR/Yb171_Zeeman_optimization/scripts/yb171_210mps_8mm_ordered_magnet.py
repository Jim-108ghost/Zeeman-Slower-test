# -*- coding: utf-8 -*-
"""Fit the 210 -> 35 m/s target with a nondecreasing <=200-magnet layout."""

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


OUTPUT_PREFIX = "Yb171_210mps_8mm_ordered_185mm"


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    source_csv = project_root / "inputs" / "Yb171_ideal_B_field.csv"
    prepared_csv = project_root / "inputs" / "Yb171_target_B_extended.csv"
    output_dir = project_root / "runs" / "14_210mps_8mm_ordered_185mm"
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
    cfg.a_min = 2.0
    cfg.a_max = 30.0
    cfg.a_step = 0.05
    cfg.refine_integer_counts = True
    cfg.count_refinement_steps = (2, 1)
    cfg.max_count_refinement_sweeps = 10
    cfg.continuous_distance_polish = True
    cfg.continuous_polish_maxiter = 1600

    optimizer = OrderedCountOptimizer(cfg)
    result = optimizer.optimize(show_progress=True)
    result = balanced_distance_polish(optimizer, result)

    if result["total_magnets"] > cfg.max_total_magnets:
        raise RuntimeError("The ordered design exceeds 200 physical magnets")
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
        title="210 to 35 m/s: ordered 8 mm x 3 mm N35 magnets",
    )
    counts_path = output_dir / f"{OUTPUT_PREFIX}_counts_vs_field.png"
    save_counts_plot(optimizer, result, counts_path)
    accuracy_path = output_dir / f"{OUTPUT_PREFIX}_accuracy_summary.csv"
    save_accuracy_summary(
        optimizer, result, source_csv, prepared_csv, accuracy_path
    )

    # Direct, uniformly sampled comparison against the selected run-05 field.
    new_field = pd.read_csv(output_dir / f"{OUTPUT_PREFIX}_field.csv")
    old_field = pd.read_csv(
        project_root
        / "runs"
        / "05_max200_startopt_final"
        / "Yb171_zeeman_15stacks_max200_startopt_final_field.csv"
    )

    def metrics(table):
        active = table.loc[
            (table.z_mm >= 0.0) & (table.z_mm <= 185.0)
        ].copy()
        z = active.z_mm.to_numpy(dtype=float)
        error = active.residual_mT.to_numpy(dtype=float)
        rows = {}
        for region, mask in (
            ("whole", (z >= 0.0) & (z <= 185.0)),
            ("start", (z >= 0.0) & (z <= 30.0)),
            ("middle", (z > 30.0) & (z < 155.0)),
            ("end", (z >= 155.0) & (z <= 185.0)),
        ):
            local = error[mask]
            rows[f"{region}_rms_mT"] = float(np.sqrt(np.mean(local**2)))
            rows[f"{region}_max_mT"] = float(np.max(np.abs(local)))
        return rows

    comparison = pd.DataFrame(
        [
            {"design": "run05_unordered", **metrics(old_field)},
            {"design": "run14_ordered", **metrics(new_field)},
        ]
    )
    comparison_path = output_dir / f"{OUTPUT_PREFIX}_vs_run05.csv"
    comparison.to_csv(comparison_path, index=False)
    print(comparison.to_string(index=False))

    print("\nSaved:")
    for path in (
        *paths,
        *figures,
        individual_path,
        layout_path,
        counts_path,
        accuracy_path,
        comparison_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
