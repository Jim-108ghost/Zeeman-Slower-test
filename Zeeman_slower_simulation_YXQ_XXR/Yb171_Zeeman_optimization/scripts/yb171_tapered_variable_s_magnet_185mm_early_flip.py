# -*- coding: utf-8 -*-
"""Early-flip variant of the variable-s 185 mm small-magnet design.

The first four stack pairs point along -z.  Beginning at x=61.200 mm, the
remaining stack pairs point along +z.  This deliberately flips one stack
before the target field's zero crossing so its fringe field helps correct the
entrance slope while all 15 longitudinal positions remain fixed.
"""

from pathlib import Path

import numpy as np

from yb171_tapered_variable_s_magnet_185mm import (
    ACTIVE_LENGTH_M,
    EXTENSION_M,
    MagnitudeOrderedOptimizer,
    prepare_extended_target,
    save_accuracy_summary,
    save_counts_plot,
    save_stack_configuration,
)
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import build_config
from yb171_zeeman_slower_zero_start_reasonable import (
    balanced_distance_polish,
    individual_magnet_dataframe,
    save_layout_plot,
)


OUTPUT_PREFIX = "Yb171_tapered_variable_s_185mm_early_flip"


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    source_csv = project_root / "inputs" / "Yb171_tapered_variable_s_B_vs_z.csv"
    prepared_csv = (
        project_root / "inputs" / "Yb171_tapered_variable_s_B_target_extended.csv"
    )
    output_dir = project_root / "runs" / "10_tapered_variable_s_185mm_early_flip"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_extended_target(source_csv, prepared_csv)

    cfg = build_config(scripts_dir)
    cfg.csv_path = str(prepared_csv)
    cfg.L_m = ACTIVE_LENGTH_M
    cfg.fit_left_extension_m = EXTENSION_M
    cfg.fit_right_extension_m = EXTENSION_M
    cfg.extension_fit_weight = 0.02
    cfg.extension_fit_decay_m = 1.0
    cfg.start_fit_weight = 30.0
    cfg.end_fit_weight = 30.0
    cfg.max_abs_error_weight = 0.10
    cfg.a_min = 4.0
    cfg.a_max = 20.0
    cfg.a_step = 0.05
    cfg.refine_integer_counts = True
    cfg.count_refinement_steps = (2, 1)
    cfg.max_count_refinement_sweeps = 5
    cfg.continuous_distance_polish = True
    cfg.continuous_polish_maxiter = 1000

    optimizer = MagnitudeOrderedOptimizer(cfg)
    optimizer.orientation = np.array(
        [-1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
        dtype=float,
    )
    result = optimizer.optimize(show_progress=True)
    result = balanced_distance_polish(optimizer, result)

    if result["total_magnets"] > cfg.max_total_magnets:
        raise RuntimeError("The optimized design exceeds 200 physical magnets")
    if not optimizer._counts_are_reasonable(result["n"]):
        raise RuntimeError("Optimized counts do not follow the target |B| trend")

    optimizer.print_summary(result)
    paths = list(
        optimizer.save_results(result, output_dir=output_dir, prefix=OUTPUT_PREFIX)
    )
    figures = list(
        optimizer.plot_results(
            result,
            output_dir=output_dir,
            prefix=OUTPUT_PREFIX,
            show=False,
        )
    )
    config_path = output_dir / f"{OUTPUT_PREFIX}_magnet_config.csv"
    save_stack_configuration(optimizer, result, config_path)
    individual_path = output_dir / f"{OUTPUT_PREFIX}_individual_magnets.csv"
    individual_df = individual_magnet_dataframe(optimizer, result)
    individual_df.to_csv(individual_path, index=False)
    layout_path = output_dir / f"{OUTPUT_PREFIX}_magnet_layout.png"
    save_layout_plot(
        individual_df,
        layout_path,
        tube_outer_diameter_mm=cfg.tube_outer_diameter_m * 1e3,
        title="Variable-s early-flip design: individual 8 mm x 3 mm N35 magnets",
    )
    counts_path = output_dir / f"{OUTPUT_PREFIX}_counts_vs_field.png"
    save_counts_plot(optimizer, result, counts_path)
    summary_path = output_dir / f"{OUTPUT_PREFIX}_accuracy_summary.csv"
    save_accuracy_summary(optimizer, result, source_csv, prepared_csv, summary_path)

    print("\nSaved:")
    for path in (
        *paths,
        *figures,
        individual_path,
        layout_path,
        counts_path,
        summary_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
