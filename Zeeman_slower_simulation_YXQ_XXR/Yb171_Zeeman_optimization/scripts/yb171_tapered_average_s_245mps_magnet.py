# -*- coding: utf-8 -*-
"""Fit the 245 m/s average-s target with at most 200 small magnets."""

from pathlib import Path

from yb171_tapered_variable_s_magnet_185mm import (
    ACTIVE_LENGTH_M,
    EXTENSION_M,
    prepare_extended_target,
    save_accuracy_summary,
    save_counts_plot,
    save_stack_configuration,
)
from yb171_zeeman_slower_optimizer_15stacks import PermanentMagnetZeemanSlower
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import build_config
from yb171_zeeman_slower_zero_start_reasonable import (
    balanced_distance_polish,
    individual_magnet_dataframe,
    save_layout_plot,
)


OUTPUT_PREFIX = "Yb171_tapered_average_s_245mps_185mm"


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    source_csv = (
        project_root / "inputs" / "Yb171_tapered_average_s_245mps_B_vs_z.csv"
    )
    prepared_csv = (
        project_root
        / "inputs"
        / "Yb171_tapered_average_s_245mps_B_target_extended.csv"
    )
    output_dir = project_root / "runs" / "12_tapered_average_s_245mps_185mm"
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
    cfg.a_min = 2.0
    cfg.a_max = 25.0
    cfg.a_step = 0.05
    # Counts are deliberately unconstrained: no entrance-low / exit-high rule
    # and no monotonic relation to the target-field magnitude.
    cfg.refine_integer_counts = True
    cfg.count_refinement_steps = (2, 1)
    cfg.max_count_refinement_sweeps = 8
    cfg.continuous_distance_polish = True
    cfg.continuous_polish_maxiter = 1400

    optimizer = PermanentMagnetZeemanSlower(cfg)
    result = optimizer.optimize(show_progress=True)
    result = balanced_distance_polish(optimizer, result)

    if result["total_magnets"] > cfg.max_total_magnets:
        raise RuntimeError("The optimized design exceeds 200 physical magnets")

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
        title="245 m/s average-s target: individual 8 mm x 3 mm N35 magnets",
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
        prepared_csv,
    ):
        print(path)


if __name__ == "__main__":
    main()
