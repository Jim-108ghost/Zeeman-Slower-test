# -*- coding: utf-8 -*-
"""Optimize the equivalent zero-offset 210 m/s field with ordered counts."""

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


OUTPUT_PREFIX = "Yb171_210mps_8mm_zero_offset_ordered_185mm"


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    source_csv = project_root / "inputs" / "Yb171_210mps_8mm_zero_offset_B.csv"
    prepared_csv = (
        project_root / "inputs" / "Yb171_210mps_8mm_zero_offset_B_extended.csv"
    )
    output_dir = (
        project_root / "runs" / "15_210mps_8mm_zero_offset_ordered_185mm"
    )
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
        raise RuntimeError("The zero-offset design exceeds 200 magnets")
    if not optimizer._counts_are_reasonable(result["n"]):
        raise RuntimeError("The zero-offset counts are not nondecreasing")

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
        title="Equivalent 210 m/s zero-offset field: ordered N35 magnets",
    )
    counts_path = output_dir / f"{OUTPUT_PREFIX}_counts_vs_field.png"
    save_counts_plot(optimizer, result, counts_path)
    accuracy_path = output_dir / f"{OUTPUT_PREFIX}_accuracy_summary.csv"
    save_accuracy_summary(
        optimizer, result, source_csv, prepared_csv, accuracy_path
    )

    field = pd.read_csv(output_dir / f"{OUTPUT_PREFIX}_field.csv")
    active = field.loc[(field.z_mm >= 0.0) & (field.z_mm <= 185.0)]
    error = active.residual_mT.to_numpy(dtype=float)
    z = active.z_mm.to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "total_magnets": int(result["total_magnets"]),
                "counts_nondecreasing": bool(np.all(np.diff(result["n"]) >= 0)),
                "B_start_target_mT": float(active.B_target_mT.iloc[0]),
                "B_start_sim_mT": float(active.B_sim_mT.iloc[0]),
                "B_end_target_mT": float(active.B_target_mT.iloc[-1]),
                "B_end_sim_mT": float(active.B_sim_mT.iloc[-1]),
                "whole_rms_mT": float(np.sqrt(np.mean(error**2))),
                "whole_max_mT": float(np.max(np.abs(error))),
                "start_30mm_rms_mT": float(np.sqrt(np.mean(error[z <= 30.0] ** 2))),
                "end_30mm_rms_mT": float(np.sqrt(np.mean(error[z >= 155.0] ** 2))),
                "equivalent_detuning_MHz": float(
                    pd.read_csv(source_csv).equivalent_detuning_MHz.iloc[0]
                ),
            }
        ]
    )
    summary_path = output_dir / f"{OUTPUT_PREFIX}_design_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))

    print("\nSaved:")
    for path in (
        *paths,
        *figures,
        individual_path,
        layout_path,
        counts_path,
        accuracy_path,
        summary_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
