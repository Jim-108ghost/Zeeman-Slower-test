# -*- coding: utf-8 -*-
"""Improve the 245 m/s magnet field entrance without degrading later regions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from yb171_tapered_variable_s_magnet_185mm import (
    save_accuracy_summary,
    save_counts_plot,
    save_stack_configuration,
)
from yb171_zeeman_slower_optimizer_15stacks import PermanentMagnetZeemanSlower
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import build_config
from yb171_zeeman_slower_zero_start_reasonable import (
    _refresh_result_metrics,
    individual_magnet_dataframe,
    save_layout_plot,
)


OUTPUT_PREFIX = "Yb171_tapered_average_s_245mps_185mm_entrance_polished"


def regional_metrics(errors_mT, start, middle, end):
    return {
        "start_rms_mT": float(np.sqrt(np.mean(errors_mT[start] ** 2))),
        "start_max_mT": float(np.max(np.abs(errors_mT[start]))),
        "middle_rms_mT": float(np.sqrt(np.mean(errors_mT[middle] ** 2))),
        "middle_max_mT": float(np.max(np.abs(errors_mT[middle]))),
        "end_rms_mT": float(np.sqrt(np.mean(errors_mT[end] ** 2))),
        "end_max_mT": float(np.max(np.abs(errors_mT[end]))),
    }


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    source_csv = project_root / "inputs" / "Yb171_tapered_average_s_245mps_B_vs_z.csv"
    prepared_csv = (
        project_root / "inputs" / "Yb171_tapered_average_s_245mps_B_target_extended.csv"
    )
    base_dir = project_root / "runs" / "12_tapered_average_s_245mps_185mm"
    base_prefix = "Yb171_tapered_average_s_245mps_185mm"
    output_dir = (
        project_root / "runs" / "13_tapered_average_s_245mps_entrance_polished"
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
    optimizer = PermanentMagnetZeemanSlower(cfg)

    base_table = pd.read_csv(base_dir / f"{base_prefix}_magnet_config.csv")
    base_n = base_table.n_magnets_each_side.to_numpy(dtype=int)
    base_d_m = base_table.d_stack_center_mm.to_numpy(dtype=float) * 1e-3
    optimizer.orientation[:] = 1.0
    base_B_T = optimizer.simulated_field_at_z(
        {"n": base_n, "d_m": base_d_m}, optimizer.z_m
    )

    active = (optimizer.z_m >= 0.0) & (optimizer.z_m <= cfg.L_m)
    start = active & (optimizer.z_m <= 30e-3)
    middle = active & (optimizer.z_m > 30e-3) & (optimizer.z_m < 155e-3)
    end = active & (optimizer.z_m >= 155e-3)
    base_errors_mT = (base_B_T - optimizer.B_target_T) * 1e3
    base_metrics = regional_metrics(base_errors_mT, start, middle, end)

    # Best feasible discrete entrance shaper found by enumerating 1--4 reverse
    # pairs at 9.467 mm and 0--4 positive pairs at 22.400 mm.  Counts at all
    # other fixed x positions remain those of run 12.
    n = base_n.copy()
    n[0] = 4
    n[1] = 2
    optimizer.orientation[0] = -1.0
    optimizer.orientation[1:] = 1.0
    if 2 * int(np.sum(n)) > cfg.max_total_magnets:
        raise RuntimeError("Entrance-polished distribution exceeds 200 magnets")

    lower_mm = np.array(
        [optimizer._effective_lower_d_bound(int(count)) for count in n]
    ) * 1e3
    upper_mm = np.full(cfg.n_stacks, cfg.d_max_m * 1e3)
    initial_mm = np.clip(base_d_m * 1e3, lower_mm, upper_mm)
    initial_mm[0] = max(lower_mm[0], 53.5)
    initial_mm[1] = max(lower_mm[1], 34.7)

    def field_and_errors(d_mm):
        B_T = optimizer.simulated_field_at_z(
            {"n": n, "d_m": np.asarray(d_mm) * 1e-3}, optimizer.z_m
        )
        return B_T, (B_T - optimizer.B_target_T) * 1e3

    def objective(d_mm):
        _, errors_mT = field_and_errors(d_mm)
        return float(
            np.mean(errors_mT[start] ** 2)
            + 0.05 * np.mean(errors_mT[middle] ** 2)
            + 0.05 * np.mean(errors_mT[end] ** 2)
        )

    def no_degradation_constraints(d_mm):
        _, errors_mT = field_and_errors(d_mm)
        metrics = regional_metrics(errors_mT, start, middle, end)
        return np.array(
            [
                base_metrics["middle_rms_mT"] - metrics["middle_rms_mT"],
                base_metrics["middle_max_mT"] - metrics["middle_max_mT"],
                base_metrics["end_rms_mT"] - metrics["end_rms_mT"],
                base_metrics["end_max_mT"] - metrics["end_max_mT"],
            ]
        )

    solution = minimize(
        objective,
        initial_mm,
        method="SLSQP",
        bounds=list(zip(lower_mm, upper_mm)),
        constraints={"type": "ineq", "fun": no_degradation_constraints},
        options={"maxiter": 1000, "ftol": 1e-13, "disp": False},
    )
    violation = max(0.0, float(-np.min(no_degradation_constraints(solution.x))))
    if not solution.success or violation > 1e-6:
        raise RuntimeError(
            f"Constrained entrance optimization failed: {solution.message}; "
            f"violation={violation:.6g} mT"
        )

    final_B_T, final_errors_mT = field_and_errors(solution.x)
    final_metrics = regional_metrics(final_errors_mT, start, middle, end)
    result = _refresh_result_metrics(
        optimizer,
        {
            "n": n,
            "d_m": solution.x * 1e-3,
            "B_sim_T": final_B_T,
            "a": np.nan,
            "counts_refined": True,
            "distance_polish_method": "entrance_constrained_slsqp",
        },
    )

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
        title="245 m/s average-s field: entrance-polished 8 mm x 3 mm N35 magnets",
    )
    counts_path = output_dir / f"{OUTPUT_PREFIX}_counts_vs_field.png"
    save_counts_plot(optimizer, result, counts_path)
    accuracy_path = output_dir / f"{OUTPUT_PREFIX}_accuracy_summary.csv"
    save_accuracy_summary(
        optimizer, result, source_csv, prepared_csv, accuracy_path
    )
    comparison = pd.DataFrame(
        [
            {"design": "run12_original", **base_metrics},
            {"design": "run13_entrance_polished", **final_metrics},
        ]
    )
    comparison_path = output_dir / f"{OUTPUT_PREFIX}_regional_comparison.csv"
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
