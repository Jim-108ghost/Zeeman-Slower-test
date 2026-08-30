# -*- coding: utf-8 -*-
"""Fit the tapered-beam 185 mm target with <=200 small permanent magnets.

The target is the variable-s magnetic field generated for the 35 mW tapered
Gaussian beam.  The 15 longitudinal positions are fixed by the user's table.
Only the number of 8 mm x 3 mm N35 disks in each upper/lower stack and the
stack-centre distance from the atomic-beam axis are optimized.

Because this is a spin-flip slower, the target magnitude decreases towards
zero and then increases.  Magnet counts therefore follow the same V-shaped
trend; the magnetic moments reverse at the zero crossing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yb171_zeeman_slower_optimizer_15stacks import (
    PermanentMagnetZeemanSlower,
)
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import (
    build_config,
)
from yb171_zeeman_slower_zero_start_reasonable import (
    balanced_distance_polish,
    individual_magnet_dataframe,
    save_layout_plot,
)


ACTIVE_LENGTH_M = 0.185
EXTENSION_M = 0.013
OUTPUT_PREFIX = "Yb171_tapered_variable_s_185mm"


def prepare_extended_target(source_csv, output_csv):
    """Preserve 0--185 mm data and add smooth 13 mm endpoint extensions."""
    source = pd.read_csv(source_csv)
    z_m = source["z_m"].to_numpy(dtype=float)
    B_T = source["B_T"].to_numpy(dtype=float)
    good = np.isfinite(z_m) & np.isfinite(B_T)
    z_m, B_T = z_m[good], B_T[good]
    order = np.argsort(z_m)
    z_m, B_T = z_m[order], B_T[order]
    z_m, unique_indices = np.unique(z_m, return_index=True)
    B_T = B_T[unique_indices]

    active = (z_m >= 0.0) & (z_m <= ACTIVE_LENGTH_M)
    z_active_m, B_active_T = z_m[active], B_T[active]
    if z_active_m.size < 20:
        raise ValueError("The variable-s target contains too few active samples")
    if not np.isclose(z_active_m[0], 0.0, atol=1e-12):
        raise ValueError("The variable-s target must start at z=0")
    if not np.isclose(z_active_m[-1], ACTIVE_LENGTH_M, atol=1e-12):
        raise ValueError("The variable-s target must end at z=185 mm")

    # Cubic local fits retain the measured/designed active samples exactly and
    # supply only the short outside trend used to control endpoint fringe fields.
    edge_width_m = 30e-3
    left_fit = z_active_m <= edge_width_m
    right_fit = z_active_m >= ACTIVE_LENGTH_M - edge_width_m
    left_poly = np.poly1d(np.polyfit(z_active_m[left_fit], B_active_T[left_fit], 3))
    right_poly = np.poly1d(np.polyfit(z_active_m[right_fit], B_active_T[right_fit], 3))

    z_left_m = np.linspace(-EXTENSION_M, 0.0, 142, endpoint=False)
    z_right_m = np.linspace(
        ACTIVE_LENGTH_M,
        ACTIVE_LENGTH_M + EXTENSION_M,
        142,
    )[1:]
    prepared = pd.DataFrame(
        {
            "z_m": np.concatenate((z_left_m, z_active_m, z_right_m)),
            "B_T": np.concatenate(
            ((left_poly(z_left_m), B_active_T, right_poly(z_right_m)))
            ),
            "target_region": (
                ["left_cubic_extension"] * z_left_m.size
                + ["original_variable_s_data"] * z_active_m.size
                + ["right_cubic_extension"] * z_right_m.size
            ),
        }
    )
    prepared["z_mm"] = prepared["z_m"] * 1e3
    prepared["B_mT"] = prepared["B_T"] * 1e3
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_csv, index=False)
    return prepared


class MagnitudeOrderedOptimizer(PermanentMagnetZeemanSlower):
    """Keep integer counts ordered in the direction of increasing |B|."""

    def _counts_are_reasonable(self, n):
        n = np.asarray(n, dtype=int)
        if np.any(n < 0):
            return False
        count_change = np.diff(n)
        magnitude_change = np.diff(np.abs(self.B_target_at_stack_T))
        decreasing = magnitude_change < -1e-12
        increasing = magnitude_change > 1e-12
        return bool(
            np.all(count_change[decreasing] <= 0)
            and np.all(count_change[increasing] >= 0)
        )

    def _refine_counts_and_distances(self, initial):
        if not self._counts_are_reasonable(initial["n"]):
            raise ValueError("Initial counts do not follow the target |B| trend")

        current = {
            "loss": float(initial["loss"]),
            "n": initial["n"].copy(),
            "d_m": initial["d_m"].copy(),
            "B_sim_T": initial["B_sim_T"].copy(),
        }
        cache = {tuple(current["n"].tolist()): current}

        def solve_counts(n_try):
            n_try = np.asarray(n_try, dtype=int)
            if not self._counts_are_reasonable(n_try):
                return None
            key = tuple(n_try.tolist())
            if key not in cache:
                cache[key] = self._optimize_d_for_fixed_counts(n_try)
            return cache[key]

        # Include short block moves so equal-count plateaus can move while the
        # physically intuitive V-shaped ordering remains intact.
        for count_step in self.cfg.count_refinement_steps:
            count_step = int(count_step)
            for _ in range(self.cfg.max_count_refinement_sweeps):
                trials = []
                for first in range(self.cfg.n_stacks):
                    for width in (1, 2, 3):
                        last = first + width
                        if last > self.cfg.n_stacks:
                            continue
                        for delta in (-count_step, count_step):
                            trial = current["n"].copy()
                            trial[first:last] += delta
                            trials.append(trial)

                best = current
                for trial in trials:
                    candidate = solve_counts(trial)
                    if candidate is not None and candidate["loss"] < best["loss"] - 1e-20:
                        best = candidate
                if best["loss"] >= current["loss"] - 1e-20:
                    break
                current = {
                    "loss": float(best["loss"]),
                    "n": best["n"].copy(),
                    "d_m": best["d_m"].copy(),
                    "B_sim_T": best["B_sim_T"].copy(),
                }
        return current


def save_stack_configuration(optimizer, result, output_path):
    table = optimizer.result_dataframe(result)
    target_at_stack_T = np.interp(
        optimizer.x_stack_m, optimizer.z_m, optimizer.B_target_T
    )
    simulated_at_stack_T = optimizer.simulated_field_at_z(
        result, optimizer.x_stack_m
    )
    table["target_B_mT"] = target_at_stack_T * 1e3
    table["simulated_B_mT"] = simulated_at_stack_T * 1e3
    table["residual_mT"] = (simulated_at_stack_T - target_at_stack_T) * 1e3
    table["magnetization_axis"] = np.where(
        table["orientation"].to_numpy() > 0, "+z", "-z"
    )
    table.to_csv(output_path, index=False)
    return table


def save_accuracy_summary(optimizer, result, source_csv, prepared_csv, output_path):
    residual_mT = (result["B_sim_T"] - optimizer.B_target_T) * 1e3
    active = (optimizer.z_m >= 0.0) & (optimizer.z_m <= ACTIVE_LENGTH_M)
    at_start = int(np.argmin(np.abs(optimizer.z_m)))
    at_end = int(np.argmin(np.abs(optimizer.z_m - ACTIVE_LENGTH_M)))
    magnitude_change = np.diff(np.abs(optimizer.B_target_at_stack_T))
    count_change = np.diff(result["n"])
    follows_magnitude = bool(
        np.all(count_change[magnitude_change < -1e-12] <= 0)
        and np.all(count_change[magnitude_change > 1e-12] >= 0)
    )
    summary = pd.DataFrame(
        [
            {
                "source_target_csv": str(source_csv),
                "prepared_target_csv": str(prepared_csv),
                "active_length_mm": ACTIVE_LENGTH_M * 1e3,
                "magnet_type": "N35 disk 8 mm diameter x 3 mm thickness",
                "fixed_longitudinal_stacks": optimizer.cfg.n_stacks,
                "total_physical_magnets": int(result["total_magnets"]),
                "total_magnet_limit": int(optimizer.cfg.max_total_magnets),
                "counts_follow_target_abs_field": follows_magnitude,
                "negative_orientation_stacks": int(np.sum(optimizer.orientation < 0)),
                "positive_orientation_stacks": int(np.sum(optimizer.orientation > 0)),
                "target_B_start_mT": optimizer.B_target_T[at_start] * 1e3,
                "simulated_B_start_mT": result["B_sim_T"][at_start] * 1e3,
                "target_B_end_mT": optimizer.B_target_T[at_end] * 1e3,
                "simulated_B_end_mT": result["B_sim_T"][at_end] * 1e3,
                "active_rms_error_mT": float(np.sqrt(np.mean(residual_mT[active] ** 2))),
                "active_mae_mT": float(np.mean(np.abs(residual_mT[active]))),
                "active_max_abs_error_mT": float(np.max(np.abs(residual_mT[active]))),
                "start_30mm_rms_error_mT": result["start_30mm_rms_error_T"] * 1e3,
                "start_30mm_max_abs_error_mT": result["start_30mm_max_abs_error_T"] * 1e3,
                "end_30mm_rms_error_mT": result["end_30mm_rms_error_T"] * 1e3,
                "end_30mm_max_abs_error_mT": result["end_30mm_max_abs_error_T"] * 1e3,
                "distance_polish_method": result.get("distance_polish_method", "base_optimizer"),
            }
        ]
    )
    summary.to_csv(output_path, index=False)
    return summary


def save_counts_plot(optimizer, result, output_path):
    target_stack_mT = optimizer.B_target_at_stack_T * 1e3
    fig, axis_B = plt.subplots(figsize=(10, 5))
    axis_n = axis_B.twinx()
    axis_B.plot(
        optimizer.x_stack_m * 1e3,
        target_stack_mT,
        "o-",
        color="tab:blue",
        label="target B at stack",
    )
    axis_n.step(
        optimizer.x_stack_m * 1e3,
        result["n"],
        where="mid",
        color="tab:orange",
        linewidth=2,
        label="magnets per side",
    )
    axis_B.axhline(0.0, color="0.4", linewidth=0.8)
    axis_B.set_xlabel("Fixed longitudinal position x (mm)")
    axis_B.set_ylabel("Target field (mT)", color="tab:blue")
    axis_n.set_ylabel("Number of magnets on each side", color="tab:orange")
    axis_n.set_ylim(bottom=0)
    handles = axis_B.get_lines() + axis_n.get_lines()
    axis_B.legend(handles, [h.get_label() for h in handles], loc="upper left")
    axis_B.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    source_csv = project_root / "inputs" / "Yb171_tapered_variable_s_B_vs_z.csv"
    prepared_csv = (
        project_root / "inputs" / "Yb171_tapered_variable_s_B_target_extended.csv"
    )
    output_dir = project_root / "runs" / "09_tapered_variable_s_185mm"
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
        title="Variable-s spin-flip target: individual 8 mm x 3 mm N35 magnets",
    )
    counts_path = output_dir / f"{OUTPUT_PREFIX}_counts_vs_field.png"
    save_counts_plot(optimizer, result, counts_path)
    summary_path = output_dir / f"{OUTPUT_PREFIX}_accuracy_summary.csv"
    save_accuracy_summary(
        optimizer, result, source_csv, prepared_csv, summary_path
    )

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
