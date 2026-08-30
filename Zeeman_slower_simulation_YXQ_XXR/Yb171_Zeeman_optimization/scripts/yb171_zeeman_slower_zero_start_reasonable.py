# -*- coding: utf-8 -*-
"""Optimize a zero-start, physically ordered small-magnet distribution.

The source target is the root-level Yb171_single_atom_B_vs_z.csv requested by
the user.  Negative samples in the entrance 0--30 mm are clamped to zero and
the target at exactly x=0 is forced to zero.  Magnet counts are constrained to
be nondecreasing along the 15 fixed longitudinal positions, so lower-field
regions cannot contain more magnets than higher-field regions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.optimize import least_squares, minimize

from yb171_zeeman_slower_optimizer_15stacks import (
    PermanentMagnetZeemanSlower,
)
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import (
    build_config,
)


OUTPUT_PREFIX = "Yb171_single_atom_zero_start_reasonable"
ACTIVE_LENGTH_M = 0.185
ENTRANCE_CLAMP_END_M = 0.030


def prepare_target(source_csv, output_csv):
    """Create the exact 0--185 mm target used by this optimization."""
    source = pd.read_csv(source_csv)
    z_m = source["z_m"].to_numpy(dtype=float)
    original_B_mT = source["B_mT"].to_numpy(dtype=float)

    order = np.argsort(z_m)
    z_m = z_m[order]
    original_B_mT = original_B_mT[order]

    keep = z_m < ACTIVE_LENGTH_M
    z_out_m = z_m[keep]
    original_out_mT = original_B_mT[keep]

    # Always include the physical endpoint exactly, rather than relying on a
    # neighbouring source sample.
    if z_out_m.size == 0 or not np.isclose(z_out_m[-1], ACTIVE_LENGTH_M):
        z_out_m = np.append(z_out_m, ACTIVE_LENGTH_M)
        original_out_mT = np.append(
            original_out_mT,
            np.interp(ACTIVE_LENGTH_M, z_m, original_B_mT),
        )

    target_B_mT = original_out_mT.copy()
    entrance_negative = (
        (z_out_m >= 0.0)
        & (z_out_m <= ENTRANCE_CLAMP_END_M)
        & (target_B_mT < 0.0)
    )
    target_B_mT[entrance_negative] = 0.0
    target_B_mT[np.isclose(z_out_m, 0.0, atol=1e-15)] = 0.0

    prepared = pd.DataFrame(
        {
            "z_m": z_out_m,
            "z_mm": z_out_m * 1e3,
            "B_original_mT": original_out_mT,
            "B_T": target_B_mT * 1e-3,
            "B_mT": target_B_mT,
            "entrance_negative_clamped_to_zero": entrance_negative,
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_csv, index=False)
    return prepared


class OrderedCountOptimizer(PermanentMagnetZeemanSlower):
    """Keep magnet counts ordered while refining their integer values."""

    @staticmethod
    def _counts_are_reasonable(n):
        n = np.asarray(n, dtype=int)
        return bool(np.all(n >= 0) and np.all(np.diff(n) >= 0))

    def _refine_counts_and_distances(self, initial):
        if not self._counts_are_reasonable(initial["n"]):
            raise ValueError("Initial magnet counts are not nondecreasing")

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

        # Single-stack changes retain an intuitive monotonic count profile.
        # Block moves let an equal-count plateau move without becoming trapped
        # by the monotonic constraint.
        for count_step in self.cfg.count_refinement_steps:
            count_step = int(count_step)
            for _ in range(self.cfg.max_count_refinement_sweeps):
                candidates_n = []
                for i in range(self.cfg.n_stacks):
                    for delta in (-count_step, count_step):
                        trial = current["n"].copy()
                        trial[i] += delta
                        candidates_n.append(trial)

                # Also adjust short neighbouring blocks. This is useful when
                # counts such as [4, 4, 4] should all change together.
                for width in (2, 3):
                    for first in range(self.cfg.n_stacks - width + 1):
                        last = first + width
                        for delta in (-count_step, count_step):
                            trial = current["n"].copy()
                            trial[first:last] += delta
                            candidates_n.append(trial)

                best = current
                for trial in candidates_n:
                    candidate = solve_counts(trial)
                    if candidate is None:
                        continue
                    if candidate["loss"] < best["loss"] - 1e-20:
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


def _refresh_result_metrics(optimizer, result):
    """Recompute all reported metrics after an additional distance polish."""
    updated = dict(result)
    residual = updated["B_sim_T"] - optimizer.B_target_T
    z_m = optimizer.z_m
    active = (z_m >= 0.0) & (z_m <= optimizer.cfg.L_m)
    start = active & (z_m <= min(30e-3, optimizer.cfg.L_m))
    end = active & (z_m >= max(0.0, optimizer.cfg.L_m - 30e-3))
    middle = active & ~start & ~end

    updated.update(
        {
            "loss": optimizer._loss(updated["B_sim_T"]),
            "fit_domain_rms_error_T": float(
                np.sqrt(np.mean(residual**2))
            ),
            "rms_error_T": float(np.sqrt(np.mean(residual[active] ** 2))),
            "mae_T": float(np.mean(np.abs(residual[active]))),
            "max_abs_error_T": float(np.max(np.abs(residual[active]))),
            "start_30mm_rms_error_T": float(
                np.sqrt(np.mean(residual[start] ** 2))
            ),
            "start_30mm_max_abs_error_T": float(
                np.max(np.abs(residual[start]))
            ),
            "middle_rms_error_T": float(
                np.sqrt(np.mean(residual[middle] ** 2))
            ),
            "end_30mm_rms_error_T": float(
                np.sqrt(np.mean(residual[end] ** 2))
            ),
            "end_30mm_max_abs_error_T": float(
                np.max(np.abs(residual[end]))
            ),
            "total_magnets": int(2 * np.sum(updated["n"])),
        }
    )
    return updated


def balanced_distance_polish(optimizer, result):
    """Reach the 05_final RMS while explicitly controlling the worst point.

    Stage 1 is a weighted nonlinear least-squares solve. Stage 2 introduces an
    auxiliary maximum-error variable and uses constrained SLSQP to balance
    whole-curve RMS against the largest absolute residual. Distances are
    optimized in millimetres for numerical scaling; magnet counts never change.
    """
    n = np.asarray(result["n"], dtype=int)
    d_m = np.asarray(result["d_m"], dtype=float).copy()
    moving = np.flatnonzero(n > 0)
    active = (
        (optimizer.z_m >= 0.0)
        & (optimizer.z_m <= optimizer.cfg.L_m)
    )
    z_active_m = optimizer.z_m[active]

    lower_m = np.array(
        [optimizer._effective_lower_d_bound(int(n[i])) for i in moving]
    )
    upper_m = np.full(moving.size, optimizer.cfg.d_max_m)

    def field_from_active_distances(d_active_m):
        d_trial_m = d_m.copy()
        d_trial_m[moving] = d_active_m
        B_sim_T = optimizer.simulated_field_at_z(
            {"n": n, "d_m": d_trial_m}, optimizer.z_m
        )
        return B_sim_T, d_trial_m

    weights = np.ones(z_active_m.size, dtype=float)
    weights[z_active_m <= 30e-3] = optimizer.cfg.start_fit_weight
    weights[
        z_active_m >= optimizer.cfg.L_m - 30e-3
    ] = optimizer.cfg.end_fit_weight

    def weighted_residual(d_active_m):
        B_sim_T, _ = field_from_active_distances(d_active_m)
        return (
            (B_sim_T[active] - optimizer.B_target_T[active])
            * np.sqrt(weights)
            * 1e3
        )

    least_squares_result = least_squares(
        weighted_residual,
        d_m[moving],
        bounds=(lower_m, upper_m),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
        x_scale="jac",
    )

    # SLSQP uses millimetres here so its finite differences are well scaled.
    d_start_mm = least_squares_result.x * 1e3
    lower_mm = lower_m * 1e3
    upper_mm = upper_m * 1e3

    def errors_mT(d_active_mm):
        B_sim_T, d_trial_m = field_from_active_distances(
            np.asarray(d_active_mm) * 1e-3
        )
        errors = (B_sim_T[active] - optimizer.B_target_T[active]) * 1e3
        return errors, B_sim_T, d_trial_m

    initial_errors_mT, _, _ = errors_mT(d_start_mm)
    x0 = np.append(d_start_mm, np.max(np.abs(initial_errors_mT)) + 1e-5)
    start_edge = z_active_m <= 30e-3
    end_edge = z_active_m >= optimizer.cfg.L_m - 30e-3

    def objective(x):
        errors, _, _ = errors_mT(x[:-1])
        edge_errors = np.concatenate(
            (errors[start_edge], errors[end_edge])
        )
        max_error_mT = x[-1]
        return (
            np.mean(errors**2)
            + 0.2 * np.mean(edge_errors**2)
            + 0.1 * max_error_mT**2
        )

    def maximum_error_constraints(x):
        errors, _, _ = errors_mT(x[:-1])
        max_error_mT = x[-1]
        return np.concatenate(
            (max_error_mT - errors, max_error_mT + errors)
        )

    minimax_result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[*zip(lower_mm, upper_mm), (0.0, 5.0)],
        constraints={"type": "ineq", "fun": maximum_error_constraints},
        options={"maxiter": 600, "ftol": 1e-13, "disp": False},
    )
    final_errors_mT, final_B_sim_T, final_d_m = errors_mT(
        minimax_result.x[:-1]
    )
    constraint_violation_mT = max(
        0.0,
        float(np.max(np.abs(final_errors_mT)) - minimax_result.x[-1]),
    )
    if not minimax_result.success or constraint_violation_mT > 1e-6:
        raise RuntimeError(
            "Balanced minimax distance polish failed: "
            f"{minimax_result.message}; constraint violation "
            f"{constraint_violation_mT:.6g} mT"
        )

    polished = dict(result)
    polished.update(
        {
            "d_m": final_d_m,
            "B_sim_T": final_B_sim_T,
            "distance_polish_method": (
                "weighted_least_squares_then_constrained_minimax"
            ),
            "minimax_bound_mT": float(minimax_result.x[-1]),
        }
    )
    return _refresh_result_metrics(optimizer, polished)


def individual_magnet_dataframe(optimizer, result):
    """Return the physical centre coordinates of every disk magnet."""
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
        positive_centres_m = d_m + offsets_m

        for side_name, side_sign in (("upper", 1), ("lower", -1)):
            for local_i, positive_centre_m in enumerate(
                positive_centres_m, start=1
            ):
                rows.append(
                    {
                        "magnet_id": magnet_id,
                        "stack_index": stack_i,
                        "stack_side": side_name,
                        "magnet_index_on_side": local_i,
                        "longitudinal_x_mm": x_m * 1e3,
                        "transverse_z_mm": side_sign
                        * positive_centre_m
                        * 1e3,
                        "distance_to_axis_mm": positive_centre_m * 1e3,
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


def save_layout_plot(
    individual_df,
    output_path,
    tube_outer_diameter_mm,
    title="Zero-start target: individual 8 mm x 3 mm N35 magnets",
):
    """Draw a longitudinal/transverse view of every physical magnet."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colours = {"upper": "tab:red", "lower": "tab:blue"}

    for row in individual_df.itertuples(index=False):
        ax.add_patch(
            Rectangle(
                (
                    row.longitudinal_x_mm - 0.5 * row.magnet_diameter_mm,
                    row.transverse_z_mm - 0.5 * row.magnet_thickness_mm,
                ),
                row.magnet_diameter_mm,
                row.magnet_thickness_mm,
                facecolor=colours[row.stack_side],
                edgecolor="black",
                linewidth=0.25,
                alpha=0.8,
            )
        )

    tube_radius_mm = 0.5 * tube_outer_diameter_mm
    ax.axhline(0.0, color="black", linewidth=1.0, label="atomic-beam axis")
    ax.axhline(
        tube_radius_mm,
        color="0.5",
        linestyle="--",
        linewidth=1.0,
        label="tube outer surface",
    )
    ax.axhline(-tube_radius_mm, color="0.5", linestyle="--", linewidth=1.0)
    ax.scatter([], [], marker="s", color=colours["upper"], label="upper")
    ax.scatter([], [], marker="s", color=colours["lower"], label="lower")
    ax.set_xlim(-5.0, 200.0)
    max_distance = individual_df["distance_to_axis_mm"].max() + 8.0
    ax.set_ylim(-max_distance, max_distance)
    ax.set_xlabel("Longitudinal position x (mm)")
    ax.set_ylabel("Transverse magnet-centre position (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_accuracy_summary(
    optimizer, result, source_csv, prepared_csv, output_path
):
    z_m = optimizer.z_m
    at_zero = int(np.argmin(np.abs(z_m)))
    table = optimizer.result_dataframe(result)
    pd.DataFrame(
        [
            {
                "source_target_csv": str(source_csv),
                "prepared_target_csv": str(prepared_csv),
                "target_B_at_0mm_mT": (
                    optimizer.B_target_T[at_zero] * 1e3
                ),
                "simulated_B_at_0mm_mT": result["B_sim_T"][at_zero] * 1e3,
                "active_length_mm": optimizer.cfg.L_m * 1e3,
                "total_magnets": result["total_magnets"],
                "distance_polish_method": result.get(
                    "distance_polish_method", "base_optimizer"
                ),
                "minimax_bound_mT": result.get("minimax_bound_mT", np.nan),
                "counts_nondecreasing": bool(
                    np.all(np.diff(result["n"]) >= 0)
                ),
                "first_stack_magnets_each_side": int(result["n"][0]),
                "last_stack_magnets_each_side": int(result["n"][-1]),
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
                "middle_rms_error_mT": result["middle_rms_error_T"] * 1e3,
                "end_30mm_rms_error_mT": (
                    result["end_30mm_rms_error_T"] * 1e3
                ),
                "end_30mm_max_abs_error_mT": (
                    result["end_30mm_max_abs_error_T"] * 1e3
                ),
                "min_count_each_side": int(table.n_magnets_each_side.min()),
                "max_count_each_side": int(table.n_magnets_each_side.max()),
            }
        ]
    ).to_csv(output_path, index=False)


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    workspace_root = project_root.parent
    source_csv = workspace_root / "Yb171_single_atom_B_vs_z.csv"
    prepared_csv = (
        project_root
        / "inputs"
        / "Yb171_single_atom_B_vs_z_zero_start_0_185mm.csv"
    )
    output_dir = project_root / "runs" / "07_single_atom_zero_start_reasonable"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_target(source_csv, prepared_csv)
    if float(prepared.loc[prepared["z_m"].idxmin(), "B_mT"]) != 0.0:
        raise RuntimeError("Prepared target does not start at exactly zero")

    cfg = build_config(scripts_dir)
    cfg.csv_path = str(prepared_csv)
    cfg.L_m = ACTIVE_LENGTH_M
    cfg.fit_left_extension_m = 0.0
    cfg.fit_right_extension_m = 0.0
    cfg.extension_fit_weight = 1.0
    cfg.a_min = 4.0
    cfg.a_max = 20.0
    cfg.a_step = 0.05
    cfg.refine_integer_counts = True
    cfg.count_refinement_steps = (2, 1)
    cfg.max_count_refinement_sweeps = 5
    cfg.continuous_distance_polish = True
    cfg.continuous_polish_maxiter = 1000

    optimizer = OrderedCountOptimizer(cfg)
    result = optimizer.optimize(show_progress=True)
    result = balanced_distance_polish(optimizer, result)

    if result["total_magnets"] > 200:
        raise RuntimeError("The optimized design exceeds 200 magnets")
    if not np.all(np.diff(result["n"]) >= 0):
        raise RuntimeError("The optimized magnet counts are not ordered")
    if result["rms_error_T"] * 1e3 > 0.045:
        raise RuntimeError(
            "The final RMS error is worse than the 05_final reference"
        )
    if result["max_abs_error_T"] * 1e3 > 0.2202:
        raise RuntimeError(
            "The final maximum error exceeds 05_final plus the allowed 1 G"
        )

    optimizer.print_summary(result)
    paths = optimizer.save_results(
        result, output_dir=output_dir, prefix=OUTPUT_PREFIX
    )
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
    save_accuracy_summary(
        optimizer, result, source_csv, prepared_csv, summary_path
    )

    print("\nSaved:")
    for path in (
        prepared_csv,
        *paths,
        *figures,
        individual_path,
        layout_path,
        summary_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
