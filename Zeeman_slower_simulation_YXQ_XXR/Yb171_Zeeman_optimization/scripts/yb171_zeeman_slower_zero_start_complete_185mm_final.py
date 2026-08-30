# -*- coding: utf-8 -*-
"""Export the selected magnet design for the current single-atom CSV."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import e, hbar, m_e

from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import (
    build_config,
)
from yb171_zeeman_slower_zero_start_reasonable import (
    OrderedCountOptimizer,
    _refresh_result_metrics,
    individual_magnet_dataframe,
    save_accuracy_summary,
    save_layout_plot,
)


OUTPUT_PREFIX = "Yb171_single_atom_complete_185mm"
ACTIVE_LENGTH_M = 0.185
ENTRANCE_CLAMP_END_M = 0.030
DESIGN_DELTA_MHZ = -600.0
DESIGN_INITIAL_VELOCITY_M_S = 230.0
DESIGN_FINAL_VELOCITY_M_S = 35.0

# Counts are for ONE side.  The upper/lower pair therefore contains 2*n_i
# physical magnets.  The sequence is nondecreasing: fewer magnets at the
# low-field entrance and progressively more in the high-field exit region.
FINAL_N_EACH_SIDE = np.array(
    [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7],
    dtype=int,
)
FINAL_D_MM = np.array(
    [
        30.856960203,
        32.393576899,
        27.576302728,
        27.414945605,
        30.144717553,
        30.358736268,
        31.806804505,
        31.111576760,
        30.011590218,
        31.480386619,
        30.566216610,
        32.138308327,
        32.106585432,
        31.423908700,
        27.460978467,
    ],
    dtype=float,
)


def prepare_target(source_csv, output_csv):
    """Copy the current 0--185 mm source and clamp entrance negatives."""
    source = pd.read_csv(source_csv)
    z_source_m = source["z_m"].to_numpy(dtype=float)
    B_source_mT = source["B_mT"].to_numpy(dtype=float)
    good = np.isfinite(z_source_m) & np.isfinite(B_source_mT)
    z_source_m = z_source_m[good]
    B_source_mT = B_source_mT[good]
    order = np.argsort(z_source_m)
    z_source_m = z_source_m[order]
    B_source_mT = B_source_mT[order]

    inside = (z_source_m >= 0.0) & (z_source_m < ACTIVE_LENGTH_M)
    source_endpoint_B_mT = np.interp(
        ACTIVE_LENGTH_M, z_source_m, B_source_mT
    )
    z_target_m = np.append(z_source_m[inside], ACTIVE_LENGTH_M)
    B_original_mT = np.append(
        B_source_mT[inside], source_endpoint_B_mT
    )

    B_target_mT = B_original_mT.copy()
    entrance_negative = (
        (z_target_m <= ENTRANCE_CLAMP_END_M) & (B_target_mT < 0.0)
    )
    B_target_mT[entrance_negative] = 0.0

    prepared = pd.DataFrame(
        {
            "z_m": z_target_m,
            "z_mm": z_target_m * 1e3,
            "B_original_mT": B_original_mT,
            "B_T": B_target_mT * 1e-3,
            "B_mT": B_target_mT,
            "entrance_negative_clamped_to_zero": entrance_negative,
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_csv, index=False)
    return prepared


def build_result(optimizer):
    result = {
        "n": FINAL_N_EACH_SIDE.copy(),
        "d_m": FINAL_D_MM.copy() * 1e-3,
        "a": np.nan,
        "counts_refined": True,
        "distance_polish_method": (
            "weighted_least_squares_then_constrained_minimax"
        ),
        "minimax_bound_mT": 0.212323,
    }
    result["B_sim_T"] = optimizer.simulated_field_at_z(
        result, optimizer.z_m
    )
    return _refresh_result_metrics(optimizer, result)


def add_design_compatibility_to_summary(optimizer, result, summary_path):
    summary = pd.read_csv(summary_path)
    wavelength_m = 398.9e-9
    k = 2 * np.pi / wavelength_m
    mu_eff = -e * hbar / (2 * m_e)
    delta = DESIGN_DELTA_MHZ * 1e6 * 2 * np.pi

    active = (
        (optimizer.z_m >= 0.0)
        & (optimizer.z_m <= optimizer.cfg.L_m)
    )
    active_indices = np.flatnonzero(active)
    B_start_T = result["B_sim_T"][active_indices[0]]
    B_end_T = result["B_sim_T"][active_indices[-1]]

    def resonant_velocity(B_T):
        return (-delta + mu_eff * B_T / hbar) / k

    summary["design_delta_MHz"] = DESIGN_DELTA_MHZ
    summary["design_initial_velocity_m_s"] = DESIGN_INITIAL_VELOCITY_M_S
    summary["design_final_velocity_m_s"] = DESIGN_FINAL_VELOCITY_M_S
    summary["simulated_B_start_mT"] = B_start_T * 1e3
    summary["simulated_B_end_mT"] = B_end_T * 1e3
    summary["resonant_velocity_start_m_s"] = resonant_velocity(B_start_T)
    summary["resonant_velocity_end_m_s"] = resonant_velocity(B_end_T)
    summary.to_csv(summary_path, index=False)


def main():
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent
    workspace_root = project_root.parent
    source_csv = workspace_root / "Yb171_single_atom_B_vs_z.csv"
    prepared_csv = (
        project_root
        / "inputs"
        / "Yb171_single_atom_B_vs_z_current_185mm.csv"
    )
    output_dir = (
        project_root / "runs" / "08_single_atom_complete_185mm"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    target = prepare_target(source_csv, prepared_csv)
    if not np.isclose(float(target.iloc[-1]["z_m"]), ACTIVE_LENGTH_M):
        raise RuntimeError("The target does not end at 185 mm")

    cfg = build_config(scripts_dir)
    cfg.csv_path = str(prepared_csv)
    cfg.L_m = ACTIVE_LENGTH_M
    cfg.target_mode = "crop"
    cfg.fit_left_extension_m = 0.0
    cfg.fit_right_extension_m = 0.0
    cfg.extension_fit_weight = 1.0

    optimizer = OrderedCountOptimizer(cfg)
    result = build_result(optimizer)

    if result["total_magnets"] > 200:
        raise RuntimeError("The final design exceeds 200 magnets")
    if not np.all(np.diff(result["n"]) >= 0):
        raise RuntimeError("The magnet counts are not nondecreasing")
    # 05_final max error was 0.12012 mT; the user allowed about 1 G more.
    if result["max_abs_error_T"] * 1e3 > 0.2202:
        raise RuntimeError("The maximum error exceeds the accepted limit")

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
        title="Current Yb171 single-atom target: 8 mm x 3 mm N35 magnets",
    )

    summary_path = output_dir / f"{OUTPUT_PREFIX}_accuracy_summary.csv"
    save_accuracy_summary(
        optimizer, result, source_csv, prepared_csv, summary_path
    )
    add_design_compatibility_to_summary(
        optimizer, result, summary_path
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
