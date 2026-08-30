# -*- coding: utf-8 -*-
"""Export the selected <=200-magnet entrance-optimized solution."""

from pathlib import Path

import numpy as np

from yb171_zeeman_slower_optimizer_15stacks import (
    PermanentMagnetZeemanSlower,
)
from yb171_zeeman_slower_optimizer_15stacks_max200_startopt import (
    build_config,
)


# Integer counts were selected by constrained enumeration followed by
# one-magnet transfer neighbourhood search. Distances were polished jointly
# with bounded nonlinear least squares. n_i is the count on ONE side.
FINAL_N_EACH_SIDE = np.array(
    [23, 14, 2, 1, 2, 1, 5, 6, 7, 7, 8, 9, 8, 3, 4],
    dtype=int,
)
FINAL_D_MM = np.array(
    [
        75.783130291,
        83.610287062,
        32.478721987,
        28.260928101,
        28.625700735,
        26.232077910,
        36.410581116,
        41.768274720,
        40.940676974,
        42.163503459,
        40.229087257,
        41.272484611,
        34.889596618,
        23.922778992,
        21.519945285,
    ]
)
OUTPUT_PREFIX = "Yb171_zeeman_15stacks_max200_startopt_final"


def build_result(optimizer):
    n = FINAL_N_EACH_SIDE.copy()
    d_m = FINAL_D_MM * 1e-3
    B_sim_T = optimizer.simulated_field_at_z(
        {"n": n, "d_m": d_m}, optimizer.z_m
    )
    residual = B_sim_T - optimizer.B_target_T
    z_m = optimizer.z_m
    active = (z_m >= 0.0) & (z_m <= optimizer.cfg.L_m)
    start = active & (z_m <= 30e-3)
    end = active & (z_m >= optimizer.cfg.L_m - 30e-3)
    middle = active & ~start & ~end

    return {
        "loss": optimizer._loss(B_sim_T),
        "n": n,
        "d_m": d_m,
        "B_sim_T": B_sim_T,
        "a": np.nan,
        "counts_refined": True,
        "fit_domain_rms_error_T": float(np.sqrt(np.mean(residual**2))),
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
        "total_magnets": int(2 * np.sum(n)),
    }


def main():
    here = Path(__file__).resolve().parent
    output_dir = here.parent / "runs" / "05_max200_startopt_final"
    optimizer = PermanentMagnetZeemanSlower(build_config(here))
    result = build_result(optimizer)
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
    print("\nSaved:")
    for path in (*paths, *figures):
        print(path)


if __name__ == "__main__":
    main()
