# -*- coding: utf-8 -*-
"""Run the fixed-position Yb171 Zeeman-slower optimization with <=200 magnets."""

from pathlib import Path

import numpy as np

from yb171_zeeman_slower_optimizer_15stacks import (
    PermanentMagnetZeemanSlower,
    ZeemanSlowerConfig,
)


def main():
    here = Path(__file__).resolve().parent
    project_root = here.parent
    cfg = ZeemanSlowerConfig(
        csv_path=str(project_root / "inputs" / "Yb171_target_B_extended.csv"),
        L_m=0.185,
        n_stacks=15,
        max_total_magnets=200,
        x_stack_positions_m=tuple(
            np.array(
                [
                    9.467, 22.400, 35.333, 48.267, 61.200,
                    74.133, 87.067, 100.000, 112.933, 125.867,
                    138.800, 151.733, 164.667, 177.600, 190.533,
                ]
            )
            * 1e-3
        ),
        fit_left_extension_m=13e-3,
        fit_right_extension_m=13e-3,
        extension_fit_weight=0.02,
        extension_fit_decay_m=1.0,
        start_fit_region_m=30e-3,
        start_fit_weight=20.0,
        end_fit_region_m=30e-3,
        end_fit_weight=20.0,
        max_abs_error_weight=0.05,
        n_sample=641,
        Br_T=1.2,
        magnet_diameter_m=8e-3,
        magnet_thickness_m=3e-3,
        d0_m=50e-3,
        d_min_m=20e-3,
        d_max_m=100e-3,
        target_mode="crop",
        stack_model="resolved",
        tube_outer_diameter_m=16e-3,
        tube_clearance_m=0.0,
        a_min=3.0,
        a_max=20.0,
        a_step=0.05,
        d_steps_m=(2e-3, 1e-3, 0.5e-3, 0.25e-3, 0.1e-3),
    )

    optimizer = PermanentMagnetZeemanSlower(cfg)
    result = optimizer.optimize(show_progress=True)
    optimizer.print_summary(result)

    prefix = "Yb171_zeeman_15stacks_max200"
    output_dir = project_root / "runs" / "02_max200"
    paths = optimizer.save_results(result, output_dir=output_dir, prefix=prefix)
    figures = optimizer.plot_results(
        result,
        output_dir=output_dir,
        prefix=prefix,
        show=True,
    )

    print("\nSaved:")
    for path in (*paths, *figures):
        print(path)


if __name__ == "__main__":
    main()
