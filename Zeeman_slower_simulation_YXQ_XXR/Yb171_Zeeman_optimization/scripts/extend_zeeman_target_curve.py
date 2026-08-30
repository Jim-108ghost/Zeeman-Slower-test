# -*- coding: utf-8 -*-
"""Keep the measured target curve and extend only its two outer tails."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial


def main():
    here = Path(__file__).resolve().parent
    input_dir = here.parent / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_path = input_dir / "Yb171_single_atom_B_vs_z.csv"
    output_path = input_dir / "Yb171_target_B_extended.csv"

    source = pd.read_csv(source_path)
    z_source_m = source["z_m"].to_numpy(dtype=float)
    B_source_T = source["B_T"].to_numpy(dtype=float)

    # A global degree-6 polynomial follows the complete 0--185 mm curve with
    # about 0.033 mT RMS error, while extrapolating more stably than a local
    # endpoint cubic at the steep high-field end.
    fitted_curve = Polynomial.fit(
        z_source_m,
        B_source_T,
        deg=6,
    )

    # Keep every original sample.  The dense grid is added only to make the
    # exported CSV convenient for plotting/interpolation.
    z_extended_m = np.unique(
        np.concatenate(
            (
                np.linspace(-13e-3, 198e-3, 1201),
                z_source_m,
                np.array([z_source_m.min(), z_source_m.max()]),
            )
        )
    )

    in_original_range = (
        (z_extended_m >= z_source_m.min())
        & (z_extended_m <= z_source_m.max())
    )
    B_original_interp_T = np.full_like(z_extended_m, np.nan)
    B_original_interp_T[in_original_range] = np.interp(
        z_extended_m[in_original_range],
        z_source_m,
        B_source_T,
    )

    # In 0--185 mm, preserve the original curve exactly (linear interpolation
    # between its 300 samples).  Outside it, use the degree-6 fit's trend but
    # add a constant offset on each side so both joins are value-continuous.
    B_extended_T = np.empty_like(z_extended_m)
    B_extended_T[in_original_range] = B_original_interp_T[in_original_range]
    left_mask = z_extended_m < z_source_m.min()
    right_mask = z_extended_m > z_source_m.max()
    B_extended_T[left_mask] = (
        fitted_curve(z_extended_m[left_mask])
        + B_source_T[0]
        - fitted_curve(z_source_m[0])
    )
    B_extended_T[right_mask] = (
        fitted_curve(z_extended_m[right_mask])
        + B_source_T[-1]
        - fitted_curve(z_source_m[-1])
    )

    fit_on_source_T = fitted_curve(z_source_m)
    fit_rms_mT = float(
        np.sqrt(np.mean((fit_on_source_T - B_source_T) ** 2)) * 1e3
    )

    result = pd.DataFrame(
        {
            "z_m": z_extended_m,
            "z_mm": z_extended_m * 1e3,
            "B_T": B_extended_T,
            "B_mT": B_extended_T * 1e3,
            "in_original_range": in_original_range,
            "B_original_interp_mT": B_original_interp_T * 1e3,
            "B_degree6_unanchored_mT": fitted_curve(z_extended_m) * 1e3,
        }
    )
    result.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Extended range: {z_extended_m[0]*1e3:.3f} ... {z_extended_m[-1]*1e3:.3f} mm")
    print(f"Degree-6 fit RMS on original data: {fit_rms_mT:.6f} mT")


if __name__ == "__main__":
    main()
