# -*- coding: utf-8 -*-
"""Shift the 210 m/s field to zero entrance and compute equivalent detuning."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parent.parent
    source_path = project_root / "inputs" / "Yb171_ideal_B_field.csv"
    extended_source_path = project_root / "inputs" / "Yb171_target_B_extended.csv"
    output_path = project_root / "inputs" / "Yb171_210mps_8mm_zero_offset_B.csv"
    extended_output_path = (
        project_root / "inputs" / "Yb171_210mps_8mm_zero_offset_B_extended.csv"
    )

    source = pd.read_csv(source_path)
    offset_T = float(source.B_T.iloc[0])
    source["B_original_T"] = source.B_T
    source["B_original_mT"] = source.B_mT
    source["B_T"] = source.B_T - offset_T
    source["B_mT"] = source.B_T * 1e3

    hbar = 1.054571817e-34
    h = 2.0 * np.pi * hbar
    elementary_charge = 1.602176634e-19
    electron_mass = 9.1093837139e-31
    mu_eff = -elementary_charge * hbar / (2.0 * electron_mass)
    original_detuning_Hz = -600e6
    shifted_detuning_Hz = original_detuning_Hz - mu_eff * offset_T / h
    source["original_detuning_MHz"] = original_detuning_Hz / 1e6
    source["equivalent_detuning_MHz"] = shifted_detuning_Hz / 1e6
    source["field_offset_removed_mT"] = offset_T * 1e3
    source.to_csv(output_path, index=False)

    extended = pd.read_csv(extended_source_path)
    extended["B_original_T"] = extended.B_T
    extended["B_original_mT"] = extended.B_mT
    extended["B_T"] = extended.B_T - offset_T
    extended["B_mT"] = extended.B_T * 1e3
    extended["equivalent_detuning_MHz"] = shifted_detuning_Hz / 1e6
    extended["field_offset_removed_mT"] = offset_T * 1e3
    extended.to_csv(extended_output_path, index=False)

    print(f"removed field offset : {offset_T*1e3:.9f} mT")
    print(f"equivalent detuning  : {shifted_detuning_Hz/1e6:.9f} MHz")
    print(f"shifted B            : {source.B_mT.iloc[0]:.9f} -> {source.B_mT.iloc[-1]:.9f} mT")
    print(output_path)
    print(extended_output_path)


if __name__ == "__main__":
    main()
