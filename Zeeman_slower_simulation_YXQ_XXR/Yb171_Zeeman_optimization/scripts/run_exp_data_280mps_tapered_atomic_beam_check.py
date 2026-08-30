# -*- coding: utf-8 -*-
"""Run EXP DATA with the atomic-beam waist following the laser-beam waist."""

from run_exp_data_280mps_check import run_exp_data_check


if __name__ == "__main__":
    run_exp_data_check(
        "Zeeman slower_trapezoid_beam_35mW_280mps_atomic_beam_tapered.ipynb",
        "Yb171_tapered_variable_s_280mps_tapered_atomic_EXP_DATA_summary.csv",
        "Yb171_tapered_variable_s_280mps_tapered_atomic_EXP_DATA_5000atoms.csv",
    )
