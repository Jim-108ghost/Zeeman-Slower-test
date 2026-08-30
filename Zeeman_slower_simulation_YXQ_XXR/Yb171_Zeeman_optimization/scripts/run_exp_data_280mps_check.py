# -*- coding: utf-8 -*-
"""Run the notebook's EXP DATA cells reproducibly for the run-11 magnet field."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


RANDOM_SEED = 171280


def run_exp_data_check(notebook_name, summary_name, atoms_name):
    workspace = Path(__file__).resolve().parents[2]
    notebook_path = workspace / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    environment = {"__name__": "exp_data_validation"}

    def execute_cell(index):
        source = "".join(notebook["cells"][index]["source"])
        exec(compile(source, f"notebook_cell_{index}", "exec"), environment)

    execute_cell(3)
    execute_cell(5)
    np = environment["np"]
    environment["plt"].show = lambda *args, **kwargs: environment["plt"].close("all")

    # Same Yb-171 constants used by the notebook before the 5000-atom cells.
    atomic_mass_unit = 1.66053906660e-27
    environment.update(
        {
            "m_171": 171.0 * atomic_mass_unit,
            "Gamma_399": 2.0 * np.pi * 28.9e6,
            "lmd_399": 398.9e-9,
            "k_399": 2.0 * np.pi / 398.9e-9,
            "slower_length": 0.185,
        }
    )
    environment["mu_B"] = (
        environment["e"] * environment["hbar"] / (2.0 * environment["m_e"])
    )
    environment["mu_eff_Yb"] = -environment["mu_B"]

    np.random.seed(RANDOM_SEED)
    execute_cell(11)
    execute_cell(12)
    execute_cell(14)

    initial_velocity = np.asarray(environment["initial_velocities"], dtype=float)
    radial_position = np.asarray(environment["rho_0_array"], dtype=float)
    final_velocity = np.asarray(environment["final_velocities"], dtype=float)
    success = np.asarray(environment["successful_slowing"], dtype=bool)
    design_window = (initial_velocity > 0.0) & (initial_velocity <= 280.0)
    initially_below_MOT = (initial_velocity > 0.0) & (initial_velocity <= 50.0)
    newly_slowed = success & ~initially_below_MOT

    pd = environment["pd"]
    summary = pd.DataFrame(
        [
            {
                "random_seed": RANDOM_SEED,
                "n_samples": initial_velocity.size,
                "temperature_K": environment["T"],
                "atomic_beam_waist_mm": environment["w_a"] * 1e3,
                "atomic_beam_exit_waist_mm": float(
                    environment.get("w_a_end", environment["w_a"])
                ) * 1e3,
                "laser_power_mW": environment["P_L_list"][0] * 1e3,
                "design_input_velocity_m_s": 280.0,
                "MOT_capture_velocity_m_s": environment["v_c_MOT"],
                "simulation_errors": len(environment["simulation_errors"]),
                "all_atoms_success_count": int(np.sum(success)),
                "all_atoms_success_percent": float(np.mean(success) * 100.0),
                "design_window_atom_count": int(np.sum(design_window)),
                "design_window_fraction_percent": float(np.mean(design_window) * 100.0),
                "design_window_success_count": int(np.sum(success & design_window)),
                "design_window_success_percent": float(
                    np.mean(success[design_window]) * 100.0
                ),
                "initially_below_MOT_count": int(np.sum(initially_below_MOT)),
                "newly_slowed_count": int(np.sum(newly_slowed)),
                "newly_slowed_percent_of_all": float(np.mean(newly_slowed) * 100.0),
                "mean_initial_velocity_m_s": float(np.mean(initial_velocity)),
                "mean_final_velocity_m_s": float(np.mean(final_velocity)),
            }
        ]
    )
    atom_table = pd.DataFrame(
        {
            "atom_index": np.arange(initial_velocity.size),
            "initial_velocity_m_s": initial_velocity,
            "rho0_mm": radial_position * 1e3,
            "final_velocity_m_s": final_velocity,
            "in_0_to_280mps_design_window": design_window,
            "initially_below_50mps": initially_below_MOT,
            "success_final_0_to_50mps": success,
        }
    )

    output_dir = (
        workspace
        / "Yb171_Zeeman_optimization"
        / "runs"
        / "11_tapered_variable_s_280mps_185mm"
    )
    summary_path = output_dir / summary_name
    atoms_path = output_dir / atoms_name
    summary.to_csv(summary_path, index=False)
    atom_table.to_csv(atoms_path, index=False)
    print("\nReproducible EXP DATA summary")
    print(summary.to_string(index=False))
    print(summary_path)
    print(atoms_path)
    return summary, atom_table, summary_path, atoms_path


def main():
    run_exp_data_check(
        "Zeeman slower_trapezoid_beam_35mW_280mps.ipynb",
        "Yb171_tapered_variable_s_280mps_EXP_DATA_summary.csv",
        "Yb171_tapered_variable_s_280mps_EXP_DATA_5000atoms.csv",
    )


if __name__ == "__main__":
    main()
