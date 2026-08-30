# -*- coding: utf-8 -*-
"""Create a 280 m/s notebook with a self-similarly tapered atomic beam."""

import json
from pathlib import Path


SOURCE_NAME = "Zeeman slower_trapezoid_beam_35mW_280mps.ipynb"
OUTPUT_NAME = "Zeeman slower_trapezoid_beam_35mW_280mps_atomic_beam_tapered.ipynb"


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def main():
    workspace = Path(__file__).resolve().parents[2]
    source_path = workspace / SOURCE_NAME
    output_path = workspace / OUTPUT_NAME
    notebook = json.loads(source_path.read_text(encoding="utf-8"))

    # Cell 5: optionally scale every atom's rho with a callable atomic-beam radius.
    cell5 = "".join(notebook["cells"][5]["source"])
    cell5 = replace_once(
        cell5,
        "    def simulate_slower(self, z0, rho0, v0, delta_laser, B_func, P_L, wL, slower_length, s_is_const = True, t_max=0.05, n_points=1000):\n",
        "    def simulate_slower(self, z0, rho0, v0, delta_laser, B_func, P_L, wL, slower_length, s_is_const=True, atomic_beam_radius_func=None, t_max=0.05, n_points=1000):\n",
        "simulate_slower signature",
    )
    cell5 = replace_once(
        cell5,
        """        self.exit_velocity = -1

        def motion_equation(t, state):
            z, v = state
            s_val = self.saturation_param(z, rho0, P_L, wL, s_is_const = s_is_const)
""",
        """        self.exit_velocity = -1

        if callable(atomic_beam_radius_func):
            atomic_radius_at_start = float(atomic_beam_radius_func(z0))
            if atomic_radius_at_start <= 0:
                raise ValueError("Atomic-beam radius at z0 must be positive.")

            def rho_at_z(z_value):
                return rho0 * np.asarray(
                    atomic_beam_radius_func(z_value), dtype=float
                ) / atomic_radius_at_start
        else:
            def rho_at_z(z_value):
                return np.zeros_like(np.asarray(z_value), dtype=float) + rho0

        def motion_equation(t, state):
            z, v = state
            rho_local = rho_at_z(z)
            s_val = self.saturation_param(z, rho_local, P_L, wL, s_is_const=s_is_const)
""",
        "rho scaling in motion equation",
    )
    cell5 = replace_once(
        cell5,
        """        # store histories along the returned trajectory only
        force_list = []
""",
        """        # Recompute rho from the final returned z grid, including event points.
        rho_actual = np.asarray(rho_at_z(z_actual), dtype=float)

        # store histories along the returned trajectory only
        force_list = []
""",
        "returned rho profile",
    )
    cell5 = replace_once(
        cell5,
        """        for ti, zi, vi in zip(t_actual, z_actual, v_actual):
            s_val = self.saturation_param(zi, rho0, P_L, wL, s_is_const = s_is_const)
""",
        """        for ti, zi, vi, rhoi in zip(t_actual, z_actual, v_actual, rho_actual):
            s_val = self.saturation_param(zi, rhoi, P_L, wL, s_is_const=s_is_const)
""",
        "rho history",
    )
    set_source(notebook["cells"][5], cell5)

    # Cell 12: sample the atomic distribution using the atomic-beam entrance radius.
    cell12 = "".join(notebook["cells"][12]["source"])
    cell12 = replace_once(
        cell12,
        "w_a = 8e-3   # atomic beam waist 5mm / diameter 10mm\n",
        """atomic_beam_radius = trapezoid_beam_radius
w_a_start = atomic_beam_radius(0.0)
w_a_end = atomic_beam_radius(slower_length)
w_a = w_a_start  # initial radial sampling uses the entrance atomic-beam radius
""",
        "atomic beam waist",
    )
    cell12 = replace_once(
        cell12,
        "print(f\"  Atom Beam waist: w_a = {w_a*1000:.1f} mm \\n\")\n",
        """print(f"  Atomic-beam radius: {w_a_start*1000:.3f} mm at x=0"
      f" -> {w_a_end*1000:.3f} mm at x={slower_length*1000:.0f} mm\\n")
""",
        "atomic beam report",
    )
    set_source(notebook["cells"][12], cell12)

    # Cell 14: apply the same atomic-beam radius function throughout each trajectory.
    cell14 = "".join(notebook["cells"][14]["source"])
    cell14 = replace_once(
        cell14,
        """                P_L=P_L, wL=wL,
                s_is_const=False,  # use local radius and each atom's rho0
                slower_length=slower_length,
""",
        """                P_L=P_L, wL=wL,
                s_is_const=False,
                atomic_beam_radius_func=atomic_beam_radius,
                slower_length=slower_length,
""",
        "EXP DATA atomic beam function",
    )
    set_source(notebook["cells"][14], cell14)

    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
