"""Build the tapered-beam notebook from the untouched browser backup."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE_NAME = "Zeeman slower_browser_backup.ipynb"
OUTPUT_NAME = "Zeeman slower_trapezoid_beam_35mW.ipynb"
EXPECTED_CELL_COUNT = 140
MODIFIED_CELLS = {5, 14, 86, 87, 89}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def set_source(cell: dict, source: str) -> None:
    cell["source"] = source.splitlines(keepends=True)


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    source_path = workspace / SOURCE_NAME
    output_path = workspace / OUTPUT_NAME

    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    if len(notebook["cells"]) != EXPECTED_CELL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CELL_COUNT} source cells, found {len(notebook['cells'])}"
        )
    original_sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]

    # Cell 5: extend the existing saturation calculation without changing scalar-wL behavior.
    cell_index = 5
    source = original_sources[cell_index]
    helper = '''def trapezoid_beam_radius(x, x_left=-0.300, r_left=4e-3, x_right=0.320, r_right=8e-3):
    """Linear 1/e^2 laser radius; x is in m and the returned radius is in m."""
    x_array = np.asarray(x, dtype=float)
    x_clipped = np.clip(x_array, x_left, x_right)
    radius = r_left + (r_right - r_left) * (x_clipped - x_left) / (x_right - x_left)
    return float(radius) if x_array.ndim == 0 else radius


'''
    source = replace_once(source, "class ZeemanSlower:\n", helper + "class ZeemanSlower:\n", "cell 5 helper")
    old_saturation = '''        I_sat = (np.pi * self.h * self.c * self.gamma) / (3 * self.wavelength**3)
        z_R = np.pi * wL**2 / self.wavelength  # rayleigh length
        w_z = wL * np.sqrt(1 + (z / z_R)**2)
        I_0 = 2 * P_L / (np.pi * wL**2)
        I_rho_z = I_0 * (wL / w_z)**2 * np.exp(-2 * rho**2 / w_z**2)
        s = I_rho_z / I_sat
        if s_is_const:
            s = 0.5* I_0 / I_sat # take average to estimate overall slowing profermance
            # print(s)
        return s
'''
    new_saturation = '''        I_sat = (np.pi * self.h * self.c * self.gamma) / (3 * self.wavelength**3)

        if callable(wL):
            # wL(x) is the local 1/e^2 radius of the tapered beam.
            w_z = np.asarray(wL(z), dtype=float)
            if np.any(w_z <= 0):
                raise ValueError("The laser-beam radius must be positive.")
            I_axis_z = 2 * P_L / (np.pi * w_z**2)
            I_rho_z = I_axis_z * np.exp(-2 * np.asarray(rho)**2 / w_z**2)
            if s_is_const:
                s = 0.5 * I_axis_z / I_sat
            else:
                s = I_rho_z / I_sat
        else:
            # Preserve the original scalar-waist behavior in unrelated sections.
            z_R = np.pi * wL**2 / self.wavelength
            w_z = wL * np.sqrt(1 + (np.asarray(z) / z_R)**2)
            I_0 = 2 * P_L / (np.pi * wL**2)
            I_rho_z = I_0 * (wL / w_z)**2 * np.exp(-2 * np.asarray(rho)**2 / w_z**2)
            s = I_rho_z / I_sat
            if s_is_const:
                s = 0.5 * I_0 / I_sat

        return float(s) if np.ndim(s) == 0 else s
'''
    source = replace_once(source, old_saturation, new_saturation, "cell 5 saturation")
    set_source(notebook["cells"][cell_index], source)

    # Cell 14: TEST/Exp-data calculation with 35 mW and the local tapered-beam intensity.
    cell_index = 14
    source = original_sources[cell_index]
    source = replace_once(
        source,
        '''delta_laser = -600 * 1e6 * 2 * np.pi
wL = 8e-3
slower_length = 0.185
''',
        '''delta_laser = -600 * 1e6 * 2 * np.pi
slower_length = 0.185

# 35 mW beam; it converges along the laser propagation direction (+x -> -x).
wL = trapezoid_beam_radius
''',
        "cell 14 beam",
    )
    old_intensity = '''I_sat_399 = 60 # mW/cm^2
I_list = 2*np.array(P_L_list)* 1e3 / (pi * (wL * 1e2)**2)  # convert to mW/cm^2
s_list = I_list / I_sat_399
print("Laser power (mW) and corresponding saturation parameter s:")
for P_L, s in zip(P_L_list, s_list):
    print(f"  P_L = {P_L*1e3:.2f} mW -> s = {s:.2f}")
'''
    new_intensity = '''I_sat_399_SI = (np.pi * slower.h * slower.c * slower.gamma) / (3 * slower.wavelength**3)
beam_radius_start = wL(0.0)
beam_radius_end = wL(slower_length)
P_L_array = np.asarray(P_L_list, dtype=float)
s_start_list = 2 * P_L_array / (np.pi * beam_radius_start**2) / I_sat_399_SI
s_end_list = 2 * P_L_array / (np.pi * beam_radius_end**2) / I_sat_399_SI
s_list = s_end_list  # conservative plot label: on-axis s at the exit
print("35 mW tapered laser beam:")
print(f"  radius: {beam_radius_start*1e3:.3f} mm at x=0 -> "
      f"{beam_radius_end*1e3:.3f} mm at x={slower_length*1e3:.0f} mm")
for P_L, s_start, s_end in zip(P_L_list, s_start_list, s_end_list):
    print(f"  P_L={P_L*1e3:.2f} mW: on-axis s={s_start:.3f} -> {s_end:.3f}")

variable_s_columns = {'s_local', 'eta_design', 'v_design_m_s', 'delta_eff_MHz'}
if variable_s_columns.issubset(field_data.columns):
    print("Variable-s CSV metadata:")
    print(f"  designed velocity: {field_data['v_design_m_s'].iloc[0]:.3f} -> "
          f"{field_data['v_design_m_s'].iloc[-1]:.3f} m/s")
    print(f"  local s: {field_data['s_local'].iloc[0]:.3f} -> "
          f"{field_data['s_local'].iloc[-1]:.3f}")
    print(f"  designed eta: {field_data['eta_design'].iloc[0]:.3f} -> "
          f"{field_data['eta_design'].iloc[-1]:.3f}")
    print(f"  effective detuning: {field_data['delta_eff_MHz'].iloc[0]:.3f} -> "
          f"{field_data['delta_eff_MHz'].iloc[-1]:.3f} MHz")
else:
    a_required = abs((v_final**2 - v_design**2) / (2 * slower_length))
    a_max = hbar * slower.k * slower.gamma / (2 * slower.mass)
    eta_required = a_required / a_max
    s_required = eta_required / (1 - eta_required)
    print(f"Constant-deceleration reference saturation: s >= {s_required:.3f}")
'''
    source = replace_once(source, old_intensity, new_intensity, "cell 14 intensity")
    source = replace_once(
        source,
        "                s_is_const=True,\n",
        "                s_is_const=False,  # use local radius and each atom's rho0\n",
        "cell 14 local intensity",
    )
    source = replace_once(
        source,
        "v_design = 210.0\n",
        "v_design = 305.3249965316503\n",
        "cell 14 design velocity",
    )
    source = replace_once(
        source,
        "'Yb171_Zeeman_optimization/inputs/Yb171_single_atom_B_vs_z.csv'\n",
        "'Yb171_Zeeman_optimization/inputs/Yb171_tapered_variable_s_B_vs_z.csv'\n",
        "cell 14 variable-s field path",
    )
    source = replace_once(
        source,
        'f"CSV - ideal field error: RMS={np.sqrt(np.mean(field_error_mT**2)):.6f} mT, "',
        'f"CSV - constant-deceleration reference: RMS={np.sqrt(np.mean(field_error_mT**2)):.6f} mT, "',
        "cell 14 reference label",
    )
    source = replace_once(
        source,
        'f"Resonant velocity: {float(resonant_velocity_from_field(0.0)):.2f}"',
        'f"Zero-effective-detuning velocity (not the variable-s trajectory): "\n'
        '        f"{float(resonant_velocity_from_field(0.0)):.2f}"',
        "cell 14 zero-detuning label",
    )
    set_source(notebook["cells"][cell_index], source)

    # Cell 86: make the modified location obvious in Jupyter's left outline.
    cell_index = 86
    source = original_sources[cell_index]
    if not source.startswith("## 3. calculate the magnetic field"):
        raise RuntimeError("Cell 86 is not the expected magnetic-field heading")
    source += (
        "\n\n> Maximum-input-speed design over 185 mm for a 35 mW tapered Gaussian beam: "
        "r(-300 mm)=4 mm, r(+320 mm)=8 mm, v_final=35 m/s."
    )
    set_source(notebook["cells"][cell_index], source)

    # Cell 87: design B(x) from the position-dependent tapered-beam saturation.
    cell_index = 87
    source = '''# Variable-s magnetic-field design for a tapered Gaussian beam
from scipy.integrate import cumulative_trapezoid

# Yb171 parameters
au = 1.66053906660e-27  # kg
m_171 = 171 * au
Gamma_399 = 2 * np.pi * 28.9e6  # rad/s
lmd_399 = 398.9e-9
k_399 = 2 * np.pi / lmd_399

g_e, g_g = 2/3, 0.0
M_e, M_g = -3/2, 1/2
mu_B = e * hbar / (2 * m_e)
mu_eff_Yb = (g_e * M_e - g_g * M_g) * mu_B

slower = ZeemanSlower(Gamma_399, k_399, m_171, mu_eff_Yb, lmd_399)

# Design inputs
z0 = 0.0
rho0 = 0.0
rho_design = 0.0       # reference atom: on the laser axis
v_final = 35.0         # m/s
slower_length = 0.185  # m
delta_laser = 2 * np.pi * (-600) * 1e6  # rad/s
P_L_design = 35e-3     # W
wL_design = trapezoid_beam_radius

# Local saturation and maximum local force for the tapered Gaussian beam.
z_design = np.linspace(0.0, slower_length, 2001)
beam_radius_design = wL_design(z_design)
s_design_profile = slower.saturation_param(
    z_design, rho_design, P_L_design, wL_design, s_is_const=False
)
s_axis_design = s_design_profile  # compatibility alias
a_max = hbar * k_399 * Gamma_399 / (2 * m_171)
eta_available = s_design_profile / (1.0 + s_design_profile)
eta_integral = cumulative_trapezoid(eta_available, z_design, initial=0.0)

# The maximum capturable input speed uses the full resonant scattering force.
# Integrate backward from the requested final speed over the fixed 185 mm length.
eta_design_profile = eta_available
v0_max = np.sqrt(v_final**2 + 2.0 * a_max * eta_integral[-1])
v0 = v0_max  # compatibility with the single-atom check below
v_design_profile = np.sqrt(
    np.maximum(v0_max**2 - 2.0 * a_max * eta_integral, 0.0)
)

# At the theoretical maximum input speed the reference atom is exactly resonant,
# so Delta_eff=0 everywhere.  This is a maximum-force design with no safety margin.
delta_eff_design = np.zeros_like(z_design)

# Zeeman resonance condition for the variable-force velocity trajectory.
B_design_profile = (hbar / mu_eff_Yb) * (
    delta_laser + k_399 * v_design_profile - delta_eff_design
)

def B_func(z):
    z_array = np.asarray(z, dtype=float)
    B = np.interp(z_array, z_design, B_design_profile)
    return float(B) if z_array.ndim == 0 else B

delta_check = (
    delta_laser + k_399 * v_design_profile
    - mu_eff_Yb * B_design_profile / hbar
)
print("Variable-s tapered Gaussian-beam magnetic-field design")
print("  beam: r(-300 mm)=4 mm -> r(+320 mm)=8 mm; P=35 mW")
print(f"  maximum input velocity: {v0_max:.3f} m/s")
print(f"  slower: {v0_max:.3f} -> {v_design_profile[-1]:.3f} m/s over {slower_length*1e3:.0f} mm")
print(f"  radius in slower: {beam_radius_design[0]*1e3:.3f} -> {beam_radius_design[-1]*1e3:.3f} mm")
print(f"  s(x,rho_design): {s_design_profile[0]:.3f} -> {s_design_profile[-1]:.3f}")
print(f"  eta(x): {eta_design_profile[0]:.3f} -> {eta_design_profile[-1]:.3f}")
print("  force fraction of local maximum: 1.000 (theoretical maximum-speed design)")
print("  effective detuning: 0 MHz (exact resonance; zero safety margin)")
print(f"  B(0)={B_func(0)*1e3:.3f} mT, B(L)={B_func(slower_length)*1e3:.3f} mT")
print(f"  detuning check max error: {np.max(np.abs(delta_check-delta_eff_design))/(2*np.pi):.3e} Hz")

# 1. Magnetic field calculated from the variable-s velocity trajectory.
plt.figure(figsize=(7, 5))
plt.plot(z_design * 100, B_design_profile * 1e3, linewidth=2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.7, label='slower start')
plt.axvline(slower_length * 100, color='red', linestyle='--', alpha=0.7, label='slower end')
plt.xlabel('x-position [cm]')
plt.ylabel('Magnetic field [mT]')
plt.title('Yb171 Variable-s Magnetic Field for a Tapered Gaussian Beam')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2. The saturation and acceleration fraction are both position dependent.
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()
line1 = ax1.plot(z_design * 100, s_design_profile, 'b-', linewidth=2, label='s(x, rho_design)')
line2 = ax2.plot(z_design * 100, eta_design_profile, 'r--', linewidth=2, label='eta(x)=|a|/a_max')
ax1.set_xlabel('x-position [cm]')
ax1.set_ylabel('Local saturation s', color='b')
ax2.set_ylabel('Designed acceleration fraction eta', color='r')
ax1.grid(True, alpha=0.3)
lines = line1 + line2
ax1.legend(lines, [line.get_label() for line in lines], loc='best')
plt.title('Position-dependent Saturation and Deceleration')
plt.tight_layout()
plt.show()

# 3. Designed velocity trajectory.
plt.figure(figsize=(7, 5))
plt.plot(z_design * 100, v_design_profile, linewidth=2)
plt.axhline(v_final, color='orange', linestyle='--', label=f'v_final={v_final:.1f} m/s')
plt.xlabel('x-position [cm]')
plt.ylabel('Designed velocity [m/s]')
plt.title('Velocity Trajectory from Variable Local Scattering Force')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Export B(x), s(x), eta(x), velocity and effective detuning.
from pathlib import Path
csv_output_dir = Path('Yb171_Zeeman_optimization/inputs')
csv_output_dir.mkdir(parents=True, exist_ok=True)
csv_filename = csv_output_dir / 'Yb171_tapered_variable_s_B_vs_z.csv'
csv_data = np.column_stack((
    z_design,
    z_design * 100,
    B_design_profile,
    B_design_profile * 1e3,
    beam_radius_design * 1e3,
    s_design_profile,
    eta_design_profile,
    v_design_profile,
    delta_eff_design / (2*np.pi*1e6),
))
np.savetxt(
    csv_filename,
    csv_data,
    delimiter=',',
    header='z_m,z_cm,B_T,B_mT,beam_radius_mm,s_local,eta_design,v_design_m_s,delta_eff_MHz',
    comments='',
)
print(f'CSV saved: {csv_filename}')
'''
    set_source(notebook["cells"][cell_index], source)

    # Cell 89: single-atom calculation directly under the design section.
    cell_index = 89
    source = original_sources[cell_index]
    source = replace_once(
        source,
        "P_L = 25e-3     # laser power [W]\nwL = 5e-3     # laser waist [m]\n",
        "P_L = P_L_design\nwL = wL_design\n",
        "cell 89 beam",
    )
    source = replace_once(
        source,
        "v0 = 300      # initial velocity m/s\ndelta_laser = 2 * np.pi * (-810) * 1e6 # rad/s\n",
        "v0 = v0_max    # maximum input velocity derived from s(x) over 185 mm\n"
        "delta_laser = 2 * np.pi * (-600) * 1e6  # same detuning as the design\n",
        "cell 89 design parameters",
    )
    source = replace_once(
        source,
        "t, z, v, rho = slower.simulate_slower(z0, rho0, v0, delta_laser, B_func, P_L, wL, slower_length, t_max=0.02)\n",
        '''t, z, v, rho = slower.simulate_slower(
    z0, rho0, v0, delta_laser, B_func, P_L, wL, slower_length,
    s_is_const=False, t_max=0.02
)
''',
        "cell 89 simulation",
    )
    source = replace_once(
        source,
        "s_traj = slower.saturation_param(z, rho, P_L, wL)\n",
        "s_traj = slower.saturation_param(z, rho, P_L, wL, s_is_const=False)\n",
        "cell 89 force",
    )
    set_source(notebook["cells"][cell_index], source)

    if len(notebook["cells"]) != EXPECTED_CELL_COUNT:
        raise RuntimeError("Cell count changed during generation")
    for index, (before, cell) in enumerate(zip(original_sources, notebook["cells"])):
        after = "".join(cell.get("source", []))
        if before != after and index not in MODIFIED_CELLS:
            raise RuntimeError(f"Unexpected source change in cell {index}")

    # Do not display stale results copied from the browser backup in edited code cells.
    for index in MODIFIED_CELLS:
        cell = notebook["cells"][index]
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Created: {output_path}")
    print(f"Cells: {len(notebook['cells'])}; modified: {sorted(MODIFIED_CELLS)}")


if __name__ == "__main__":
    main()
