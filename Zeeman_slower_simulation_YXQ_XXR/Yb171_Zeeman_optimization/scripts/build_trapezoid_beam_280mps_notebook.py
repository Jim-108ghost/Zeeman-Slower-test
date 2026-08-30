# -*- coding: utf-8 -*-
"""Create a separate 280 m/s, finite-force-margin notebook variant."""

import json
from pathlib import Path


SOURCE_NAME = "Zeeman slower_trapezoid_beam_35mW.ipynb"
OUTPUT_NAME = "Zeeman slower_trapezoid_beam_35mW_280mps.ipynb"


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

    cell14 = "".join(notebook["cells"][14]["source"])
    cell14 = replace_once(
        cell14,
        "v_design = 305.3249965316503\n",
        "v_design = 280.0\n",
        "TEST design velocity",
    )
    cell14 = replace_once(
        cell14,
        "Yb171_tapered_variable_s_B_vs_z.csv",
        "Yb171_tapered_variable_s_280mps_magnet_B_vs_z.csv",
        "TEST field CSV",
    )
    set_source(notebook["cells"][14], cell14)

    heading = "".join(notebook["cells"][86]["source"])
    heading = replace_once(
        heading,
        "> Maximum-input-speed design over 185 mm for a 35 mW tapered Gaussian beam: "
        "r(-300 mm)=4 mm, r(+320 mm)=8 mm, v_final=35 m/s.",
        "> Robust 280-to-35 m/s design over 185 mm for a 35 mW tapered Gaussian beam: "
        "r(-300 mm)=4 mm, r(+320 mm)=8 mm, with finite force margin.",
        "section heading",
    )
    set_source(notebook["cells"][86], heading)

    cell87 = r'''# Variable-s robust magnetic-field design: 280 -> 35 m/s
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
rho_design = 0.0
v_initial_design = 280.0  # m/s
v_final = 35.0            # m/s
slower_length = 0.185     # m
delta_laser = 2 * np.pi * (-600) * 1e6  # rad/s
P_L_design = 35e-3        # W
wL_design = trapezoid_beam_radius

# Position-dependent saturation and available scattering force.
z_design = np.linspace(0.0, slower_length, 2001)
beam_radius_design = wL_design(z_design)
s_design_profile = slower.saturation_param(
    z_design, rho_design, P_L_design, wL_design, s_is_const=False
)
s_axis_design = s_design_profile
a_max = hbar * k_399 * Gamma_399 / (2 * m_171)
eta_available = s_design_profile / (1.0 + s_design_profile)
available_integral = cumulative_trapezoid(eta_available, z_design, initial=0.0)
v0_max = np.sqrt(v_final**2 + 2.0 * a_max * available_integral[-1])

# Use one constant fraction of the locally available force.  This preserves
# the variable-s shape and supplies a finite restoring-force/safety margin.
force_scale = (
    v_initial_design**2 - v_final**2
) / (2.0 * a_max * available_integral[-1])
if not 0.0 < force_scale < 1.0:
    raise ValueError(f'No finite force margin: force_scale={force_scale}')
eta_design_profile = force_scale * eta_available
eta_integral = cumulative_trapezoid(eta_design_profile, z_design, initial=0.0)
v_design_profile = np.sqrt(np.maximum(
    v_initial_design**2 - 2.0 * a_max * eta_integral,
    0.0,
))
v0 = v_initial_design

# Red-detuned stable branch: a faster atom moves closer to resonance and feels
# a larger slowing force.  eta = s/[1+s+(2*Delta_eff/Gamma)^2].
detuning_square = np.maximum(
    s_design_profile / eta_design_profile - (1.0 + s_design_profile),
    0.0,
)
delta_eff_design = -0.5 * Gamma_399 * np.sqrt(detuning_square)

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
print('Variable-s tapered Gaussian-beam magnetic-field design')
print('  beam: r(-300 mm)=4 mm -> r(+320 mm)=8 mm; P=35 mW')
print(f'  theoretical zero-margin maximum v0: {v0_max:.3f} m/s')
print(f'  robust design: {v_initial_design:.3f} -> {v_design_profile[-1]:.3f} m/s over {slower_length*1e3:.0f} mm')
print(f'  local-force scale: {force_scale:.6f}; margin={(1-force_scale)*100:.3f}%')
print(f'  radius in slower: {beam_radius_design[0]*1e3:.3f} -> {beam_radius_design[-1]*1e3:.3f} mm')
print(f'  s(x,rho_design): {s_design_profile[0]:.3f} -> {s_design_profile[-1]:.3f}')
print(f'  eta designed: {eta_design_profile[0]:.3f} -> {eta_design_profile[-1]:.3f}')
print(f'  effective detuning: {delta_eff_design[0]/(2*np.pi*1e6):.3f} -> {delta_eff_design[-1]/(2*np.pi*1e6):.3f} MHz')
print(f'  B(0)={B_func(0)*1e3:.3f} mT, B(L)={B_func(slower_length)*1e3:.3f} mT')
print(f'  detuning check max error: {np.max(np.abs(delta_check-delta_eff_design))/(2*np.pi):.3e} Hz')

# 1. Robust magnetic field.
plt.figure(figsize=(7, 5))
plt.plot(z_design * 100, B_design_profile * 1e3, linewidth=2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.7, label='slower start')
plt.axvline(slower_length * 100, color='red', linestyle='--', alpha=0.7, label='slower end')
plt.xlabel('x-position [cm]')
plt.ylabel('Magnetic field [mT]')
plt.title('Yb171 Variable-s Robust Field: 280 to 35 m/s')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Local saturation, available force and designed force.
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()
line1 = ax1.plot(z_design * 100, s_design_profile, 'b-', linewidth=2, label='s(x)')
line2 = ax2.plot(z_design * 100, eta_available, color='0.5', linestyle=':', linewidth=2, label='eta available')
line3 = ax2.plot(z_design * 100, eta_design_profile, 'r--', linewidth=2, label='eta designed')
ax1.set_xlabel('x-position [cm]')
ax1.set_ylabel('Local saturation s', color='b')
ax2.set_ylabel('Acceleration fraction |a|/a_max', color='r')
ax1.grid(True, alpha=0.3)
lines = line1 + line2 + line3
ax1.legend(lines, [line.get_label() for line in lines], loc='best')
plt.title('Position-dependent Saturation and Finite Force Margin')
plt.tight_layout()
plt.show()

# 3. Designed velocity trajectory.
plt.figure(figsize=(7, 5))
plt.plot(z_design * 100, v_design_profile, linewidth=2)
plt.axhline(v_final, color='orange', linestyle='--', label=f'v_final={v_final:.1f} m/s')
plt.xlabel('x-position [cm]')
plt.ylabel('Designed velocity [m/s]')
plt.title('Robust Variable-s Velocity Trajectory')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Export B(x), s(x), force margin, velocity and effective detuning.
from pathlib import Path
csv_output_dir = Path('Yb171_Zeeman_optimization/inputs')
csv_output_dir.mkdir(parents=True, exist_ok=True)
csv_filename = csv_output_dir / 'Yb171_tapered_variable_s_280mps_B_vs_z.csv'
csv_data = np.column_stack((
    z_design,
    z_design * 100,
    B_design_profile,
    B_design_profile * 1e3,
    beam_radius_design * 1e3,
    s_design_profile,
    eta_available,
    np.full_like(z_design, force_scale),
    np.full_like(z_design, 1.0-force_scale),
    eta_design_profile,
    v_design_profile,
    delta_eff_design / (2*np.pi*1e6),
))
np.savetxt(
    csv_filename,
    csv_data,
    delimiter=',',
    header='z_m,z_cm,B_T,B_mT,beam_radius_mm,s_local,eta_available,force_scale,force_margin_fraction,eta_design,v_design_m_s,delta_eff_MHz',
    comments='',
)
print(f'CSV saved: {csv_filename}')
'''
    set_source(notebook["cells"][87], cell87)

    cell89 = "".join(notebook["cells"][89]["source"])
    cell89 = replace_once(
        cell89,
        "v0 = v0_max    # maximum input velocity derived from s(x) over 185 mm\n",
        "v0 = v_initial_design  # robust 280 m/s input-speed design\n",
        "single-atom input velocity",
    )
    set_source(notebook["cells"][89], cell89)

    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
