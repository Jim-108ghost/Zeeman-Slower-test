# Yb171 Zeeman-slower optimization files

This directory groups the files generated during the 185 mm Zeeman-slower
magnet optimization.

## Directory layout

- `inputs/`: ideal/source field data and the extended target curve.
- `scripts/`: curve-extension and magnet-optimization programs.
- `runs/01_15stacks_unrestricted/`: original unrestricted magnet-count run.
- `runs/02_max200/`: first run constrained to at most 200 magnets.
- `runs/03_max200_startopt/`: entrance-weighted run.
- `runs/04_max200_startopt_w30/`: stronger entrance-weight test run.
- `runs/05_max200_startopt_final/`: selected final 200-magnet solution.
- `runs/06_single_atom_target_final/`: the same-accuracy 200-magnet array
  validated directly against `Yb171_single_atom_B_vs_z.csv`, including an
  individual coordinate row for every physical disk magnet.
- `runs/07_single_atom_zero_start_reasonable/`: the root-level single-atom
  target with negative entrance samples clamped to zero, `B_target(0)=0`, and
  a physically ordered 96-magnet layout (fewer magnets in lower-field regions).
  This run includes individual magnet coordinates and the simulated field out
  to 400 mm.
- `runs/08_single_atom_complete_185mm/`: selected result for the current
  root-level `Yb171_single_atom_B_vs_z.csv`. It uses 98 magnets with counts
  increasing from the low-field entrance toward the high-field exit.

The field currently imported by `Zeeman slower.ipynb` is:

`runs/08_single_atom_complete_185mm/Yb171_single_atom_complete_185mm_field_to_400mm.csv`

Each run directory keeps its magnet configuration, field CSV files, full-field
plot, residual plot, and entrance/end detail plots together.

The run-08 field is matched to `v_design=230 m/s`, `v_final=35 m/s`,
`delta=-600 MHz`, and `slower_length=185 mm` in `Zeeman slower.ipynb`.
