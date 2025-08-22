# lewis_structure_generator

Takes in atomic position information (XYZ format) and outputs an rdkit molecule with information to construct a lewis structure (formal charge, bond orders, non-bonding electrons)
It will also save a .svg file representing the lewis structure of the molecule that was constructed.

Usage:
generate.py file.xyz [--metal]

- specify the `--metal` flag if there are metal ions in the system to allow the code to properly handle the bond assignment
