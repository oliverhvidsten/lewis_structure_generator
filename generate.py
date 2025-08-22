from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdDepictor, Draw, Mol
from rdkit.Geometry import Point3D
import sys
import math
from pathlib import Path

# --- Valence data ---
VALENCE_E = {
    "H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
    "Si": 4, "P": 5, "S": 6, "Cl": 7, "Br": 7, "I": 7
}
# maximum bonds usually formed (expanded octets included)
TARGET_VALENCE = {
    "H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1,
    "Si": 4, "P": 5, "S": 6, "Cl": 1, "Br": 1, "I": 1
}

def read_xyz(path:str) -> tuple[list[str], list[float]]:
    """
    Parse XYZ file-> (symbols, coords)."""
    lines = [ln.strip() for ln in open(path)]
    symbols, coords = [], []
    for ln in lines[2:]:
        sym, x, y, z = ln.split()[:4]
        symbols.append(sym)
        coords.append((float(x), float(y), float(z)))
    return symbols, coords


ALKALI_METALS = {"Li", "Na", "K", "Rb", "Cs", "Fr"}

def mol_from_xyz(symbols, coords, metal_ion_flexibility=False, charge=0):
    """
    Build RDKit Mol from XYZ with optional metal-ion flexibility:
    - Removes alkali metals initially
    - Adds negative charge to molecule
    - Determines bonds and lone pairs
    - Adds metal ions back to closest negative site if flag is True
    """
    # Separate metal ions if flag is set
    metal_atoms, nonmetal_symbols, nonmetal_coords = [], [], []
    for i, sym in enumerate(symbols):
        if metal_ion_flexibility and sym in ALKALI_METALS:
            metal_atoms.append((sym, coords[i]))
        else:
            nonmetal_symbols.append(sym)
            nonmetal_coords.append(coords[i])

    # Adjust molecular charge for removed metals
    adjusted_charge = charge
    if metal_ion_flexibility:
        adjusted_charge -= len(metal_atoms)  # each removed metal +1 → molecule more negative

    # --- Build molecule for nonmetals ---
    rw = Chem.RWMol()
    for sym in nonmetal_symbols:
        rw.AddAtom(Chem.Atom(sym))
    mol = rw.GetMol()

    # Add conformer
    conf = Chem.Conformer(len(nonmetal_symbols))
    for i, (x, y, z) in enumerate(nonmetal_coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)

    atoms = [atom for atom in rw.GetAtoms()]
    total_electrons = sum([atoms[i].GetAtomicNum() for i in range(len(atoms))]) - adjusted_charge
    
    # If we have a radical, remove it and then add it back after the molecule has been determined
    # We do this to make the DetermineConnectivity step more likely to succeed
    increment_charge = False
    if total_electrons % 2 == 1:
        increment_charge = True
        adjusted_charge += 1

    try:
        rdDetermineBonds.DetermineConnectivity(mol, charge=adjusted_charge)
        rdDetermineBonds.DetermineBonds(mol, charge=adjusted_charge)
        Chem.SanitizeMol(mol)
    except Exception:
        # fallback: use distance-based heuristic
        rw = Chem.RWMol(mol)
        rw = heuristic_bond_assignment(rw, nonmetal_coords, target_charge=adjusted_charge)
        mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    # --- Postprocess: add metals and radicals back if relevant ---
    if increment_charge:
        mol = add_nonbonding_electron(mol)

    if metal_ion_flexibility and metal_atoms:
        rw = Chem.RWMol(mol)
        for sym, coord in metal_atoms:
            idx = rw.AddAtom(Chem.Atom(sym))
            # Attach to closest negatively charged atom
            min_dist, closest_idx = float("inf"), None
            pos_new = Point3D(*coord)
            for atom in rw.GetAtoms():
                fc = atom.GetFormalCharge()
                if fc < 0:
                    idx_atom = atom.GetIdx()
                    pos_atom = rw.GetConformer().GetAtomPosition(idx_atom)
                    dist = pos_new.Distance(pos_atom)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx_atom
            if closest_idx is not None:
                rw.AddBond(idx, closest_idx, Chem.BondType.SINGLE)
                rw.GetAtomWithIdx(idx).SetFormalCharge(1)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)

    return mol


def add_nonbonding_electron(mol: Chem.Mol) -> Chem.Mol:
    """
    Adds one non-bonding electron to the molecule:
      - If atoms with positive charge exist, give it to the most electronegative one.
      - Otherwise, assign it to an atom with an unfilled valence shell (needs electrons).
    """
    # Pauling electronegativity values
    EN = {
        "H": 2.20, "Li": 0.98, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
        "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66
    }
    # Maximum valence electrons (approximate, allowing hypervalence for some)
    MAX_VALENCE = {"H": 2, "Li": 2, "C": 8, "N": 8, "O": 8, "F": 8,
                   "P": 10, "S": 12, "Cl": 8, "Br": 8, "I": 8, "B": 6, "Si": 8}

    def bond_order_sum(atom):
        return sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())

    rw = Chem.RWMol(mol)
    atoms = rw.GetAtoms()

    # Case 1: atoms with positive charge
    pos_atoms = [atom for atom in atoms if atom.GetFormalCharge() > 0]
    if pos_atoms:
        target_atom = max(pos_atoms, key=lambda a: EN.get(a.GetSymbol(), 2.5))
        target_atom.SetFormalCharge(target_atom.GetFormalCharge() - 1)
        return rw.GetMol()

    # Case 2: no positive atoms → look for atoms with unfilled valence
    unfilled = []
    for atom in atoms:
        sym = atom.GetSymbol()
        ve_max = MAX_VALENCE.get(sym, 8)
        # Current electron count = bonds*2 + lone pairs*2 + charge adjustment
        bos = bond_order_sum(atom)
        fc = atom.GetFormalCharge()
        # effective valence electrons (simplified Lewis accounting)
        nbe = atom.GetAtomicNum() - fc - bos
        if 2 * bos + nbe < ve_max:
            unfilled.append(atom)

    if unfilled:
        # Give the electron to the most electronegative unfilled atom
        target_atom = max(unfilled, key=lambda a: EN.get(a.GetSymbol(), 2.5))
        target_atom.SetFormalCharge(target_atom.GetFormalCharge() - 1)

    return rw.GetMol()

def heuristic_bond_assignment(rw, coords, target_charge):
    """
    Assign bonds heuristically using interatomic distances and valence rules.
    Adjust formal charges to match overall molecular charge.
    """
    from rdkit.Chem import BondType

    atoms = [atom for atom in rw.GetAtoms()]
    n_atoms = len(atoms)

    # Covalent radii (Å) for simple heuristic
    cov_radii = {
        "H": 0.31, "Li": 1.28, "B": 0.85, "C": 0.76, "N": 0.71, "O": 0.66,
        "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39
    }
    MAX_VALENCE = {"H": 2, "Li": 2, "C": 8, "O": 8, "N": 8, "F": 8,
                   "P": 10, "S": 12, "Cl": 8, "Br": 8, "I": 8, "B": 6, "Si": 8}

    # Step 1: Determine candidate bonds based on distances
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            pos_i = coords[i]
            pos_j = coords[j]
            dist = math.sqrt(sum((pos_i[k]-pos_j[k])**2 for k in range(3)))
            r_sum = cov_radii.get(atoms[i].GetSymbol(), 0.7) + cov_radii.get(atoms[j].GetSymbol(), 0.7) + 0.4
            if dist <= r_sum:
                # Tentatively add single bond
                rw.AddBond(i, j, BondType.SINGLE)

    # Step 2: Compute electrons per atom
    total_electrons = sum([atoms[i].GetAtomicNum() for i in range(n_atoms)]) - target_charge
    atom_electrons = [atoms[i].GetAtomicNum() for i in range(n_atoms)]  # start neutral

    # Step 3: Adjust formal charges to match valence and overall charge
    # Limit atoms to max valence
    for atom in atoms:
        sym = atom.GetSymbol()
        valence = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        max_val = MAX_VALENCE.get(sym, valence)
        # If overfilled, assign positive formal charge
        if valence > max_val:
            atom.SetFormalCharge(valence - max_val)

    # Step 4: Assign extra charge electrons to most electronegative atoms with room
    current_charge = sum([atom.GetFormalCharge() for atom in atoms])
    extra = target_charge - current_charge
    if extra != 0:
        # Sort by electronegativity
        electronegativity = {
            "H": 2.2, "Li": 1.0, "B": 2.0, "C": 2.5, "N": 3.0, "O": 3.5,
            "F": 4.0, "Si": 1.8, "P": 2.1, "S": 2.5, "Cl": 3.0, "Br": 2.8, "I": 2.5
        }
        atoms_sorted = sorted(atoms, key=lambda a: -electronegativity.get(a.GetSymbol(), 2.5))
        if extra > 0:  # need more positive
            for atom in atoms_sorted:
                if extra == 0:
                    break
                valence = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
                max_val = MAX_VALENCE.get(atom.GetSymbol(), valence)
                if valence < max_val:
                    atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                    extra -= 1
        else:  # need more negative
            for atom in atoms_sorted:
                if extra == 0:
                    break
                valence = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
                max_val = MAX_VALENCE.get(atom.GetSymbol(), valence)
                if valence + 2 <= max_val:  # can accept extra lone pair
                    atom.SetFormalCharge(atom.GetFormalCharge() - 1)
                    extra += 1

    return rw



def bond_order_sum(mol:Mol, idx:int):
    return sum(b.GetBondTypeAsDouble() for b in mol.GetAtomWithIdx(idx).GetBonds())

def estimate_lone_pairs(mol:Mol) -> list[tuple[int, int, int]]:
    """Estimate lone pairs per atom (Lewis style, adjusted for hypervalence)."""
    results = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        ve = VALENCE_E.get(sym, 0)
        fc = atom.GetFormalCharge()
        bos = bond_order_sum(mol, atom.GetIdx())

        # nonbonding electrons from Lewis formula
        nbe = ve - fc - bos
        lp = max(0, nbe // 2)
        rad = nbe % 2

        # hypervalent adjustment: if atom exceeds typical valence,
        # redistribute electrons so lone pairs don’t go negative
        max_val = TARGET_VALENCE.get(sym, bos)
        if bos > max_val and lp == 0:
            # allow expanded octet by leaving lp = 0
            lp = 0

        results.append((fc, int(lp), int(rad)))
    return results

def lewis_summary(mol:Mol) -> str:
    lines = ["Atoms (idx, symbol, FC, LP, RAD):"]
    lp_data = estimate_lone_pairs(mol)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        sym = atom.GetSymbol()
        fc, lp, rad= lp_data[i]
        lines.append(f"  {i:>2}: {sym:>2}  FC={fc:+d}  LP={lp}  RAD={rad}")
    lines.append("\nBonds (a1 - a2 : order):")
    seen = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (j, i) in seen: continue
        seen.add((i, j))
        order = b.GetBondTypeAsDouble()
        lines.append(f"  {i:>2} - {j:<2} : {order}")
    return "\n".join(lines)

def write_svg(mol:Mol, out_svg="lewis.svg"):
    # Compute 2D coords for a simple structural diagram (not explicit LPs)
    rdDepictor.Compute2DCoords(mol)
    Draw.MolToFile(mol, out_svg)

def main(xyz_path:str, metal_ion:bool=False):
    symbols, coords = read_xyz(xyz_path)
    mol = mol_from_xyz(symbols, coords, metal_ion, charge=-1)

    # Print a text Lewis summary (bonds, charges, lone pairs)
    print(lewis_summary(mol))

    # Also write an SVG 2D depiction (standard skeletal structure)
    svg_path = Path(xyz_path).with_suffix(".svg")
    write_svg(mol, str(svg_path))
    print(f"\n2D depiction written to: {svg_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lewis_from_xyz.py <xyz_file> [--metal]")
        sys.exit(1)

    xyz_file = sys.argv[1]
    metal_flag = '--metal' in sys.argv

    main(sys.argv[1], metal_flag)
