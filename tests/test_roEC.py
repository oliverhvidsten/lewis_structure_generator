import sys
sys.path.append('../')

import pytest
from rdkit import Chem
from generate import read_xyz, mol_from_xyz, bond_order_sum, estimate_lone_pairs, lewis_summary

XYZ_PATH = "noLi.xyz"  # Replace with your actual XYZ file

@pytest.fixture
def symbols_coords():
    return read_xyz(XYZ_PATH)

@pytest.fixture
def mol(symbols_coords):
    symbols, coords = symbols_coords
    return mol_from_xyz(symbols, coords, charge=-1)

def test_read_xyz(symbols_coords):
    symbols, coords = symbols_coords
    assert len(symbols) == 10
    assert len(coords) == 10
    # Optional: check first and last symbols
    assert symbols[0] == "O"
    assert symbols[-1] == "H"

def test_mol_from_xyz(mol):
    assert isinstance(mol, Chem.Mol)
    assert mol.GetNumAtoms() == 10

def test_bond_order_sum(mol):
    # Expected bond orders for each atom (example for your molecule)
    # Adjust indices to match your XYZ file
    expected_bo = {
        0: 2.0,  # O 
        1: 4.0,  # C 
        2: 1.0,  # O 
        3: 2.0,  # O 
        4: 4.0,  # C
        5: 3.0,  # terminal CH2 radical
        6: 1.0,  # H
        7: 1.0,  # H
        8: 1.0,  # H
        9: 1.0   # H
    }
    for idx, expected in expected_bo.items():
        bo = bond_order_sum(mol, idx)
        assert abs(bo - expected) < 0.1, f"Atom {idx} bond order incorrect: {bo} != {expected}"

def test_estimate_lone_pairs(mol):
    # Expected (LP, RAD) for each atom
    expected_lp_rad = {
        0: (0, 2, 0),  # O
        1: (0, 0, 0),  # C
        2: (-1, 3, 0),  # O
        3: (0, 2, 0),  # O
        4: (0, 0, 0),  # C
        5: (0, 0, 1),  # terminal CH2 radical
        6: (0, 0, 0),  # H
        7: (0, 0, 0),  # H
        8: (0, 0, 0),  # H
        9: (0, 0, 0)   # H
    }
    lp_data = estimate_lone_pairs(mol)
    assert len(lp_data) == 10
    for idx, (fc, lp, rad) in enumerate(lp_data):
        expected_fc, expected_lp, expected_rad = expected_lp_rad[idx]
        assert fc == expected_fc, f"Atom {idx} FC incorrect: {fc} != {expected_fc}"
        assert lp == expected_lp, f"Atom {idx} LP incorrect: {lp} != {expected_lp}"
        assert rad == expected_rad, f"Atom {idx} RAD incorrect: {rad} != {expected_rad}"

def test_formal_charges(mol):
    # Expected formal charges
    expected_fc = {
        0: 0,
        1: 0,
        2: -1,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0
    }
    for idx, atom in enumerate(mol.GetAtoms()):
        assert atom.GetFormalCharge() == expected_fc[idx], f"Atom {idx} FC incorrect"

def test_lewis_summary(mol):
    summary = lewis_summary(mol)
    # Check all atom symbols appear
    for sym in ["C", "O", "H"]:
        assert sym in summary
    # Check all LP and RAD values appear
    for lp, rad in [(3,0), (2,0), (0,0), (0,1)]:
        assert f"LP={lp}  RAD={rad}" in summary

def test_full_pipeline(symbols_coords):
    symbols, coords = symbols_coords
    mol = mol_from_xyz(symbols, coords)
    summary = lewis_summary(mol)
    assert "Atoms (idx, symbol, FC, LP, RAD):" in summary
    assert "Bonds (a1 - a2 : order):" in summary
