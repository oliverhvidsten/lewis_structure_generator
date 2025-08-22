# test_lewis_from_xyz.py
import sys
sys.path.append('../')

import pytest
from pathlib import Path
from generate import (
    read_xyz, mol_from_xyz, bond_order_sum,
    estimate_lone_pairs, lewis_summary
)
from rdkit import Chem

# Example XYZ content for POF3
POF3_XYZ = """5

P         -1.94427        1.39204       -0.36838
O         -1.80046        2.92326       -0.16275
F         -3.45011        1.04242        0.38096
F         -0.56439        0.76357        0.43922
F         -1.92545        1.22950       -2.07849
"""

# --- Fixtures ---
@pytest.fixture
def xyz_file(tmp_path):
    file_path = tmp_path / "POF3.xyz"
    file_path.write_text(POF3_XYZ)
    return file_path

@pytest.fixture
def symbols_coords(xyz_file):
    symbols, coords = read_xyz(xyz_file)
    return symbols, coords

@pytest.fixture
def mol(symbols_coords):
    symbols, coords = symbols_coords
    mol = mol_from_xyz(symbols, coords, charge=0)
    return mol

# --- Tests ---

def test_read_xyz(symbols_coords):
    symbols, coords = symbols_coords
    assert len(symbols) == 5
    assert symbols[0] == "P"
    assert symbols[1] == "O"
    assert all(len(c) == 3 for c in coords)

def test_mol_from_xyz(mol):
    assert isinstance(mol, Chem.Mol)
    # Check that all atoms are present
    assert mol.GetNumAtoms() == 5
    atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    assert "P" in atom_symbols
    assert atom_symbols.count("F") == 3

def test_bond_order_sum(mol):
    # P atom is index 0, should have sum of bonds ~5 (P=O double + 3 P-F singles)
    bo_sum = bond_order_sum(mol, 0)
    assert abs(bo_sum - 5.0) < 0.1

def test_estimate_lone_pairs(mol):
    lp_data = estimate_lone_pairs(mol)
    # Check correct number of entries
    assert len(lp_data) == 5
    # O atom index 1 should have 2 lone pairs
    fc, lp, rad = lp_data[1]
    assert lp == 2

def test_lewis_summary(mol):
    summary = lewis_summary(mol)
    # Check key substrings appear
    assert "P" in summary
    assert "O" in summary
    assert "F" in summary
    assert "LP=2" in summary  # O lone pairs
    # Should include all bond orders
    for a, b in [(0,1),(0,2),(0,3),(0,4)]:
        assert (f"{a} - {b}" in summary) or (f"{b} - {a}" in summary)

def test_full_run(xyz_file):
    # Run full pipeline end-to-end
    symbols, coords = read_xyz(xyz_file)
    mol = mol_from_xyz(symbols, coords)
    summary = lewis_summary(mol)
    # Ensure summary contains all atoms
    for atom in ["P","O","F"]:
        assert atom in summary
