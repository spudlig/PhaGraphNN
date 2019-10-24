#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `PhaphagraphNN` package."""


import unittest
# from click.testing import CliRunner

import pytest
import sys, os
import pickle
import phagraphnn

def test_equ():
    assert(1.0 == 1.0)

def test_phagraphNN_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "phagraphnn" in sys.modules

def test_utilities_readChemblXls_method():
    import phagraphnn.utilities as ut
    print(os.getcwd(),"test")
    data = ut.readChemblXls("../data/CHE_3.xls")
    assert("CHEMBL400569" == data[0][0])
    assert("CCNCc1cncc(c1)c2cnc3[nH]nc(c4nc5cc(ccc5[nH]4)N6CCN(C)CC6)c3c2" == data[0][1])

def test_phagraphNN_generateFromSmiles_file():
    import phagraphnn.utilities as ut
    path = "../data/twoD.smi"
    mol = ut.CDPLmolFromSmiles(path,False)
    assert(len(mol.atoms) == 22) #2D
    mol_conf = ut.CDPLmolFromSmiles(path,True)
    assert(len(mol_conf.atoms) == 37) #3D

def test_phagraphNN_generateFromSmiles_smile():
    import phagraphnn.utilities as ut

    data = phagraphNN.utilities.readChemblXls("../data/CHE_3.xls")
    mol =ut.CDPLmolFromSmiles(data[2][1],False) # same smiles then twoD.smi
    mol_conf =ut.CDPLmolFromSmiles(data[2][1],True) # same smiles then twoD.smi

    assert(len(mol.atoms) == 22) #2D
    assert(len(mol_conf.atoms) == 37) #3D with Hydrogens

def test_phagraphNN_generateFromSDF():
    import phagraphnn.utilities as ut
    path = "../data/threeD_activity.sdf"
    mol = ut.CDPLmolFromSdf(path,False)
    mol_conf = ut.CDPLmolFromSdf(path,True)
    assert(len(mol.atoms) == 37) # 3D with hydrogens
    assert(len(mol_conf.atoms) == 37) #3D with Hydrogens, but different conf.

def test_phagraphNN_nodes():
    import phagraphnn.utilities as ut
    from PhaGraph import PhaGraph,PhaNode
    data = phagraphNN.utilities.readChemblXls("../data/CHE_3.xls")
    mol =ut.CDPLmolFromSmiles(data[0][1],True)
    ag = AtomGraph()
    ag(mol)
    assert(len(ag.nodes) == 35)
    assert(ag.nodes[3].index == 3)
    assert(ag.nodes[3].features[0] == 1.0)
    assert(len(ag.nodes[3].features) == 27)
    assert(ag.nodes[0].index == ag.nodes[0].atom_idx)
    assert(ag.nodes[23].index == ag.nodes[23].atom_idx)
    assert(ag.nodes[34].index == ag.nodes[34].atom_idx)

def test_phagraphNN_edges():
    import phagraphnn.utilities as ut
    from PhaGraph import PhaGraph,PhaNode
    data = ut.readChemblXls("../data/CHE_3.xls")
    mol =ut.CDPLmolFromSmiles(data[0][1],True)
    ag = AtomGraph()
    ag(mol)
    assert(len(ag.edges) == 40)
    assert(len(ag.edge_features[(1,2)]) == 5)
    assert(ag.edge_features[(1,2)] == [1.0, 0.0, 0.0, 0.0, 0.0])
    assert(ag.connected(ag.nodes[1],ag.nodes[2]) == True)
    assert(ag.nodes[1].neighbors[1] == ag.nodes[2].index)
    assert(ag.nodes[2].neighbors[0] == ag.nodes[1].index)

def test_phagraphNN_edge_distance():
    import phagraphnn.utilities as ut
    from PhaGraph import PhaGraph,PhaNode
    import math
    data = ut.readChemblXls("../data/CHE_3.xls")
    mol =ut.CDPLmolFromSmiles(data[0][1],True)
    ag = AtomGraph()
    ag(mol)
    assert(math.isclose(ag.distance_idx(ag.nodes[1].index,ag.nodes[2].index),1.461968478419764))
    assert(math.isclose(ag.distance_idx(ag.nodes[13].index,ag.nodes[33].index),1.3856457839383056))
    assert(math.isclose(ag.distance_idx(ag.nodes[13].index,ag.nodes[33].index),ag.distance_idx(ag.nodes[13].atom_idx,ag.nodes[33].atom_idx)))