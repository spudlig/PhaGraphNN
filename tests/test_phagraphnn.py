#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `PhaphagraphNN` package."""


import unittest
# from click.testing import CliRunner

import pytest
import sys, os
import pickle
import phagraphnn
import math

def test_equ():
    assert(1.0 == 1.0)

def test_phagraphNN_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "phagraphnn" in sys.modules

def test_utilities_readChemblXls_method():
    import phagraphnn.utilities as ut
    print(os.getcwd(),"test")
    data = ut.readChemblXls("./tests/data/CHE_3.xls")
    assert("CHEMBL400569" == data[0][0])
    assert("CCNCc1cncc(c1)c2cnc3[nH]nc(c4nc5cc(ccc5[nH]4)N6CCN(C)CC6)c3c2" == data[0][1])

def test_phagraphNN_generateFromSmiles_file():
    import phagraphnn.utilities as ut
    path = "./tests/data/twoD.smi"
    mol = ut.CDPLmolFromSmiles(path,False)
    assert(len(mol.atoms) == 22) #2D
    mol_conf = ut.CDPLmolFromSmiles(path,True)
    assert(len(mol_conf.atoms) == 37) #3D

def test_phagraphNN_generateFromSmiles_smile():
    import phagraphnn.utilities as ut

    data = ut.readChemblXls("./tests/data/CHE_3.xls")
    mol =ut.CDPLmolFromSmiles(data[2][1],False) # same smiles then twoD.smi
    mol_conf =ut.CDPLmolFromSmiles(data[2][1],True) # same smiles then twoD.smi

    assert(len(mol.atoms) == 22) #2D
    assert(len(mol_conf.atoms) == 37) #3D with Hydrogens

def test_phagraphNN_generateFromSDF():
    import phagraphnn.utilities as ut
    path = "./tests/data/threeD_activity.sdf"
    mol = ut.CDPLmolFromSdf(path,False)
    mol_conf = ut.CDPLmolFromSdf(path,True)
    assert(len(mol.atoms) == 37) # 3D with hydrogens
    assert(len(mol_conf.atoms) == 37) #3D with Hydrogens, but different conf.

def test_phagraphNN_generateFromPDB_FILE():
    import phagraphnn.utilities as ut
    path = "./tests/data/1ke7.pdb"
    pdb,env,lig = ut.CDPLreadProteinFile(path,"LS3",9,True)
    assert(len(pdb.atoms) == 4608) # without water
    assert(len(env.atoms) == 1118) # without water
    assert(len(lig.atoms) == 43)

    pdb,env,lig = ut.CDPLreadProteinFile(path,"LS3",9,False)
    assert(len(pdb.atoms) == 4929) # with water
    assert(len(env.atoms) == 1154) # with water
    assert(len(lig.atoms) == 43)

def test_phagraphNN_generateFromPDB_URL():
    import phagraphnn.utilities as ut
    path = "./tests/data/1ke7.pdb"
    pdb,env,lig = ut.CDPLdownloadProteinFile("1ke7","LS3",9,True)
    assert(len(pdb.atoms) == 4608) # without water
    assert(len(env.atoms) == 1118) # without water
    assert(len(lig.atoms) == 43)

    pdb,env,lig = ut.CDPLdownloadProteinFile("1ke7","LS3",9,False)
    assert(len(pdb.atoms) == 4929) # with water
    assert(len(env.atoms) == 1154) # with water
    assert(len(lig.atoms) == 43)

def test_phagraphNN_generatePha():
    import phagraphnn.utilities as ut
    path = "./tests/data/threeD_activity.sdf"
    mol = ut.CDPLmolFromSdf(path,False)
    mol_conf = ut.CDPLmolFromSdf(path,True)
    assert(len(mol.atoms) == 37) # 3D with hydrogens
    assert(len(mol_conf.atoms) == 37) #3D with Hydrogens, but different conf.
    ut.CDPLphaGenerator(None,mol_conf,"lig_only")

def test_phagraphNN_generate_protein_phas():
    import phagraphnn.utilities as ut
    path = "./tests/data/1ke7.pdb"
    pdb,env,lig = ut.CDPLdownloadProteinFile("1ke7","LS3",9,False)
    lig_pha = ut.CDPLphaGenerator(env,lig,"lig_only")
    env_pha = ut.CDPLphaGenerator(env,lig,"env_only")
    inter = ut.CDPLphaGenerator(env,lig,None)

def test_phagraphNN():
    import phagraphnn.utilities as ut
    from phagraphnn.PhaGraph import PhaGraph,PhaNode
    path = "./tests/data/threeD_activity.sdf"
    mol = ut.CDPLmolFromSdf(path,False)
    pha = ut.CDPLphaGenerator(None,mol,"lig_only")
    graph = PhaGraph()
    graph(pha)
    assert(len(graph.nodes) == 10)
    assert(len(graph.edge_weights) == 36)
    assert(len(graph.edge_weights) == len(graph.edges))
    assert(graph.nodes[3].index == -1)
    assert(graph.nodes[0].feature_type == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert(math.isclose(graph.distance_idx(graph.nodes[0].index,graph.nodes[1].index),3.922))
    assert(math.isclose(graph.distance_idx(graph.nodes[1].index,graph.nodes[0].index),3.922))
    assert(math.isclose(graph.distance_idx(graph.nodes[1].index,graph.nodes[9].index),7.443))
    assert(graph.distance_idx(graph.nodes[0].index,graph.nodes[3].index) == False)

def test_gatNN():
    import phagraphnn.utilities as ut
    from phagraphnn.PhaGraph import PhaGraph,PhaNode
    path = "./tests/data/threeD_activity.sdf"
    mol = ut.CDPLmolFromSdf(path,False)
    pha = ut.CDPLphaGenerator(None,mol,"lig_only")
    graph = PhaGraph()
    graph(pha)

    from phagraphnn.PhaGatModel import PhaGatModel
    model = PhaGatModel()