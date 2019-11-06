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

    data = ut.readChemblXls("./tests/data/CHE_3.xls")
    graph_list = []
    for i in range(0,len(data)):
        graph = PhaGraph()
        mol = ut.CDPLmolFromSmiles(data[i][1],True)
        graph(ut.CDPLphaGenerator(None,mol,"lig_only"))
        graph.setProperty("ic50",data[i][2])
        graph_list.append(graph)
    from phagraphnn.DataPreperer import DataPreparer
    loader = DataPreparer(graph_list,3,property_string="ic50",mpn="gat",is_path=False)

    from phagraphnn.PhaGatModel2 import PhaGatModel2 as gat2
    from phagraphnn.PhaGatModel import PhaGatModel as gat
    import tensorflow as tf

    seq2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(128,),name="first_layer"),
    tf.keras.layers.Dense(32, activation='relu',name="second_layer"),
    tf.keras.layers.Dense(1,activation= None)],name="output_NN")

    seq = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
    tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
    tf.keras.layers.Dense(1,activation= None)],name="output_NN")

    gat2 = gat2(hidden_dim=32,output_nn=seq2,merge='cat')
    gat = gat(hidden_dim=32,output_nn=seq)
    lr = 0.001
    gat.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.RMSprop(lr))
    gat2.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.RMSprop(lr))
    rec = tf.keras.metrics.MeanAbsoluteError()
    for batch in loader:
        inputs,af,other = batch
        gat(inputs)
        gat2(inputs)
        for epoch in range(0,10):
            pred,loss = gat.train(inputs=inputs,outputs=af,learning_rate=lr)
            pred2,loss2 = gat2.train(inputs=inputs,outputs=af,learning_rate=lr)
            print("pred,loss",pred,loss)
            print("pred2,loss2",pred2,loss2)
            print("af",af)

def test_PhaGru():
    import phagraphnn.utilities as ut
    from phagraphnn.PhaGraph import PhaGraph,PhaNode

    data = ut.readChemblXls("./tests/data/CHE_3.xls")
    graph_list = []
    for i in range(0,len(data)):
        graph = PhaGraph()
        mol = ut.CDPLmolFromSmiles(data[i][1],True)
        graph(ut.CDPLphaGenerator(None,mol,"lig_only"))
        graph.setProperty("ic50",data[i][2])
        graph_list.append(graph)
    from phagraphnn.DataPreperer import DataPreparer
    loader = DataPreparer(graph_list,3,property_string="ic50",mpn="gru",is_path=False)

    from phagraphnn.PhaGruMPN import PhaGruMPN as gru
    from phagraphnn.PhaGruMPN2 import PhaGruMPN2 as gru2
    from phagraphnn.PhaGruMPN3 import PhaGruMPN3 as gru3
    import tensorflow as tf

    seq = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
    tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
    tf.keras.layers.Dense(1,activation= None)],name="output_NN")

    seq2 = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
    tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
    tf.keras.layers.Dense(1,activation= None)],name="output_NN")

    seq3 = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
    tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
    tf.keras.layers.Dense(1,activation= None)],name="output_NN")

    gru = gru(32,3,seq)
    gru2 = gru2(32,3,seq2)
    gru3 = gru3(32,3,seq3)

    lr = 0.001
    gru.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.RMSprop(lr))

    gru2.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.RMSprop(lr))

    gru3.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.RMSprop(lr))
    rec = tf.keras.metrics.MeanAbsoluteError()
    for batch in loader:
        inputs,af,other = batch
        gru(inputs)
        gru2(inputs)
        gru3(inputs)
        for epoch in range(0,10):
            pred,loss = gru.train(inputs=inputs,outputs=af,learning_rate=lr)
            pred,loss = gru2.train(inputs=inputs,outputs=af,learning_rate=lr)
            pred,loss = gru3.train(inputs=inputs,outputs=af,learning_rate=lr)

def test_PhaGat2():
    import phagraphnn.utilities as ut
    from phagraphnn.PhaGraph import PhaGraph,PhaNode

    data = ut.readChemblXls("./tests/data/CHE_3.xls")
    graph_list = []
    for i in range(0,len(data)):
        graph = PhaGraph()
        mol = ut.CDPLmolFromSmiles(data[i][1],True)
        graph(ut.CDPLphaGenerator(None,mol,"lig_only"))
        graph.setProperty("ic50",data[i][2])
        graph_list.append(graph)
    from phagraphnn.DataPreperer import DataPreparer
    loader = DataPreparer(graph_list,3,property_string="ic50",mpn="gru",is_path=False)

    from phagraphnn.PhaGatModel2 import PhaGatModel2 as gat
    import tensorflow as tf

    seq = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(128,),name="first_layer"),
    tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
    tf.keras.layers.Dense(1,activation= None)],name="output_NN")

    gat = gat(hidden_dim=32,output_nn=seq)
    lr = 0.001
    gat.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.RMSprop(lr))
    rec = tf.keras.metrics.MeanAbsoluteError()
    for batch in loader:
        inputs,af,other = batch
        gat(inputs)
        for epoch in range(0,10):
            pred,loss = gat.train(inputs=inputs,outputs=af,learning_rate=lr)

def test_PhaGat3():
    import phagraphnn.utilities as ut
    from phagraphnn.PhaGraph import PhaGraph,PhaNode

    data = ut.readChemblXls("./tests/data/CHE_3.xls")
    graph_list = []
    for i in range(0,len(data)):
        graph = PhaGraph()
        mol = ut.CDPLmolFromSmiles(data[i][1],True)
        graph(ut.CDPLphaGenerator(None,mol,"lig_only"))
        graph.setProperty("ic50",data[i][2])
        graph_list.append(graph)
    from phagraphnn.DataPreperer import DataPreparer
    loader = DataPreparer(graph_list,3,property_string="ic50",mpn="gru",is_path=False)

    from phagraphnn.PhaGatModel3 import PhaGatModel3 as gat
    import tensorflow as tf

    seq = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(128,),name="first_layer"),
    tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
    tf.keras.layers.Dense(1,activation= None)],name="output_NN")

    gat = gat(hidden_dim=32,output_nn=seq)
    lr = 0.001
    gat.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.RMSprop(lr))
    rec = tf.keras.metrics.MeanAbsoluteError()
    for batch in loader:
        inputs,af,other = batch
        gat(inputs)
        for epoch in range(0,10):
            pred,loss = gat.train(inputs=inputs,outputs=af,learning_rate=lr)

def test_PhaGat2_classification():
    import phagraphnn.utilities as ut
    from phagraphnn.PhaGraph import PhaGraph,PhaNode

    data = ut.readChemblXls("./tests/data/CHE_3.xls")
    graph_list = []
    for i in range(0,len(data)):
        graph = PhaGraph()
        mol = ut.CDPLmolFromSmiles(data[i][1],True)
        graph(ut.CDPLphaGenerator(None,mol,"lig_only"))
        graph.setProperty("ic50",(0,1))
        graph_list.append(graph)
    from phagraphnn.DataPreperer import DataPreparer
    loader = DataPreparer(graph_list,3,property_string="ic50",mpn="gru",is_path=False)

    from phagraphnn.PhaGatModel2 import PhaGatModel2 as gat
    import tensorflow as tf

    seq = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(128,),name="first_layer"),
    tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
    tf.keras.layers.Dense(2,activation= None)],name="output_NN")

    gat = gat(hidden_dim=32,output_nn=seq,regression = False)
    lr = 0.001
    gat.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.RMSprop(lr))
    rec = tf.keras.metrics.MeanAbsoluteError()
    for batch in loader:
        inputs,af,other = batch
        gat(inputs)
        for epoch in range(0,10):
            pred,loss = gat.train(inputs=inputs,outputs=af,learning_rate=lr)