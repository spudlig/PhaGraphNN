import numpy as np
import xlrd
import os

import CDPL.Chem as Chem
import CDPL.Biomol as Biomol
import CDPL.Util as Util
import CDPL.Pharm as Pharm
import CDPL.Base as Base
import CDPL.ConfGen as ConfGen
import CDPL.Math as Math

import rdkit.Chem.AllChem as RDAllChem
from rdkit import Chem as RDChem

def readChemblXls(path_to_xls,col_entries = [0,7,10],sheet_index=0,n_entries=10000):
    '''
    reads the xls files and retrieves the ChemblID, smiles and activity metric \n
    Input: \n
    path_to_xls (string): path to the file.xls \n
    col_entries (list): the entries one wants to retrieve. Default: [0,7,10] \n
    sheet_index (int): Which sheet should be adressed. Default: 0 \n
    n_entries (int): How many rows are in the file and should be retieved. Default: 10000 \n
    Returns: \n
    list: all values retrieved from the xls file
    
    '''
    wb = xlrd.open_workbook(path_to_xls)
    sheet = wb.sheet_by_index(0)
    nrow = n_entries
    rown = 0
    colEntries = col_entries
    
    data = []
    try:
        for row in range(nrow):
            single_entry = []
            for col_entry in col_entries:
                single_entry.append(sheet.cell_value(row, col_entry))
            rown += 1
            data.append(single_entry)
    except Exception as e:
        print("End of xls file with",rown,"entries.")
        pass
    
    return data


def molFromSmiles(smiles,conformation):
    ''' 
    generates a RDKit Molecule from smiles. If confromations is true, then
    one random conformation will be generated. \n
    Input: \n
    smiles (string): smiles string \n
    conformation (boolean): generates one 3d conformation according to MMFF94 \n
    Return: \n
    (RDKitMolecule): the corresponding RDKit molecule 
    '''
    m = RDChem.MolFromSmiles(smiles)
    if conformation:
        m_conf = RDChem.AddHs(m)
        RDAllChem.EmbedMolecule(m_conf)
        RDAllChem.MMFFOptimizeMolecule(m_conf)
        return m_conf

    return m

def molFromSdf(sdf_path,conformation):
    ''' 
    generates a RDKit Molecule from smiles. If confromations is true, then
    one random conformation will be generated. \n
    Input: \n
    smiles (string): smiles string \n
    conformation (boolean): generates one 3d conformation according to MMFF94 \n
    Return: \n
    (RDKitMolecule): the corresponding RDKit molecule 
    '''
    suppl = RDChem.SDMolSupplier(sdf_path)

    if(len(suppl)>1):
        print(sys.stderr, '! More than 1 sdf in file - please use only one sdf per path !')
        return
    for m in suppl:
        if conformation:
            return _generateConformation(m)
        else:
            return m


def CDPLmolFromSmiles(smiles_path,conformation):
    ''' 
    generates a CDPL Molecule from smiles. If confromations is true, then
    one random conformation will be generated with explicit hydrogens. \n
    Input: \n
    smiles (string): smiles string \n
    conformation (boolean): generates one 3d conformation according to MMFF94 \n
    Return: \n
    (CDPL BasicMolecule): the corresponding CDPL BasicMolecule
    '''
    mol = Chem.BasicMolecule()
    if ".smi" in smiles_path:
        smi_reader = Chem.FileSMILESMoleculeReader(smiles_path)
        if not smi_reader.read(mol):
            print("COULD NOT READ SDF",file)
            return False
    else:
        mol = Chem.parseSMILES(smiles_path)
    if conformation:
        return _CDPLgenerateConformation(mol)
    else:
        return mol 

def CDPLmolFromSdf(sdf_path,conformation):
    '''
    generates a single CDPL Molecule from an sdf-file. If conformations is true, then
    one random conformation will be generated. \n
    Input: \n
    sdf_path (string): path to the sdf file \n
    conformation (boolean): generates one 3d conformation according to MMFF94 \n
    Return: \n
    (CDPL BasicMolecule): the corresponding CDPL BasicMolecule 
    '''
    mol = Chem.BasicMolecule()
    ifs = Base.FileIOStream(sdf_path, 'r')
    sdf_reader = Chem.SDFMoleculeReader(ifs)

    if not sdf_reader.read(mol):
        print("COULD NOT READ SDF",file)
        return False
    if conformation:
        return _CDPLgenerateConformation(mol)
    return mol

def _generateConformation(mol):
    ''' 
    PRIVATE METHOD
    generates a random confromation for a RDKit Molecule. \n
    Input: \n
    sdf_path (string): path to the sdf file \n
    conformation (boolean): generates one 3d conformation according to MMFF94 \n
    Return: \n
    (RDKitMolecule): the corresponding RDKit molecule 
    '''
    m_conf = RDChem.AddHs(mol)
    RDAllChem.EmbedMolecule(m_conf)
    RDAllChem.MMFFOptimizeMolecule(m_conf)
    return m_conf

def _CDPLgenerateConformation(cdplMol):
    '''
    PRIVAT METHOD
    configures a CDPL Molecule for conformation generation. \n
    Input: \n
    mol (CDPL BasicMolecule): a CDPL BasicMolecule \n
    Return: \n
    (CDPL BasicMolecule): the corresponding random conf. for the input BasicMolecule
     '''

    _CDPLconfigForConformation(cdplMol)
    cg = ConfGen.RandomConformerGenerator()
    coords = Math.Vector3DArray()
    i = 0

    cg.strictMMFF94AtomTyping = False

    ConfGen.prepareForConformerGeneration(cdplMol)

    coords.resize(cdplMol.numAtoms, Math.Vector3D())

    cg.setup(cdplMol)

    if cg.generate(coords) != ConfGen.RandomConformerGenerator.SUCCESS:
        print(sys.stderr, '! Conformer generation failed !')

    Chem.set3DCoordinates(cdplMol, coords)

    return cdplMol


def _CDPLconfigForConformation(mol):
    '''
    PRIVAT METHOD
    configures a CDPL Molecule for conformation generation. \n
    Input: \n
    mol (CDPL BasicMolecule): a CDPL BasicMolecule \n
    Return: \n
    (CDPL BasicMolecule): the configured input BasicMolecule
     '''
    Chem.perceiveComponents(mol, False)
    Chem.perceiveSSSR(mol, False)
    Chem.setRingFlags(mol, False)
    Chem.calcImplicitHydrogenCounts(mol, False)
    Chem.perceiveHybridizationStates(mol, False)
    Chem.setAromaticityFlags(mol, False)
    Chem.calcCIPPriorities(mol, False)
    Chem.calcAtomCIPConfigurations(mol, False)
    Chem.calcBondCIPConfigurations(mol, False)
    Chem.calcAtomStereoDescriptors(mol, False)
    Chem.calcBondStereoDescriptors(mol, False)
    Chem.calcTopologicalDistanceMatrix(mol, False)

    Chem.generate2DCoordinates(mol, False)
    Chem.generateBond2DStereoFlags(mol, True)

