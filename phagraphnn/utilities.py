import logging
log = logging.getLogger(__name__)

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

import tensorflow as tf

import rdkit.Chem.AllChem as RDAllChem
from rdkit import Chem as RDChem
from urllib.request import urlretrieve


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
    nr_row = n_entries
    row_nr = 0
    colEntries = col_entries
    
    data = []
    try:
        for row in range(nr_row):
            single_entry = []
            for col_entry in col_entries:
                single_entry.append(sheet.cell_value(row, col_entry))
            row_nr += 1
            data.append(single_entry)
    except Exception as e:
        log.info("End of xls file with",row_nr,"entries.",exc_info=True)
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
    m = None
    try:
        m = RDChem.MolFromSmiles(smiles)
    except Exception as e:
        log.error("Could not parse RDKitSmiles smiles",exc_info=True)
        
    if conformation:
        try:
            m_conf = RDChem.AddHs(m)
            RDAllChem.EmbedMolecule(m_conf)
            RDAllChem.MMFFOptimizeMolecule(m_conf)
            return m_conf
        except Exception as e:
            log.error("Could not parse generate a valid conformation",exc_info=True)
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
        log.error('! More than 1 sdf in file - please use only one sdf per path !')
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
            log.error("COULD NOT READ Smiles",smiles_path)
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
        log.error("COULD NOT READ SDF",sdf_path)
        return False
    if conformation:
        return _CDPLgenerateConformation(mol)
    return mol

def CDPLphaFromPML(pml_path):
    '''
    reads a single CDPL BasicPharmacophore from an pml-file.
    Input: \n
    pml_path (string): path to the pml file \n
    Return: \n
    (CDPL BasicPharmacophore): the corresponding CDPL BasicPharmacophore 
    '''
    pha = Pharm.BasicPharmacophore()
    ifs = Base.FileIOStream(pml_path, 'r')
    pml_reader = Pharm.PMLPharmacophoreReader(ifs)

    if not pml_reader.read(pha):
        log.error("COULD NOT READ PML",pml_path)
        return False
    return pha

def CDPLphaGenerator(protein,ligand,pha_type):
    '''
    generates the pharmacophore for either the ligand, the environment or
    the interaction between them.
    Input: \n
    protein (CDPL Fragment): the CDPL protein fragment (=env)  \n
    ligand (CDPL BasicMolecule): a molecule or a ligand in the corresponding
     protein pocket  \n
    pha_type (string): either "lig_only", "env_only" or None - then its the
    interaction pharamcophore  \n
    Return: \n
    (CDPL BasicPharmacophore): the corresponding pharmacophore
     '''
    lig_pharm = None
    if pha_type is 'lig_only':
        Chem.perceiveSSSR(ligand, True)
        lig_pharm = _CDPLgeneratePha(ligand,pha_type)
        return lig_pharm
    Chem.perceiveSSSR(protein, True)
    env_pharm = None
    if pha_type is 'env_only':
        env_pharm = _CDPLgeneratePha(protein,pha_type)
        return env_pharm

    Chem.perceiveSSSR(ligand, True)
    lig_pharm = _CDPLgeneratePha(ligand,pha_type)
    env_pharm = _CDPLgeneratePha(protein,pha_type)
    
    mapping = Pharm.FeatureMapping()

    Pharm.DefaultInteractionAnalyzer().analyze(lig_pharm, env_pharm, mapping)
    int_pharm = Pharm.BasicPharmacophore()
    Pharm.buildInteractionPharmacophore(int_pharm, mapping)
    return int_pharm

def CDPLdownloadProteinFile(pdb_four_letter_code,lig_three_letter_code,radius,remove_water=True):
    '''
    downloads the PDB with the corresponding four letter code.
    Input:\n
    pdb_four_letter_code (String): the pdb_four_letter_code of the protein structure -
    it then tries to download the corresponding protein structure. \n
    Return: \n
    (CDPL BasicMolecule): the protein structure \n
    '''
    directory = os.getcwd()

    path = directory +"/temp_pdb/"

    if not os.path.exists(path):
        os.makedirs(path)
    urlretrieve('http://files.rcsb.org/download/'+pdb_four_letter_code+'.pdb',(path+pdb_four_letter_code+".pdb"))
    pdb_mol = _CDPLreadFromPDBFile(path+pdb_four_letter_code+".pdb")

    if remove_water:
        _removeWater(pdb_mol)

    environment, ligand = _CDPLextractProteinFragments(pdb_mol,lig_three_letter_code=lig_three_letter_code,radius=radius)
    
    os.remove(path+pdb_four_letter_code+".pdb")

    return pdb_mol, environment, ligand

def CDPLreadProteinFile(path_to_pdb,lig_three_letter_code,radius,remove_water=True):
    '''
    Reads a pdb file from a path
    Input: \n
    path_to_pdb (String): the path to the protein structure  \n
    lig_three_letter_code (string): the three letter code for the ligand \n
    radius (float): the radius within every residue is being extracted for the environment fragment.
    The origin of the radius is the defined ligand. \n
    Return: \n
    (CDPL BasicMolecule): the protein structure \n
    (CDPL Fragment): the environment residues within the defined radius of the ligand \n
    (CDPL Fragment): the defined ligand \n
    '''
    pdb_mol = _CDPLreadFromPDBFile(path_to_pdb)
    if remove_water:
        _removeWater(pdb_mol)
    
    environment, ligand = _CDPLextractProteinFragments(pdb_mol,lig_three_letter_code,radius=radius)

    return pdb_mol, environment, ligand

def savePharmacophore(pha,path):
    '''
    Saves a particula pha at the target path.\n
    Input:\n
    pha (CDPL BasicPharmacophore): the pharmacophore to be saved as a pml file \n
    path (String): path where to save the pml file (includes the filename.pml)
    '''

    Pharm.PMLFeatureContainerWriter(Base.FileIOStream(path
                                                      ,'w')).write(pha)

    return True
    

def _removeWater(mol):
    to_remove = list()
    for atom in mol.atoms:
        if Biomol.getResidueCode(atom) == 'HOH':
            to_remove.append(atom)

    for atom in to_remove:
        mol.removeAtom(mol.getAtomIndex(atom))


def _CDPLextractProteinFragments(pdb_mol, lig_three_letter_code, radius=6.0):
    lig = Chem.Fragment()
    _CDPLcalcProteinProperties(pdb_mol)

    # extract ligand
    for atom in pdb_mol.atoms:
        if Biomol.getResidueCode(atom) == lig_three_letter_code:
            Biomol.extractResidueSubstructure(atom, pdb_mol, lig, False)
    if lig.numAtoms == 0:
        log.error("The defined three letter code is not existing:",lig_three_letter_code)
    # extract environment
    env = Chem.Fragment()
    Biomol.extractEnvironmentResidues(lig, pdb_mol, env, float(radius))
    
    return env,lig


def _CDPLgeneratePha(mol,pha_type):
    '''
    PRIVAT METHOD
    generates the pharmacophore for the molecule and is used by the CDPLphaGenerator.
    Input: \n
    mol (CDPL BasicMolecule): the molecule the pharmacophore needs to be generated for
    lig_only (string): either True, then there are is no hydrogens coordinates being 
    calculated  \n
    Return: \n
    (CDPL BasicPharmacophore): the corresponding pharmacophore
     '''
    if pha_type is not 'lig_only': #TODO What exactly should be in the config for the pha generation?
        Chem.generateHydrogen3DCoordinates(mol, True)
    pharm = Pharm.BasicPharmacophore()
    pharm_generator = Pharm.DefaultPharmacophoreGenerator(True)
    pharm_generator.generate(mol, pharm)
    return pharm

def _CDPLreadFromPDBFile(pdb_file):
    '''
    PRIVAT METHOD
    reads a pdb file and is used by the CDPLreadProteinFile method.
    Input: \n
    pdb_file (string): the path to the pdb file  \n
    Return: \n
    (CDPL BasicMolecule): the corresponding pdb molecule
     '''
    ifs = Base.FileIOStream(pdb_file, 'r')
    pdb_reader = Biomol.PDBMoleculeReader(ifs)
    pdb_mol = Chem.BasicMolecule()

    Biomol.setPDBApplyDictAtomBondingToNonStdResiduesParameter(pdb_reader, False) #TODO Should this be there for the pdb readin? or also in the config?

    if not pdb_reader.read(pdb_mol):
        log.error("COULD NOT READ PDB",pdb_file)
        return False

    return pdb_mol

def createVar(tensor, requires_grad=None):
    '''
    Initializes a tensorflow variable
    '''
    if requires_grad is None:
        return tf.Variable(tensor,trainable=False)
    else:
        return tf.Variable(tensor,trainable=True)

def indexSelect(source, dim, index):
    '''
    Selects the corresponding indices for MPN
    '''
    index_size = tf.shape(index)
    suffix_dim = tf.shape(source)[1:]
    final_size = tf.concat((index_size, suffix_dim),axis=0)
    inter = tf.reshape(index,shape=[-1])
    target = tf.gather(source,indices=inter,axis=dim)
    return tf.reshape(target,shape=final_size)

def updateConnected(atom_features,scope):
    updated = []
    for i in scope:
        updated.append(atom_features[i])
    return updated

def updateConnectedDict(features,feature_dict,scope):
    updated = []
    for i in scope:
        updated.append(features[feature_dict[i]])
    return updated

def getConnectedFeatures(features, scope):
    '''
    this functions selects according to the scope, the selected connected
    features and returns them. 0 based indices. \n
    INPUT: \n
    features (list): a list of features \n
    scope (list): a list of connections \n
    RETURNS: \n
    (list): 
    '''
    gat = tf.gather(features,indices=(scope)) # get for each feature entry the connected ones
    gat = tf.reshape(gat,(tf.shape(gat)[0]*tf.shape(gat)[1],tf.shape(gat)[2])) # remove the 3 shape introduced
    intermediate_tensor = tf.reduce_sum(tf.abs(gat), 1) # search for all the 0 entries introduced by the gather an concat above
    zero_vector = tf.zeros(shape=(tf.shape(gat)[0]), dtype=tf.float32) # generate dummy matrix for comparison
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector) # compare dummy with other matrix
    return tf.boolean_mask(gat, bool_mask)

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
    m_conf = None
    try:
        m_conf = RDChem.AddHs(mol)
        RDAllChem.EmbedMolecule(m_conf)
        RDAllChem.MMFFOptimizeMolecule(m_conf)
    except Exception as e:
        log.error("Could not generate valid conformation",exc_info=True)
        pass

    return m_conf

def _CDPLgenerateConformation(cdpl_mol):
    '''
    PRIVAT METHOD
    configures a CDPL Molecule for conformation generation. \n
    Input: \n
    mol (CDPL BasicMolecule): a CDPL BasicMolecule \n
    Return: \n
    (CDPL BasicMolecule): the corresponding random conf. for the input BasicMolecule
     '''

    _CDPLconfigForConformation(cdpl_mol) #TODO What exactly should be in the config for the cmp generation?
    cg = ConfGen.RandomConformerGenerator()
    coords = Math.Vector3DArray()
    i = 0

    cg.strictMMFF94AtomTyping = False

    ConfGen.prepareForConformerGeneration(cdpl_mol)

    coords.resize(cdpl_mol.numAtoms, Math.Vector3D())

    cg.setup(cdpl_mol)

    if cg.generate(coords) != ConfGen.RandomConformerGenerator.SUCCESS:
        log.error('! Conformer generation failed !')
        return

    Chem.set3DCoordinates(cdpl_mol, coords)

    return cdpl_mol


def _CDPLconfigForConformation(mol): # TODO is this the right way to handle ligands for conf. generation?
    '''
    PRIVAT METHOD
    configures a CDPL BasicMolecule for conformation generation. \n
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

def _CDPLcalcProteinProperties(pdb_mol): # TODO is this the right way to handle protein structures?
    '''
    PRIVAT METHOD
    configures a CDPL BasicMolecule for a protein structure. Is used in the _CDPLextractProteinFragments method \n
    Input: \n
    pdb_mol (CDPL BasicMolecule): a CDPL BasicMolecule representing the protein structure \n
    '''
    Chem.calcImplicitHydrogenCounts(pdb_mol, True)
    Chem.perceiveHybridizationStates(pdb_mol, True)        
    Chem.makeHydrogenComplete(pdb_mol)
    Chem.setAtomSymbolsFromTypes(pdb_mol, False)        
    Chem.calcImplicitHydrogenCounts(pdb_mol, True)
    Chem.setRingFlags(pdb_mol, True)
    Chem.setAromaticityFlags(pdb_mol, True)
    Chem.generateHydrogen3DCoordinates(pdb_mol, True)   
    Biomol.setHydrogenResidueSequenceInfo(pdb_mol, False) 

