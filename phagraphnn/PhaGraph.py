import CDPL.Chem as Chem
import CDPL.Biomol as Biomol
import CDPL.Util as Util
import CDPL.Pharm as Pharm
import CDPL.Base as Base
import CDPL.ConfGen as ConfGen
import CDPL.Math as Math

import numpy as np
import sys
from copy import deepcopy
import math

ELEM_LIST =[0,1,2,3,4,5,6,7] #0=unk,1=H,2=AR,3=NI,4=PI, 5=HBD,6=HBA,7=XV

class PhaNode():

    def __init__(self,feature_type):
        self.feature_type = feature_type
        self.coords = np.array([0.0, 0.0, 0.0])
        self.index = 0
        self.other_feature_types = [0,0,0,0,0,0,0,0]

class PhaGraph():

    def __init__(self):
        self.nodes = list()
        self.edges = set()
        self.edge_weights = dict()
        self.affinity = None
        self.pdb_name = None
        self.activ = [0,0]
        self.properties = dict()

    def getAffinityValue(self):
        return self.affinity

    def setAffinityValue(self,affinity):
        self.affinity = deepcopy(affinity)

    def getProperty(self,property_name):
        if property_name in properties.keys():
            return self.properties[property_name]
        else:
            print("Property name",property_name,"does not exist. Only those names are present:",properties.keys())

    def setProperty(self,property_name,property_):
        self.properties[property_name] = property_
]

    def getPDBName(self):
        return self.pdb_name

    def setPDBName(self,name):
        self.pdb_name = name

    def distance(self, node1, node2):
        return float(self.distMatrix[node1.index, node2.index])

    def distance_idx(self, idx1, idx2):
        return float(self.distMatrix[idx1, idx2])

    def connected(self, node1, node2):
        if (node1.index, node2.index) in self.edges:
            return True

        if (node2.index, node1.index) in self.edges:
            return True

        return False

    def readProteinFile(self,path_to_file):
        # reads pdb or mmtf files. Other formats are not supported
        # input: Either .pdb or .mmtf files
        # output: Protein structure as BasicMolecule#
        if ".mmtf" in path_to_file:
            return self._readFromMMTFFile(path_to_file)

        if ".pdb" in path_to_file:
            return self._readFromPDBFile(path_to_file)

        return False

    def generateConformation(self,m):
        cg = ConfGen.RandomConformerGenerator()
        coords = Math.Vector3DArray()
        i = 0

        cg.strictMMFF94AtomTyping = False

        ConfGen.prepareForConformerGeneration(m)

        coords.resize(m.numAtoms, Math.Vector3D())

        cg.setup(m)

        if cg.generate(coords) != ConfGen.RandomConformerGenerator.SUCCESS:
            print(sys.stderr, '! Conformer generation failed !')

        Chem.set3DCoordinates(m, coords)

        return m

    def generateFromSmiles(self,smiles,pha_type="lig_only"):
        # generates ph graph from smiles
        # input: smiles
        lig = Chem.parseSMILES(smiles)

        self._config(lig)

        self.generateConformation(lig)

        self._generate_nodes(None,lig,pha_type)
        self._calcDistMatrix()
        self._generate_edges()

    def generateFromPDBFile(self, pdb_mol, lig_code,pha_type, radius=6.0):
        lig = Chem.Fragment()

        PhaGraph._calcMolProperties(pdb_mol)

        # extract ligand
        for atom in pdb_mol.atoms:
            if Biomol.getResidueCode(atom) == lig_code:
                Biomol.extractResidueSubstructure(atom, pdb_mol, lig, False)
        if lig.numAtoms == 0:
            return False
        # extract environment
        env = Chem.Fragment()
        Biomol.extractEnvironmentResidues(lig, pdb_mol, env, float(radius))
        # remove water
        to_remove = list()
        for atom in env.atoms:
            if Biomol.getResidueCode(atom) == 'HOH':
                to_remove.append(atom)

        for atom in to_remove:
            env.removeAtom(env.getAtomIndex(atom))

        self.generate(env, lig,pha_type)
        return True

    def generate(self,env,lig,pha_type):
        self._generate_nodes(env,lig,pha_type)
        self._calcDistMatrix()
        self._generate_edges()

    def _generate_edges(self):
        for i in range(0, len(self.nodes)):
            phan1 = self.nodes[i]

            for j in range(i+1, len(self.nodes)):
                phan2 = self.nodes[j]
                if phan1.index == phan2.index:continue

                dist = self.distance(phan1,phan2)

                self.edges.add((i, j))
                self.edge_weights[(i,j)] = (dist)

    def _config(self,mol):
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

    def _generate_nodes(self,env,lig,pha_type):
        pha = self._get_pha(env,lig,pha_type)
        index_counter = 0 # zero based
        if pha_type == "lig_only":
            for feature in pha:
                phan = PhaNode(Pharm.getType(feature))
                phan.other_feature_types = self.onek_encoding_unk(Pharm.getType(feature), ELEM_LIST)
                phan.tolerance = Pharm.getTolerance(feature)
                phan.coords[0]= round(Chem.get3DCoordinates(feature)[0],6)
                phan.coords[1]= round(Chem.get3DCoordinates(feature)[1],6)
                phan.coords[2]= round(Chem.get3DCoordinates(feature)[2],6)
                phan.index = index_counter
                index_counter += 1
                self.nodes.append(phan)
            return

        for feature in pha:
            phan = PhaNode(Pharm.getType(feature))
            phan.tolerance = Pharm.getTolerance(feature)
            phan.coords[0]= round(Chem.get3DCoordinates(feature)[0],6)
            phan.coords[1]= round(Chem.get3DCoordinates(feature)[1],6)
            phan.coords[2]= round(Chem.get3DCoordinates(feature)[2],6)
            phan.index = index_counter
            index_counter += 1
            self.nodes.append(phan)

    def _get_pha(self,protein, ligand,pha_type):
        lig_pharm = None
        if pha_type is 'lig_only':
            lig_pharm = self._generate_pha(ligand,pha_type)
            return lig_pharm
        Chem.perceiveSSSR(protein, True)
        env_pharm = None
        if pha_type is 'env_only':
            env_pharm = self._generate_pha(protein)
            return env_pharm
        mapping = Pharm.FeatureMapping()

        Pharm.DefaultInteractionAnalyzer().analyze(lig_pharm, env_pharm, mapping)
        int_pharm = Pharm.BasicPharmacophore()
        Pharm.buildInteractionPharmacophore(int_pharm, mapping)
        return int_pharm


    def _generate_pha(self,mol,pha_type):
        if pha_type is not 'lig_only':
            Chem.generateHydrogen3DCoordinates(mol, True)
        pharm = Pharm.BasicPharmacophore()
        pharm_generator = Pharm.DefaultPharmacophoreGenerator(True)
        pharm_generator.generate(mol, pharm)
        return pharm

    def _readFromPDBFile(self,pdb_file):
        # reads protein structure from pdb files
        # input: path to pdb file
        # output: pdb_mol#

        ifs = Base.FileIOStream(pdb_file, 'r')
        pdb_reader = Biomol.PDBMoleculeReader(ifs)
        pdb_mol = Chem.BasicMolecule()

        # Biomol.setPDBApplyDictAtomBondingToNonStdResiduesParameter(pdb_reader, False)
        # Biomol.setPDBApplyDictBondOrdersToNonStdResiduesParameter(pdb_reader, False)

        if not pdb_reader.read(pdb_mol):
            print("COULD NOT READ PDB",pdb_file)
            return False

        return pdb_mol

    def onek_encoding_unk(self,x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))

    def _readFromMMTFFile(self,mmtf_file):
        # reads protein structure from mmtf files
        # input: path to mmtf file
        # output: pdb_mol#
        structure = self._get_from_decoded(mmtf_file)
        pdb_reader = Biomol.FileMMTFMoleculeReader(structure)
        pdb_mol = Chem.BasicMolecule()

        Biomol.setPDBApplyDictAtomBondingToNonStdResiduesParameter(pdb_reader, False)

        if not pdb_reader.read(pdb_mol):
            return False

        return pdb_mol

    def _calcDistMatrix(self):
        # self.distMatrix = np.ndarray(shape=(len(self.nodes), len(self.nodes)), dtype=float)

        # for i in range(0, len(self.nodes)):
        #     for j in range(i, len(self.nodes)):
        #         if i == j:
        #             self.distMatrix[i, j] = 0
        #         else:
        #             dist = np.linalg.norm(self.nodes[i].coords - self.nodes[j].coords)
        #             self.distMatrix[i, j] = dist
        #             self.distMatrix[j, i] = dist

        self.distMatrix = np.ndarray(shape=(len(self.nodes), len(self.nodes)), dtype=float)
        for i in range(0, len(self.nodes)):
            for j in range(i, len(self.nodes)):
                if i == j:
                    self.distMatrix[i, j] = 0.0
                else:
                    if self.nodes[j].index == -1 or self.nodes[i].index == -1: continue
                    # when dist is 0 and its not itself (i==J), create a new feature type, safe the old
                    # one in the other_feature_type list. 
                    dist = (round(np.linalg.norm(self.nodes[i].coords - self.nodes[j].coords),3))
                    if dist < round(0.2,3):
                        self.nodes[i].other_feature_types = [x+y for x,y in zip(self.nodes[i].other_feature_types,self.onek_encoding_unk(self.nodes[j].feature_type, ELEM_LIST))]
                        self.nodes[j].index = -1
                        continue

                    self.distMatrix[i, j] = dist
                    self.distMatrix[j, i] = dist


    def _calcMolProperties(pdb_mol):
        Chem.calcImplicitHydrogenCounts(pdb_mol, True)
        Chem.perceiveHybridizationStates(pdb_mol, True)        
        Chem.makeHydrogenComplete(pdb_mol)
        Chem.setAtomSymbolsFromTypes(pdb_mol, False)        
        Chem.calcImplicitHydrogenCounts(pdb_mol, True)
        Chem.setRingFlags(pdb_mol, True)
        Chem.setAromaticityFlags(pdb_mol, True)
        Chem.generateHydrogen3DCoordinates(pdb_mol, True)   
        Biomol.setHydrogenResidueSequenceInfo(pdb_mol, False) 