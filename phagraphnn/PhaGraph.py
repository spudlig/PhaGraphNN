import CDPL.Chem as Chem
import CDPL.Biomol as Biomol
import CDPL.Pharm as Pharm
import CDPL.Base as Base
import CDPL.ConfGen as ConfGen
import CDPL.Math as Math

import numpy as np
import sys
from copy import deepcopy
import math

### all feature types considered (8) - XV are currently not used
ELEM_LIST =[0,1,2,3,4,5,6,7] #0=unk,1=H,2=AR,3=NI,4=PI, 5=HBD,6=HBA,7=XV

class PhaNode():
    '''  
    This class represents a feature of the pharmacophore graph - if there are
    more than one feature at the same position, other_feature_types is used.
    '''
    def __init__(self,feature_type):
        self.feature_type = feature_type
        self.coords = np.array([0.0, 0.0, 0.0])
        self.index = 0
        self.other_feature_types = [0,0,0,0,0,0,0,0] # as soon as there are two or more features at one position, it is added here

class PhaGraph():
    ''' This class is the pharmacophore graph class.
    It can be utilized to represent any kind of Pharmacophore.
    Upon initialization, there is no calculations being done. If you
    want to generate the graph, use the call method. \n
    Currently all features are connected to all other features,
    however, in a later version, this can also be split up and only
    features within a certain range are all interconnected.

    '''
    def __init__(self):
        self.nodes = list() # PhaNode pbjects
        self.edges = set() # Set of all pairwise connected features
        self.edge_weights = dict() # the distance between the center of feature1 to feature2
        
        self.properties = dict()
        self.name = None

    def __call__(self,pharmacophore):
        self._generateNodes(pharmacophore)
        self._calcDistMatrix()
        self._generateEdges()

    def getProperty(self,property_name):
        '''
        Input \n
        property_name (string): Demanded property name \n
        Returns \n
        (string): containing the "property_name" property with property_name \n
        (ValueError): graph does not contain "property_name"
        '''
        if property_name in properties.keys():
            return self.properties[property_name]
        else:
            raise ValueError("Property name",property_name,"does not exist. Only those names are present:",properties.keys())


    def setProperty(self,property_name,property_):
        '''
        Input \n
        property_name (string): the property name the property string should be saved
        under.\n
        property_ (string): the "real" property, that should be saved.
        '''
        self.properties[property_name] = property_

    def getName(self):
        return self.name

    def setName(self,name):
        self.name = name
    
    def distance(self, node1, node2):
        '''
        Input \n
        node1 (AtomNode): origin node \n
        node2 (AtomNode): target node \n
        Returns \n
        (float): the distance between origin node1 and target node2
        '''
        return float(self.distMatrix[node1.index, node2.index])

    def distance_idx(self, idx1, idx2):
        '''
        Returns the distance between node with index 1 and node with index 2
        Input \n
        idx1 (int): index of origin node \n
        idx2 (int): index of target node \n
        Returns \n
        (float): the distance between origin node1 and target node2
        '''
        return float(self.distMatrix[idx1, idx2])

    def connected(self, node1, node2):
        '''
        Returns True if the nodes are connected \n
        node1 (PhaNode): origin node \n
        node2 (PhaNode): target node \n
        Returns \n
        (boolean): False if the nodes are not connected - True otherwise
        '''
        if (node1.index, node2.index) in self.edges:
            return True

        if (node2.index, node1.index) in self.edges:
            return True

        return False

    def _generateNodes(self,pha):
        ''' 
        PRIVATE METHOD
        generates the nodes of the graph \n
        Input \n
        pha (CDPL BasicPharmacophore): pha the graph is based on
        '''
        index_counter = 0
        for feature in pha:
            node = PhaNode(Pharm.getType(feature))
            node.other_feature_types = self._getAllowedSet(Pharm.getType(feature), ELEM_LIST)
            node.coords[0]= round(Chem.get3DCoordinates(feature)[0],6)
            node.coords[1]= round(Chem.get3DCoordinates(feature)[1],6)
            node.coords[2]= round(Chem.get3DCoordinates(feature)[2],6)
            node.index = index_counter
            index_counter += 1
            self.nodes.append(node)

    def _calcDistMatrix(self):
        ''' 
        PRIVATE METHOD
        generates the edges for the graph \n
        '''
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
                        self.nodes[i].other_feature_types = [x+y for x,y in zip(self.nodes[i].other_feature_types,self._getAllowedSet(self.nodes[j].feature_type, ELEM_LIST))]
                        self.nodes[j].index = -1
                        continue

                    self.distMatrix[i, j] = dist
                    self.distMatrix[j, i] = dist

    def _generateEdges(self):
        ''' 
        PRIVATE METHOD
        generates the edges for the graph \n
        '''
        for i in range(0, len(self.nodes)):
            phan1 = self.nodes[i]

            for j in range(i+1, len(self.nodes)):
                phan2 = self.nodes[j]
                if phan1.index == phan2.index:continue

                dist = self.distance(phan1,phan2)

                self.edges.add((i, j))
                self.edge_weights[(i,j)] = (dist)


    def _getAllowedSet(self,x, allowable_set):
        ''' 
        PRIVATE METHOD 
        generates the edges for the graph \n
        Input \n
        mol (CDPL BasicMolecule): molecule the edges are calculated for
        '''
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))