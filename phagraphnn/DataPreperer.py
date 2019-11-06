import os, random
import logging
log = logging.getLogger(__name__)
import pickle as pickle
from phagraphnn.PhaGatModel import *
from phagraphnn.PhaGruMPN import *
import time

class DataPreparer(object):
    '''
    This class prepares the graph representation into a format, that the corresponding
    NN can work with. \n
    INPUT: \n
    data_folder (list): the list of graphs \n
    batch_size (int): the size of the batch \n
    property_string (string): the property in the graph that needs to be predicted \n
    num_workers (int): how many cpus you want to use (default=4) \n
    shuffle (boolean): do you want to shuffle your dataset (default=True) \n
    mpn (string): what kind of NN do you want to use. Either 'mpn' or 'gru' is currently
    allowed (default='mpn') \n
    OUTPUT: \n
    (GraphDataset): class that needs to be iterated over in order to get each batch
    '''
    def __init__(self, data_folder, batch_size,property_string, num_workers=4, shuffle=True,mpn="mpn"):
        self.data_folder = data_folder
        self.data_files = []
        # self.is_path = is_path
        # if self.is_path:
        #     self.data_files = [fn for fn in os.listdir(data_folder)]
        # else:
        self.data_files = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mpn = mpn
        self.property_string = property_string

    def __iter__(self):
        # if self.is_path:
        #     for fn in self.data_files:
        #         fn = os.path.join(self.data_folder, fn)
        #         with open(fn,'rb') as f:
        #             data = pickle.load(f)
        #         dataset = self._getDataset(data)
        #         for b in dataset:
        #             yield b
        #         del data, dataset
        #     else:
        #         return
        # else:
        dataset = self._getDataset(self.data_files)
        for b in dataset:
            yield b
        del dataset

    def _getDataset(self,data):
        random.seed(time.time()) # always random
        # random.seed(100)
        if self.shuffle: 
            random.shuffle(data)
        
        batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        try:
            if len(batches[-1]) < self.batch_size:
                batches.pop()
        except Exception as e:
            print("error with batch:",e)

        return GraphDataset(batches,self.mpn,self.property_string)


class GraphDataset():

    def __init__(self, data,mpn,property_string):
        self.data = data
        self.mpn = mpn
        self.property_string = property_string

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx],self.mpn,self.property_string)

def tensorize(graph_batch,mpn,property_string):
    if mpn == "gat":
        return PhaGatModel.tensorize(graph_batch,property_string)
    if mpn == "gru":
        return PhaGruMPN.tensorize(graph_batch,property_string)