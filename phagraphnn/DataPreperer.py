import os, random
import logging
log = logging.getLogger(__name__)
import pickle as pickle
from phagraphnn.PhaGatModel import *
from phagraphnn.PhaGruMPN import *
import time

class DataPreparer(object):

    def __init__(self, data_folder, batch_size,property_string, num_workers=4, shuffle=True,mpn="mpn",is_path = True):
        self.data_folder = data_folder
        self.data_files = []
        self.is_path = is_path
        if self.is_path:
            self.data_files = [fn for fn in os.listdir(data_folder)]
        else:
            self.data_files = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mpn = mpn
        self.property_string = property_string

    def __iter__(self):
        if self.is_path:
            for fn in self.data_files:
                fn = os.path.join(self.data_folder, fn)
                with open(fn,'rb') as f:
                    data = pickle.load(f)
                dataset = self._getDataset(data)
                for b in dataset:
                    yield b
                del data, dataset
            else:
                return
        else:
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