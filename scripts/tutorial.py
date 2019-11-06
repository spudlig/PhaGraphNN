#%%

##### This tutorial should introduce the most important features in the
##### graph classification/regression tasks and show, how to work with this
##### library.
##### The first part is the regression part and the second one is concerned with
##### classification.
##### As there are several types of networks, there will only be a showcase regarding
##### the two main different ones - in particular the multihead versions and
##### the single layer use case.

#%%
# first step is to load the necessary classes #
import tensorflow as tf 
import phagraphnn.utilities as ut # a utility class with several useful methods
from phagraphnn.PhaGraph import PhaGraph,PhaNode # the graph representation
from phagraphnn.DataPreperer import DataPreparer # prepares the data for the NN

#%%
# as we only are using two different types of the same NN class#
from phagraphnn.PhaGatModel import PhaGatModel as gat 
from phagraphnn.PhaGatModel3 import PhaGatModel3 as gat3

# There are the other possible NNs - look into
# the description of how exactly they differ. 
# Overall, they can be used in a similar fashion
# and therefore are not being used in this tutorial.
# from phagraphnn.PhaGruMPN import PhaGruMPN as gru
# from phagraphnn.PhaGruMPN2 import PhaGruMPN2 as gru2
# from phagraphnn.PhaGruMPN3 import PhaGruMPN3 as gru3
# from phagraphnn.PhaGatModel2 import PhaGatModel2 as gat2

#%%
# Data loading and graph generation:
# The utility class has a lot of different possibilities of how to get your data
# into the graph readable format. You can either use sdf/smiles or pdb files
# for your pharmacophore generation. In addition, there is an xls reader for your
# chembl dataset (if saved as an xls). 
# The first step is to load the data into a molecular representation. #

# load smiles via xls
data = ut.readChemblXls("./tests/data/CHE_3.xls")

# then put them into a molecular representation. 
# As we want to create pharmacophores, we need 3D
# information, so we generate one random conformation. (This will 
# hopefully be change in the future).
mol =ut.CDPLmolFromSmiles(data[2][1],True)

# then we need to generate the pharmacophorem, the graph is based on.
# as we only want the ligand based pharmacophore, we can choose to
# define None for the protein, add the mol and define 'lig_only'.#
lig_pha = ut.CDPLphaGenerator(None,mol,"lig_only")

# now we can initialized the graph#
graph = PhaGraph()
# then we can fill it with the pha #
graph(lig_pha)
# then we add certain properties - the ones we want to predict.
# as it is saved in a dictionary, one can define as many
# properties (keys) as they want - with the correspronding
# values of course. in this case it is the ic50 value.#
graph.setProperty("ic50",data[2][2])

# if there are more entries, you can also iterate over them,
# generate the graph and
# save the important properties there (can be different ones)#

graph_list = []
for i in range(0,len(data)):
    graph = PhaGraph()
    mol = ut.CDPLmolFromSmiles(data[i][1],True)
    graph(ut.CDPLphaGenerator(None,mol,"lig_only"))
    graph.setProperty("ic50",data[i][2])
    graph_list.append(graph)

#%%
# The next step is to load the graph representation
# into a format, that the neural network can work with.
# Therefore we need the DataPreperer class.
# Here we need to put the list of graphs, the name of the property
# we want to prediction, what kind of neural network we want to use. 
# the mpn options are currently either 'gru' or 'mpn'. #
loader = DataPreparer(graph_list,3,property_string="ic50",mpn="mpn")


#%%
### load data
data = ut.readChemblXls("./tests/data/CHE_3.xls")
graph_list = []
for i in range(0,len(data)):
    graph = PhaGraph()
    mol = ut.CDPLmolFromSmiles(data[i][1],True)
    graph(ut.CDPLphaGenerator(None,mol,"lig_only"))
    graph.setProperty("ic50",(0,1))
    graph_list.append(graph)
loader = DataPreparer(graph_list,3,property_string="ic50",mpn="gat",is_path=False)

#%%
###generate network

# this is the MLP part and is put after the "encoding" of the graph 
seq = tf.keras.Sequential([
tf.keras.layers.Dense(16, activation='relu', input_shape=(128,),name="first_layer"),
tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
tf.keras.layers.Dense(2,activation= 'softmax')],name="output_NN")

seq2 = tf.keras.Sequential([
tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
tf.keras.layers.Dense(1,activation= None)],name="output_NN")

seq3 = tf.keras.Sequential([
tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
tf.keras.layers.Dense(1,activation= None)],name="output_NN")

# here the model is defined
gat = gat(hidden_dim=32,output_nn=seq,regression=False)
gru = gru(32,1,seq2)
gru2 = gru2(32,1,seq3)
lr = 0.001
gat.compile(loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr))
gru.compile(loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr))
gru2.compile(loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr))
rec = tf.keras.metrics.MeanAbsoluteError()

#%%
### training
for batch in loader:
    inputs,af,other = batch
    gat(inputs)
    # gru(inputs)
    # gru2(inputs)
    for epoch in range(0,3000):
        GATpred,GATloss = gat.train(inputs=inputs,outputs=af,learning_rate=lr)
        # GRUpred,GRUloss = gru.train(inputs=inputs,outputs=af,learning_rate=lr)
        # GRU2pred,GRU2loss = gru2.train(inputs=inputs,outputs=af,learning_rate=lr)
        if epoch % 100 ==0:
            print("epoch",epoch)
            print("GAT: (pred,loss)",GATpred,GATloss)
            # print("GRU: (pred,loss)",GRUpred,GRUloss)
            # print("GRU2: (pred,loss)",GRU2pred,GRU2loss)
            print("PRED",af)


#%%
