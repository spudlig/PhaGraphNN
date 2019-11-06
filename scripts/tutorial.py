#%%

##### This tutorial should introduce the most important features in the
##### graph classification/regression tasks.
##### the first part is the regression part and the second one is concerned with
##### classifications. 
##### As there are several types of networks, I am only going to show the two main
##### differences regarding the layer treatment - in particular the multihead
##### approach and the single one. In one the 

#%%
import tensorflow as tf
import phagraphnn.utilities as ut
from phagraphnn.PhaGraph import PhaGraph,PhaNode
from phagraphnn.DataPreperer import DataPreparer
from phagraphnn.PhaGatModel2 import PhaGatModel2 as gat
from phagraphnn.PhaGruMPN import PhaGruMPN as gru
from phagraphnn.PhaGruMPN2 import PhaGruMPN2 as gru2

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
