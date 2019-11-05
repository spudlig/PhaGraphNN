
#%%

#%%
### get some important libraries
import tensorflow as tf
import phagraphnn.utilities as ut
from phagraphnn.PhaGraph import PhaGraph,PhaNode
from phagraphnn.DataPreperer import DataPreparer
from phagraphnn.PhaGatModel import PhaGatModel as gat
from phagraphnn.PhaGatModel2 import PhaGatModel2 as gat2
from phagraphnn.PhaGruMPN import PhaGruMPN as gru
from phagraphnn.PhaGruMPN2 import PhaGruMPN2 as gru2
from phagraphnn.PhaGruMPN3 import PhaGruMPN3 as gru3

#%%
### load data
data = ut.readChemblXls("./tests/data/CHE_3.xls")
graph_list = []
for i in range(0,len(data)):
    graph = PhaGraph()
    mol = ut.CDPLmolFromSmiles(data[i][1],True)
    graph(ut.CDPLphaGenerator(None,mol,"lig_only"))
    graph.setProperty("ic50",data[i][2])
    graph_list.append(graph)
loader = DataPreparer(graph_list,3,property_string="ic50",mpn="gat",is_path=False)


#%%
###generate network

# this is the MLP part and is put after the "encoding" of the graph 
seq_gat = tf.keras.Sequential([
tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
tf.keras.layers.Dense(1,activation= None)],name="output_NN")

seq_gat2 = tf.keras.Sequential([
tf.keras.layers.Dense(48, activation='relu', input_shape=(96,),name="first_layer"),
tf.keras.layers.Dense(16, activation='relu',name="second_layer"),
tf.keras.layers.Dense(1,activation= None)],name="output_NN")

seq_gru = tf.keras.Sequential([
tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
tf.keras.layers.Dense(1,activation= None)],name="output_NN")

seq_gru2 = tf.keras.Sequential([
tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
tf.keras.layers.Dense(1,activation= None)],name="output_NN")

seq_gru3 = tf.keras.Sequential([
tf.keras.layers.Dense(16, activation='relu', input_shape=(32,),name="first_layer"),
tf.keras.layers.Dense(8, activation='relu',name="second_layer"),
tf.keras.layers.Dense(1,activation= None)],name="output_NN")

# here the model is defined
gat = gat(hidden_dim=32,emb_dim=32,output_nn=seq_gat)
gat2 = gat2(hidden_dim=32,emb_dim=32,output_nn=seq_gat2,num_heads=3)
gru = gru(32,1,seq_gru)
gru2 = gru2(32,1,seq_gru2)
gru3 = gru3(32,1,seq_gru3)
lr = 0.001
gat.compile(loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr))
gat2.compile(loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr))
gru.compile(loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr))
gru2.compile(loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr))
gru3.compile(loss=tf.keras.losses.mse,
            optimizer=tf.keras.optimizers.RMSprop(lr))
rec = tf.keras.metrics.MeanAbsoluteError()



#%%
### training
for batch in loader:
    inputs,af,other = batch
    gat(inputs)
    gru(inputs)
    gru2(inputs)
    for epoch in range(0,3000):
        GATpred,GATloss = gat.train(inputs=inputs,outputs=af,learning_rate=lr)
        GAT2pred,GAT2loss = gat2.train(inputs=inputs,outputs=af,learning_rate=lr)
        GRUpred,GRUloss = gru.train(inputs=inputs,outputs=af,learning_rate=lr)
        GRU2pred,GRU2loss = gru2.train(inputs=inputs,outputs=af,learning_rate=lr)
        GRU3pred,GRU3loss = gru3.train(inputs=inputs,outputs=af,learning_rate=lr)
        if epoch % 100 ==0:
            print("epoch",epoch)
            print("GAT: (pred,loss)",GATpred,GATloss)
            print("GAT2: (pred,loss)",GAT2pred,GAT2loss)
            print("GRU: (pred,loss)",GRUpred,GRUloss)
            print("GRU2: (pred,loss)",GRU2pred,GRU2loss)
            print("GRU3: (pred,loss)",GRU3pred,GRU3loss)
            print("PRED",af)


#%%
