import tensorflow as tf
import numpy as np
from phagraphnn.utilities import indexSelect, createVar, getConnectedFeatures, updateConnectedDict
from phagraphnn.Gru import GraphGRU
from phagraphnn.PhaGruMPN import *

class PhaGruMPN3(PhaGruMPN):
    '''
    This class uses a gru as an update function. And does embedding over
    the origin feature vector plus the distance instead of only the distance.
    Moreover, it uses depth number of gru networks (all different).
    '''
    
    def __init__(self, hidden_size, depth,output_nn=None):
        super().__init__(hidden_size=hidden_size, depth=depth,output_nn=output_nn)
        self.hidden_size_bond = int(hidden_size*2)
        self.input_size_bond = ALL_FDIM

        self.W_i_a = tf.keras.layers.Dense(self.hidden_size,input_shape=(FEATURE_FDIM,),
            name="W_i_a",use_bias=False,activity_regularizer=tf.keras.regularizers.l2(0.025),
            kernel_initializer=tf.keras.initializers.he_normal())
        self.W_i_b = tf.keras.layers.Dense(self.hidden_size,input_shape=(self.input_size_bond,),
            name="W_i_b",use_bias=False,activity_regularizer=tf.keras.regularizers.l2(0.025),
            kernel_initializer=tf.keras.initializers.he_normal())
        
        self.grus = []

        for i in range(self.depth):
            self.grus.append(GraphGRU(ALL_FDIM,self.hidden_size,1))

    def call(self,x_batch):
        target_features,feature_dist_graph,rij_dist_pairs,b_scope,start_end_env,l_scope,scope_update,scope_update_lig =x_batch
        target_features = createVar(target_features)
        feature_dist_graph = createVar(feature_dist_graph)
        rij_dist_pairs = createVar(rij_dist_pairs)

        target_features = self.W_i_a(target_features)

        rij_dist_pairs = tf.reshape(rij_dist_pairs,(tf.shape(rij_dist_pairs)[0],1))
        rij_dist_pairs = tf.concat([feature_dist_graph,rij_dist_pairs ], axis=1)
        rij_dist_pairs = self.W_i_b(rij_dist_pairs)
        rij_dist_pairs = tf.keras.activations.relu(rij_dist_pairs)
        message = self.W_h(rij_dist_pairs)
        
        message = tf.concat([[np.zeros(32)],message],axis=0)
        message = tf.gather(message,indices=(b_scope))
        message = tf.reduce_sum((message), 1)

        for i in range(self.depth):
            target_features = self.grus[i](message,target_features)
            message = tf.convert_to_tensor(updateConnectedDict(target_features,scope_update_lig,scope_update))
            message = self.W_h(message)
            message = tf.concat([[np.zeros(32)],message],axis=0)
            message = tf.gather(message,indices=(b_scope))
            message = tf.reduce_sum(message, 1)
        
        feature_hiddens = tf.concat([[np.zeros(32)],target_features],axis=0)
        mol_vecs = tf.gather(feature_hiddens,indices=(l_scope),axis=0)
        mol_vecs = tf.reduce_sum(mol_vecs,1)

        if self.output_nn:
            return tf.reshape(self.output_nn(mol_vecs),[-1])
        return mol_vecs 