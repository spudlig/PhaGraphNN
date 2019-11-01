import tensorflow as tf
import numpy as np
from phagraphnn.PhaGAT import *


class PhaGatModel(tf.keras.Model):
    ### this model uses a pha clique graph and then does predictions 

    def __init__(self, hidden_dim = 8, out_dim = 10, emb_dim = 10, dropout_rate = 0.00 , num_heads = 2,
                merge='cat',emb_initializer = tf.random_uniform_initializer(0.1,0.9)):
        super(PhaGatModel, self).__init__(name='PhaGatModel')

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.merge = merge

        self.heads = []

        for head in range(1,num_heads):
            self.heads.append(GATLayer(self.emb_dim*head,self.out_dim,self.dropout_rate))

        initializer = emb_initializer
        self.embedding = tf.keras.layers.Dense(self.emb_size, input_shape=(FEATURE_FDIM,),
            name="embedding",activation=None,activity_regularizer=self.dropout_rate,
            kernel_initializer=self.initializer)
        self.dist_embedding = tf.keras.layers.Dense(emb_size, input_shape=(ALL_FDIM,),
            name="dist_embedding",activation=None,activity_regularizer=self.dropout_rate,
            kernel_initializer=self.initializer)
        
        self.W_out = tf.keras.layers.Dense((2), input_shape=(self.hidden_inter_out_size,),
            name="W_out",activation="softmax")


    def call(self, x_batch):

        ### GAT with embedding:
        target_features,feature_dist_graph,rij_dist_pairs,b_scope,affinities,l_scope,names,start_end_env =x_batch
        target_features = self.embedding(target_features)
        feature_dist_graph = self.dist_embedding(feature_dist_graph)

        target_features = tf.concat([[np.zeros(tf.shape(target_features)[1])],target_features],axis=0)

        gat = tf.gather(target_features,indices=(start_end_env)) # get for each target feature entry all other features
        gat = tf.reshape(gat,(tf.shape(gat)[0]*tf.shape(gat)[1],tf.shape(gat)[2])) # remove the 3. shape introduced
        intermediate_tensor = tf.reduce_sum(tf.abs(gat), 1) # search for all the 0 entries introduced by the gather and concat above
        zero_vector = tf.zeros(shape=(tf.shape(gat)[0]), dtype=tf.float32) # generate dummy matrix for comparison
        bool_mask = tf.not_equal(intermediate_tensor, zero_vector) # compare dummy with other matrix
        multible_entry_f_lig = tf.boolean_mask(gat, bool_mask)
        print("multible_entry_f_lig",tf.shape(multible_entry_f_lig))
        
        cmp_enc = self.PhaGAT(multible_entry_f_lig,feature_dist_graph,b_scope)

        print("cmp_enc",tf.shape(cmp_enc))

