import tensorflow as tf
import numpy as np
from phagraphnn.PhaGAT import GATLayer as GATLayer
from phagraphnn.PhaGAT import FEATURE_FDIM,ALL_FDIM
from phagraphnn.utilities import indexSelect,getConnectedFeatures, updateConnectedDict


class PhaGatModel(tf.keras.Model):
    '''
    This class uses a GAT as an update function. And does embedding over
    the origin feature vector plus the distance.
    It only uses one GAT network over num_iters length.
    '''
    

    def __init__(self, hidden_dim = 8, out_dim = 10, dropout_rate = 0.001 , num_iters = 2,
                emb_initializer = tf.random_uniform_initializer(0.1,0.9),output_nn=None):
        super(PhaGatModel, self).__init__(name='PhaGatModel')

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.num_iters = num_iters

        self.heads = []
        self.output_nn = output_nn

        print("self.hidden_dim",self.hidden_dim)
        print("self.out_dim",self.out_dim)
        print("self.dropout_rate",self.dropout_rate)

        self.GATLayer = GATLayer(self.hidden_dim,self.out_dim,self.dropout_rate)

        self.embedding = tf.keras.layers.Dense(self.hidden_dim, input_shape=(FEATURE_FDIM,),
            name="embedding",activation=None,
            kernel_initializer=emb_initializer)
        self.dist_embedding = tf.keras.layers.Dense(self.hidden_dim, input_shape=(ALL_FDIM,),
            name="dist_embedding",activation=None,
            kernel_initializer=emb_initializer)
        
    def call(self, x_batch):

        ### GAT with embedding:
        target_features,feature_dist_graph,rij_dist_pairs,b_scope,start_end_env,l_scope,scope_update,scope_update_lig =x_batch
        target_features = self.embedding(target_features)
        rij_dist_pairs = tf.reshape(rij_dist_pairs,(tf.shape(rij_dist_pairs)[0],1))
        message = tf.concat([feature_dist_graph,rij_dist_pairs ], axis=1)
        message = self.dist_embedding(message)

        target_features = tf.concat([[np.zeros(tf.shape(target_features)[1])],target_features],axis=0)
        for i in range(0,self.num_iters):
            multible_entry_f_lig = getConnectedFeatures(target_features,start_end_env)
            target_features = self.GATLayer(multible_entry_f_lig,message,b_scope)
            message = updateConnectedDict(target_features,scope_update_lig,scope_update)

        cmp_enc = tf.gather(target_features,indices=(l_scope),axis=0)
        mol_vecs = tf.reduce_sum(cmp_enc,1)

        if self.output_nn:
            return tf.reshape(self.output_nn(mol_vecs),[-1])
        return mol_vecs 


    def train(self, inputs, outputs, learning_rate):
        '''
        Trains the compiled model. Uses the defined optimizer and loss. 
        The learning rate is NOT taken from the optimizer, 
        needs to be applied here. \n
        INPUT: \n
        inputs (list of list): the batch, that is being returned by the 
                                tensorize method of the corresponding model. \n
        outputs (list): the batch of the "true" values. \n
        learning_rate (float): learning rate - needs to be defined \n
        RETURN: \n
        (list): batch sized list of predictions
        (list): batch size averaged loss
        '''
        loss = self.__getattribute__("loss")
        optimizer = self.__getattribute__("optimizer")
        optimizer._learning_rate = learning_rate
        with tf.GradientTape() as tape:
            predictions = self.call(inputs)
            current_loss = tf.reduce_mean(loss(outputs,predictions))
            grads = tape.gradient(current_loss, self.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.1)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return predictions,current_loss

    def evaluate(self, inputs, outputs):
        '''
        evaluates the trained model. Uses the defined loss.\n
        INPUT: \n
        inputs (list of list): the batch, that is being returned by the 
                                tensorize method of the corresponding model. \n
        outputs (list): the batch of the "true" values. \n
        RETURN: \n
        (list): batch sized list of predictions
        (list): batch size averaged loss
        '''
        loss = self.__getattribute__("loss")
        predictions = self.call(*inputs)
        current_loss = tf.reduce_mean(loss(outputs,predictions))
        return predictions,current_loss