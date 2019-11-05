import tensorflow as tf
import numpy as np
from phagraphnn.utilities import indexSelect, createVar, getConnectedFeatures, updateConnectedDict
from phagraphnn.Gru import GraphGRU

ELEM_LIST =[0,1,2,3,4,5,6,7] #0=unk,1=H,2=AR,3=NI,4=PI, 5=HBD,6=HBA,7=XV
FEATURE_FDIM = len(ELEM_LIST)
EDGE_FDIM = 1
ALL_FDIM = FEATURE_FDIM+EDGE_FDIM
MAX_NB = 15
MAX_NR_SURROUND_FEATURES = 30
MAX_NR_FEATURES = 50

class PhaGruMPN3(tf.keras.Model):
    '''
    This class uses a gru as an update function. And does embedding over
    the origin feature vector plus the distance instead of only the distance.
    Moreover, it uses depth number of gru networks (all different).
    '''
    
    def __init__(self, hidden_size, depth,output_nn=None):
        super(PhaGruMPN3, self).__init__(name="PhaGruMPN3")
        self.hidden_size = hidden_size
        self.depth = depth
        self.hidden_size_bond = int(hidden_size*2)
        self.input_size_bond = ALL_FDIM

        self.W_i_a = tf.keras.layers.Dense(self.hidden_size,input_shape=(FEATURE_FDIM,),
            name="W_i_a",use_bias=False,activity_regularizer=tf.keras.regularizers.l2(0.025),
            kernel_initializer=tf.keras.initializers.he_normal())
        self.W_i_b = tf.keras.layers.Dense(self.hidden_size,input_shape=(self.input_size_bond,),
            name="W_i_b",use_bias=False,activity_regularizer=tf.keras.regularizers.l2(0.025),
            kernel_initializer=tf.keras.initializers.he_normal())
        
        self.W_h = tf.keras.layers.Dense(self.hidden_size,
            input_shape=(self.hidden_size,),name="W_h",use_bias=False,
            activity_regularizer=tf.keras.regularizers.l2(0.025),
            kernel_initializer=tf.keras.initializers.he_normal())
        
        self.grus = []

        for i in range(self.depth):
            self.grus.append(GraphGRU(ALL_FDIM,self.hidden_size,1))

        self.output_nn = output_nn


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