import tensorflow as tf
import numpy as np
from phagraphnn.PhaGAT import GATLayer as GATLayer
from phagraphnn.PhaGAT import FEATURE_FDIM,ALL_FDIM
from phagraphnn.utilities import indexSelect,getConnectedFeatures, updateConnectedDict
import logging
log = logging.getLogger(__name__)

class PhaGatModel2(tf.keras.Model):
    '''
    This class uses a GAT as an update function. And does embedding over
    the origin feature vector plus the distance.
    Moreover, it uses num_heads of GAT networks (all different).
    The output of each layer can be either concated (merge='cat'), multiplied (merge='mul'),
    added (merge='add') or just left as is (merge='none').
    '''

    def __init__(self, hidden_dim = 8, out_dim = 10, emb_dim = 10, dropout_rate = 0.001 , num_heads = 4,
                merge='cat',emb_initializer = tf.random_uniform_initializer(0.1,0.9),output_nn=None):
        super(PhaGatModel2, self).__init__(name='PhaGatModel2')

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.merge = merge

        self.heads = []
        self.output_nn = output_nn

        if self.merge == 'cat':
            for head in range(1,self.num_heads):
                self.heads.append(GATLayer(self.emb_dim*head,self.out_dim,self.dropout_rate))
        else:
            for head in range(1,self.num_heads):
                self.heads.append(GATLayer(self.emb_dim*head,self.out_dim,self.dropout_rate))

        self.embedding = tf.keras.layers.Dense(self.emb_dim, input_shape=(FEATURE_FDIM,),
            name="embedding",activation=None,
            kernel_initializer=emb_initializer)
        self.dist_embedding = tf.keras.layers.Dense(self.emb_dim, input_shape=(ALL_FDIM,),
            name="dist_embedding",activation=None,
            kernel_initializer=emb_initializer)
        
    def call(self, x_batch):

        ### GAT with embedding:
        target_features_orig,feature_dist_graph,rij_dist_pairs,b_scope,start_end_env,l_scope,scope_update,scope_update_lig =x_batch
        target_features_orig = self.embedding(target_features_orig)
        rij_dist_pairs = tf.reshape(rij_dist_pairs,(tf.shape(rij_dist_pairs)[0],1))
        message = tf.concat([feature_dist_graph,rij_dist_pairs ], axis=1)
        message = self.dist_embedding(message)

        target_features_orig = tf.concat([[np.zeros(tf.shape(target_features_orig)[1])],target_features_orig],axis=0)
        for i in range(0,self.num_heads-1):
            multible_entry_f_lig = getConnectedFeatures(target_features_orig,start_end_env)
            target_features = self.heads[i](multible_entry_f_lig,message,b_scope)
            target_features = self._update_target_features(target_features,target_features_orig)
            message = updateConnectedDict(target_features,scope_update_lig,scope_update)

        cmp_enc = tf.gather(target_features,indices=(l_scope),axis=0)
        mol_vecs = tf.reduce_sum(cmp_enc,1)

        try:
            if self.output_nn:
                return tf.reshape(self.output_nn(mol_vecs),[-1])
            return mol_vecs 
        except Exception as e:
            log.error('There seems to be an issue with the input dimensions.'+
                'Please check the input dimension of the output_nn you defined outside.'+
                'Currently the NN output dimension is:'+str(self.num_heads*self.emb_dim),e)


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

    def _update_target_features(self,features_new,features_old):
        '''
        PRIVATE METHOD \n
        depending on the self.merge flag, how sould the different
        outcomes of the GAT layer be processed.
        '''
        if self.merge == 'cat':
            return tf.concat([features_new,features_old],axis=1)
        if self.merge == 'mul':
            return tf.math.multiply([features_new,features_old])
        if self.merge == 'add':
            return tf.math.add([features_new,features_old])
        if self.merge == 'none':
            return features_new
        
        log.error("Please define a valid option ('cat','mul','add' or 'none') for"+
                "the PhaGatModel2 merge flat. currently:",self.merge)
        