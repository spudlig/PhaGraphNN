import tensorflow as tf
import numpy as np
import phagraphnn.PhaGatModel
from phagraphnn.PhaGAT import GATLayer as GATLayer
from phagraphnn.utilities import indexSelect,getConnectedFeatures, updateConnectedDict
import logging
log = logging.getLogger(__name__)

class PhaGatModel2(phagraphnn.PhaGatModel):
    '''
    This class uses a GAT as an update function. And does embedding over
    the origin feature vector plus the distance.
    Moreover, it uses num_heads of GAT networks (all different).
    The output of each layer can be either concated (merge='cat'), multiplied (merge='mul'),
    added (merge='add') or just left as is (merge='none') with the original input.
    '''

    def __init__(self, hidden_dim = 8, out_dim = 10, dropout_rate = 0.001 , num_heads = 4,
                merge='cat',emb_initializer = tf.random_uniform_initializer(0.1,0.9),output_nn=None,
                regression = False):
        super().__init__(hidden_dim = hidden_dim, out_dim = out_dim, dropout_rate = dropout_rate ,
                emb_initializer = emb_initializer,output_nn=output_nn,regression = regression)
        self.num_heads = num_heads
        self.merge = merge

        if self.merge == 'cat':
            for head in range(1,self.num_heads):
                self.heads.append(GATLayer(self.hidden_dim*head,self.out_dim,self.dropout_rate))
        else:
            for head in range(1,self.num_heads):
                self.heads.append(GATLayer(self.hidden_dim*head,self.out_dim,self.dropout_rate))

    def call(self, x_batch):

        ### GAT with embedding:
        target_features_orig,feature_dist_graph,rij_dist_pairs,b_scope,start_end_env,l_scope,scope_update,scope_update_lig =x_batch
        target_features_orig = self.embedding(target_features_orig)
        rij_dist_pairs = tf.reshape(rij_dist_pairs,(tf.shape(rij_dist_pairs)[0],1))
        message = tf.concat([feature_dist_graph,rij_dist_pairs ], axis=1)
        message = self.dist_embedding(message)

        target_features = tf.concat([[np.zeros(tf.shape(target_features_orig)[1])],target_features_orig],axis=0)
        target_features_orig = tf.concat([[np.zeros(tf.shape(target_features_orig)[1])],target_features_orig],axis=0)
        for i in range(0,self.num_heads-1):
            multible_entry_f_lig = getConnectedFeatures(target_features,start_end_env)
            target_features = self.heads[i](multible_entry_f_lig,message,b_scope)
            target_features = self._update_target_features(target_features,target_features_orig)
            message = updateConnectedDict(target_features,scope_update_lig,scope_update)

        cmp_enc = tf.gather(target_features,indices=(l_scope),axis=0)
        mol_vecs = tf.reduce_sum(cmp_enc,1)

        try:
            if self.output_nn and self.regression:
                return tf.reshape(self.output_nn(mol_vecs),[-1])
            elif self.output_nn:
                return self.output_nn(mol_vecs)
            return mol_vecs 
        except Exception as e:
            log.error('There seems to be an issue with the input dimensions.'+
                'Please check the input dimension of the output_nn you defined outside.'+
                'Currently the NN output dimension is:'+str(self.num_heads*self.emb_dim),e)

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
        