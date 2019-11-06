import logging
log = logging.getLogger(__name__)
import tensorflow as tf
import numpy as np
from phagraphnn.PhaGAT import GATLayer as GATLayer
from phagraphnn.utilities import indexSelect,getConnectedFeatures, updateConnectedDict


ELEM_LIST =[0,1,2,3,4,5,6,7] #0=unk,1=H,2=AR,3=NI,4=PI, 5=HBD,6=HBA,7=XV
FEATURE_FDIM = len(ELEM_LIST)
EDGE_FDIM = 1
ALL_FDIM = FEATURE_FDIM+EDGE_FDIM
MAX_NB = 15
MAX_NR_SURROUND_FEATURES = 30
MAX_NR_FEATURES = 50

class PhaGatModel(tf.keras.Model):
    '''
    This class uses a GAT as an update function. And does embedding over
    the origin feature vector plus the distance.
    It only uses one GAT network over num_iters length.
    '''
    

    def __init__(self, hidden_dim = 8, out_dim = 10, dropout_rate = 0.001 , num_iters = 2,
                emb_initializer = tf.random_uniform_initializer(0.1,0.9),output_nn=None,regression=True):
        super(PhaGatModel, self).__init__(name='PhaGatModel')

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.num_iters = num_iters
        self.regression = regression

        self.heads = []
        self.output_nn = output_nn

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


    @staticmethod
    def tensorize(graph_batch,property_name, cutoff=9.0):
        '''

        '''
        b_scope = []
        start_end_env = []
        l_scope = []
        properties = []
        scope_update = []
        scope_update_lig = dict()

        # fatm_dist_padding =np.zeros(ATOM_FDIM_ENV)
        feature_dist_graph = []
        rij_dist_pairs = []
        target_features = []
        total_atoms = 0 
        total_t_features = 1 # needs to be one, because of the gather function and separate from total_atoms (as long 
        # as its going for a MPN)
        total_other_features = 1 # needs to be one, because of the gather function
        names = [] 
        feature_n = 1
        update_n_features = 0
        graph_nr = 0
        for graph in graph_batch:
            # affinities.append(graph.affinity)
            properties.append(graph.getProperty(property_name))
            names.append(graph.getName())
            pha = graph.nodes
            n_features = 0
            total_update_features = 0
            graph_nr += 1
            for feature in pha:
                if feature.index == -1: continue
                n_other_f = 0
                scope_update_lig[str(feature.index)+"_"+str(graph_nr)] = update_n_features
                update_n_features += 1
                n_features +=1
                for other_feature in pha:
                    if other_feature.index == -1: continue
                    if feature.index == other_feature.index: continue
                    distance = graph.distance(feature,other_feature)
                    if distance > cutoff:continue
                    rij_dist_pairs.append(tf.convert_to_tensor(tf.cast(distance,tf.float32)))
                    # inter = copy.copy(other_feature.feature_type)
                    # inter.append(distance)
                    feature_dist_graph.append(other_feature.feature_type)
                    scope_update.append(str(other_feature.index)+"_"+str(graph_nr))
                    n_other_f +=1
                    total_update_features +=1
                target_features.append(feature.feature_type)
                feature_n = tf.cast(feature_n,tf.int32)
                range_dist = tf.range(total_other_features,total_other_features+ n_other_f)
                range_dist_2 = np.repeat(feature_n,n_other_f)
                if len(range_dist) < MAX_NR_SURROUND_FEATURES:
                    padding_d = np.repeat(0,(MAX_NR_SURROUND_FEATURES-len(range_dist)))
                    range_dist= tf.concat([range_dist,padding_d],0)
                    range_dist = tf.stack(range_dist)
                    b_scope.append(range_dist)
                    padding_d_2 = np.repeat(0,(MAX_NR_SURROUND_FEATURES - len(range_dist_2)))
                    range_dist_2 =tf.concat([range_dist_2,padding_d_2],0)
                    range_dist_2 = tf.stack(range_dist_2)
                    start_end_env.append(range_dist_2)
                feature_n += 1
                total_other_features += n_other_f
            range_lig = tf.range(total_t_features,total_t_features+ n_features)
            total_atoms += n_features
            total_t_features += n_features
            if len(range_lig) < MAX_NR_FEATURES:
                padding_l = np.repeat(0,(MAX_NR_FEATURES - len(range_lig)))
                range_lig= tf.concat([range_lig,padding_l],0)
                range_lig = tf.stack(range_lig)
                l_scope.append(range_lig) 
        feature_dist_graph = tf.stack(feature_dist_graph,0)
        target_features = tf.stack(target_features,0)
        rij_dist_pairs = tf.stack(rij_dist_pairs,0)
        properties = tf.stack(properties, 0)
        return (target_features,feature_dist_graph,rij_dist_pairs,b_scope,start_end_env,l_scope,scope_update,scope_update_lig),properties,names

