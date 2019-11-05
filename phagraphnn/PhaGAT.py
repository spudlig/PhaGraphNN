import tensorflow as tf
import numpy as np
import copy
from phagraphnn.utilities import getConnectedFeatures, updateConnected

ELEM_LIST =[0,1,2,3,4,5,6,7] #0=unk,1=H,2=AR,3=NI,4=PI, 5=HBD,6=HBA,7=XV
FEATURE_FDIM = len(ELEM_LIST)
EDGE_FDIM = 1
ALL_FDIM = FEATURE_FDIM+EDGE_FDIM
MAX_NB = 15
MAX_NR_SURROUND_FEATURES = 30
MAX_NR_FEATURES = 50

class GATLayer(tf.keras.Model):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super(GATLayer, self).__init__(name='GATLayer')
        self.fc = tf.keras.layers.Dense(out_dim, input_shape=(in_dim,), use_bias=False,kernel_initializer=tf.keras.initializers.he_normal())
        self.fc_env = tf.keras.layers.Dense(out_dim, input_shape=(in_dim,), use_bias=False,kernel_initializer=tf.keras.initializers.he_normal())
        self.attn_fc = tf.keras.layers.Dense((1),input_shape=(out_dim*2,), use_bias=False,kernel_initializer=tf.keras.initializers.he_normal(),
                activity_regularizer=tf.keras.regularizers.l2(dropout_rate))

    def attention_score(self, f_lig,f_env):
        z2 = tf.concat([f_lig, f_env], axis=1)
        e = self.attn_fc(z2)
        return tf.nn.softplus(e)

    def reduce_func(self, e,z_env,scope):
        e = tf.concat([[[0]],e],axis=0)
        e = tf.gather(e,indices=(scope))
        f_shape =tf.shape(z_env)[1]
        z_env = tf.concat([[tf.zeros(f_shape)],z_env],axis=0)
        z_env = tf.gather(z_env,indices=(scope))

        f_lig_up = tf.convert_to_tensor([tf.zeros(f_shape)])
        for n_lig_atm in tf.range(0,tf.shape(e)[0]):
            e_i_j = e[n_lig_atm]
            z_j = z_env[n_lig_atm]
            non_zeros = ~tf.equal(e_i_j, 0)
            alpha_i_j = tf.nn.softmax(tf.boolean_mask(e_i_j, non_zeros,axis=0))
            padding = tf.zeros((tf.shape(z_j)[0]-tf.shape(alpha_i_j)[0]))
            alpha_i_j = tf.reshape(tf.concat([alpha_i_j,padding],0),(tf.shape(z_j)[0],1))
            inter = tf.math.multiply(z_j,alpha_i_j)
            cont = tf.reduce_sum(inter,0, keepdims=True)
            f_lig_up = tf.concat([f_lig_up,cont],0)
        return f_lig_up

    def update(self,e,z_env,scope):
        return self.reduce_func(e,z_env,scope)

    def call(self, z_feature,z_others,scope):
        # z_lig = self.fc(f_lig_mpn)
        # z_env = self.fc_env(f_env)
        e = self.attention_score(z_feature,z_others)
        return self.update(e,z_others,scope)

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

