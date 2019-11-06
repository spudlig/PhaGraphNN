import tensorflow as tf
import numpy as np
from phagraphnn.PhaGruMPN import *
import logging
log = logging.getLogger(__name__)

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))

def get_features(feature):
    return tf.convert_to_tensor((onek_encoding_unk(feature.feature_type, ELEM_LIST)))

# def edge_features(bond):
#     return tf.convert_to_tensor(fbond + fstereo)

def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return tf.Variable(tensor,trainable=False)
    else:
        return tf.Variable(tensor,trainable=True)

def index_select_ND(source, dim, index):
    index_size = tf.shape(index)
    suffix_dim = tf.shape(source)[1:]
    final_size = tf.concat((index_size, suffix_dim),axis=0)
    inter = tf.reshape(index,shape=[-1])
    target = tf.gather(source,indices=inter,axis=dim)
    return tf.reshape(target,shape=final_size)

def getAffinity(affinity_string):
    micro_value = 0
    # more or less the same
    inter_string = affinity_string.replace("Ki=","").replace("Kd=","")
    ### put the affinity in the micro space and adjust the mili and nano to it
    try:
        if "pM" in inter_string:
            micro_value = float(inter_string.replace("pM",""))/1000/1000
        if "nM" in inter_string:
            micro_value = float(inter_string.replace("nM",""))/1000
        if "mM" in inter_string:
            micro_value = float(inter_string.replace("mM",""))*1000
        if "uM" in inter_string:
            micro_value = float(inter_string.replace("uM",""))
    except Exception as e:
        print("convertion issue:",e)
        return micro_value
    return micro_value

class PhaMPN(tf.keras.Model):
    # This MPNN is only here to do message passing and updating each atom in the ligand to
    # derive new atom features
    # Could be extended with a MPNN for the environment as well#
    # Input 
    # hidden_size: what is the size of the dense layers in between
    # depthG: the depth of the layer#
    
    def __init__(self, hidden_size, depth):
        super(PhaMPN, self).__init__(name="PhaMPN")
        self.hidden_size = hidden_size
        self.depth = depth
        self.W_i = tf.keras.layers.Dense(self.hidden_size,input_shape=(FEATURE_FDIM + EDGE_FDIM,),name="W_i_mpn",use_bias=False)
        self.W_h = tf.keras.layers.Dense(self.hidden_size, input_shape=(self.hidden_size,),name="W_h_mpn",use_bias=False)
        self.W_o = tf.keras.layers.Dense(self.hidden_size, input_shape=(FEATURE_FDIM + hidden_size,),name="W_o_mpn",
                    activation='sigmoid')

    def call(self, features,fedges,agraph,egraph,scope):
        features = create_var(features)
        fedges = create_var(fedges)
        agraph = create_var(agraph)
        egraph = create_var(egraph)

        binput = self.W_i(fedges)

        message = tf.keras.activations.sigmoid(binput)
        # print("message",message)

        for i in range(self.depth - 1):
            nei_message = index_select_ND(message, 0, egraph)
            # print("nei_message1",nei_message[:10])
            nei_message = tf.reduce_sum(nei_message,axis=1)
            # print("nei_message2",nei_message[:10])
            nei_message = self.W_h(nei_message)
            # print("nei_message3",nei_message[:10])
            message = tf.keras.activations.sigmoid(binput + nei_message)

        nei_message = index_select_ND(message, 0, agraph)
        # print("nei_message4",nei_message[:10])
        nei_message = tf.reduce_sum(nei_message,axis=1)
        # print("nei_message5",nei_message[:10])
        ainput = tf.concat([features, nei_message], axis=1)
        # print("ainput",ainput[:10])
        feature_hiddens = self.W_o(ainput)
        # print("feature_hiddens",feature_hiddens[:10])

        max_len = max([x for _,x in scope])
        batch_vecs = []
        for st,le in scope:
            cur_vecs = tf.reduce_mean(feature_hiddens[st : st + le],axis=0)
            batch_vecs.append( cur_vecs )

        mol_vecs = tf.stack(batch_vecs, axis=0)

        return mol_vecs 

    @staticmethod
    def tensorize(graph_batch):
        padding = np.zeros((FEATURE_FDIM + EDGE_FDIM))
        features,fedges = [],[padding] #Ensure bond is 1-indexed
        connected,all_edges = [],[(-1,-1)] #Ensure bond is 1-indexed
        scope = []
        affinities = []
        total_features = 0
        for pha_graph in graph_batch:
            print(pha_graph.activ)
            n_features = len(pha_graph.nodes)
            # affinities.append(getAffinity(pha_graph.affinity))#TODO
            affinities.append(pha_graph.activ)
            for feature in pha_graph.nodes:
                features.append( feature.other_feature_types )
                connected.append([])

            # for edge in pha_graph.edges:
            #     if pha_graph.distance_idx(edge[0],edge[1]) > 5.:continue
            #     connected.append([])
            
            # print("len conn",len(connected))
            # print("pha_graph.edges",len(pha_graph.edges))
            for edge in pha_graph.edges:
                f1 = edge[0]
                f2 = edge[1]
                # if pha_graph.distance_idx(f1,f2) > 5.:continue
                x = f1 + total_features
                y = f2 + total_features
                print("f1",f1,x)
                print("f2",f2,y)

                b = len(all_edges) 
                all_edges.append((x,y))
                fedges.append( tf.concat([features[x], [int(pha_graph.edge_weights[f1,f2])]], 0) )
                connected[y].append(b)

                b = len(all_edges)
                all_edges.append((y,x))
                fedges.append( tf.concat([features[y], [int(pha_graph.edge_weights[f1,f2])]], 0) )
                connected[x].append(b)
            
            scope.append((total_features,n_features))
            total_features += n_features
        total_edges = len(all_edges)
        features = tf.stack(features, 0)
        fedges = tf.stack(fedges, 0)
        agraph = np.zeros((total_features,MAX_NB),dtype=np.int32) # should be long int64
        egraph = np.zeros((total_edges,MAX_NB),dtype=np.int32) # should be long int64
        affinities = tf.stack(affinities, 0)
        for a in range(total_features):
            for i,b in enumerate(connected[a]):
                agraph[a,i] = b

        for b1 in range(1, total_edges):
            x,y = all_edges[b1]
            for i,b2 in enumerate(connected[x]):
                if all_edges[b2][0] != y:
                    egraph[b1,i] = b2
        return (features, fedges, tf.convert_to_tensor(agraph), tf.convert_to_tensor(egraph), scope,affinities)