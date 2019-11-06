import logging
log = logging.getLogger(__name__)
import tensorflow as tf
import numpy as np
import copy
from phagraphnn.utilities import getConnectedFeatures, updateConnected

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