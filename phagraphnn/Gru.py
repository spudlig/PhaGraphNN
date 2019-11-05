
import tensorflow as tf
from collections import deque
from phagraphnn.utilities import indexSelect, getConnectedFeatures

class GraphGRU(tf.keras.Model):

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        #GRU Weights
        self.W_z = tf.keras.layers.Dense(self.hidden_size,input_shape=(input_size+self.hidden_size,),name="W_z_encoder",
           kernel_initializer=tf.initializers.he_normal(),bias_initializer=tf.initializers.he_normal())
        self.W_r = tf.keras.layers.Dense(self.hidden_size, input_shape=(input_size,),name="W_r_encoder",
           kernel_initializer=tf.initializers.he_normal(),bias_initializer=tf.initializers.he_normal(),
           use_bias=False)
        self.W_h = tf.keras.layers.Dense(self.hidden_size, input_shape=(input_size+self.hidden_size,),name="W_h_encoder",
           kernel_initializer=tf.initializers.he_normal(),bias_initializer=tf.initializers.he_normal())
        self.U_r = tf.keras.layers.Dense(self.hidden_size, input_shape=(self.hidden_size,),name="W_h_encoder",
           kernel_initializer=tf.initializers.he_normal(),bias_initializer=tf.initializers.he_normal())

    def call(self, h, x):

        for it in range(self.depth):
            h_nei = h
            # h_nei = getConnectedFeatures(h, mess_graph)
            # sum_h = tf.math.reduce_sum(h_nei,axis=1)
            z_input = tf.concat([x, h_nei], axis=1)
            z = tf.keras.activations.sigmoid(self.W_z(z_input))

            r_1 = tf.dtypes.cast(tf.reshape(self.W_r(x),shape=(-1,1,self.hidden_size)),dtype=tf.float32)
            r_2 = self.U_r(h_nei)
            r = tf.keras.activations.sigmoid(r_1 + r_2)
            
            gated_h = r * h_nei
            sum_gated_h = tf.reduce_sum(gated_h,axis=1)
            h_input = tf.concat([x, sum_gated_h], axis=1)
            pre_h = tf.keras.activations.tanh(self.W_h(h_input))
            h = (1.0 - z) * h_nei + z * pre_h

        return h