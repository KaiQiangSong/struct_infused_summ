import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from activation_layer import *

'''
(T1, 64, h_d) --- X (h_d, h_att) --> (T1, 64, h_att)
(T2, 64, h_e) --- X (h_e, h_att) --> (T2, 64, h_att)

(T1, 64, h_att) --- dimshuffle(1,2,0,'x')--> (64, h_att, T1, 1)
(64, h_att, T1, 1) --- X one_like(1 , T2) --> (64, h_att, T1, T2)
(T2, 64, h_att) --- dimshuffle(1,2,'x',0)--> (64, h_att, 1, T2)
(64, h_att, T1, T2) + (64, h_att, 1, T2) --> (64, h_att, T1, T2)
(64, h_att, T1, T2) --- dimshuffle(2,0,3,1) --> (T1, 64, T2, h_att)

(T1, 64, T2, h_att) --- X (h_att, 1) --> (T1, 64, T2, 1)
(T1, 64, T2, 1) --- dimshuffle(0,1,2,) --> (T1, 64, T2)
'''

def attention_init(prefix, params, layer_setting):
    n_e = layer_setting['n_e']
    n_d = layer_setting['n_d']
    n_att = layer_setting['n_att']
    params[join(prefix,'W_e2att')] = random_weights(n_e, n_att)
    params[join(prefix,'W_d2att')] = random_weights(n_d, n_att)
    params[join(prefix,'b_att_in')] = zero_weights(n_att)
    params[join(prefix,'V_att')] = random_weights(n_att, 1)
    params[join(prefix,'b_att_out')] = zero_weights(1)
    params = activation_init(prefix+'_softmax', params, layer_setting['_softmax'])
    return params

def attention_calc(prefix, params, layer_setting, h_d, h_e):
    n_e = layer_setting['n_e']
    n_d = layer_setting['n_d']
    n_att = layer_setting['n_att']
    h_d = T.dot(h_d, params[join(prefix,'W_d2att')]).dimshuffle(1,2,0,'x')
    h_e = T.dot(h_e, params[join(prefix,'W_e2att')]).dimshuffle(1,2,'x',0)
    #h_d = T.dot(h_d, T.ones([1, h_e.shape[2]], dtype = "float32"))
    activation = (h_d + h_e).dimshuffle(2,0,3,1) + params[join(prefix,'b_att_in')]
    activation = tanh(activation)
    activation = T.dot(activation, params[join(prefix,'V_att')]) + params[join(prefix,'b_att_out')]
    shp = activation.shape
    activation = T.reshape(activation, [shp[0], shp[1], shp[2]], ndim = 3)
    result = activation_calc(prefix+'_softmax', params, layer_setting['_softmax'], activation)
    return result