import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from vocabulary.vocabulary import I2E

from dropout_layer import *
'''
Layer Name : Embedding_Layer

'''

def embedding_init(prefix, params, layer_setting, I2E):
    params[join(prefix,'Wemb')] = I2E
    return params
    
def embedding_calc(prefix, params, layer_setting, state_below, training = True):
    
    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]
    n_dim = layer_setting['dim']
    
    result = params[join(prefix,'Wemb')][state_below.flatten()].reshape([n_steps, n_samples, n_dim])
    
    if training:
        result = dropout_calc(result, layer_setting['dropout'])
    
    return result 