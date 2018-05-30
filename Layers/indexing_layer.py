import theano
import theano.tensor as T
import numpy as np

from utility.utility import *

'''
Layer Name: indexing Layer
    state_below is what you want copy, a 3D tensor of Value
    indexes is where you need it, a 2D tensor of Index
'''

def indexing_init(prefix, params, layer_settings):
    return params

def indexing_calc(prefix, params, state_below, indexes):
    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]
    n_dim = state_below.shape[2]
    
    m_steps = indexes.shape[0]
    
    zero = T.zeros_like(state_below[:1], dtype = 'float32')
    table = T.concatenate([zero, state_below], axis = 0)
    table = T.reshape(table, [(n_steps+1) * n_samples, n_dim], ndim = 2)
    
    bias = T.tile(T.arange(n_samples, dtype = 'int64'), m_steps) * (n_steps + 1)
    
    result = table[indexes.flatten() + bias + 1]
    result = T.reshape(result, [m_steps, n_samples, n_dim], ndim = 3)
    return result
    