import theano
import theano.tensor as T
import numpy as np

from utility.utility import *

'''
Layer Name: Feedforword Layer
Math:
    (D1, D2, D3) X (D3 , D4) -> (D1, D2, D4)
    activation on D4 
'''

def feedforward_init(prefix, params, layer_setting):
    params[join(prefix,'W')] = random_weights(layer_setting['n_in'], layer_setting['n_out'])
    params[join(prefix,'b')] = zero_weights(layer_setting['n_out'])
    return params

def feedforward_calc(prefix, params, layer_setting, state_below ):
    # one sample with no time series
    shp = state_below.shape
    in_dim = layer_setting['n_in']
    out_dim = layer_setting['n_out']
    activation = eval(layer_setting['activation'])
    
    num = shp[0] * shp[1]
    
    result = T.reshape(state_below, [num, in_dim], ndim = 2)
    result = T.dot(result, params[join(prefix,'W')])
    result += params[join(prefix,'b')]
    result = activation(result)
    result = T.reshape(result, [shp[0], shp[1], out_dim], ndim = 3)
    return result
