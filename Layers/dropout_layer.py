import theano
import theano.tensor as T
import numpy as np

from utility.utility import *

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed = 19940609)

def dropout_init(prefix, params, layer_setting):
    
    return params

def dropout_calc(x, rate = 0.0):
    mask = srng.binomial(
        n = 1,
        p = (1- rate),
        size = x.shape,
        dtype = theano.config.floatX
        )
    
    return x * mask / (1 - rate)