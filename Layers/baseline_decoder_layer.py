import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from lstm_layer import *
from feedforward_layer import *

def baseline_decoder_init(prefix, params, layer_setting):
    params = lstm_init(prefix+'_lstm', params, layer_setting['_lstm'])
    params = feedforward_init(prefix+'_feedforward', params, layer_setting['_feedforward'])
    return params

def baseline_decoder_calc(prefix, params, layer_setting, state_below, h_init = None, c_init = None, mask = None, training = True):
    vocab_size = params[join(prefix+'_feedforward', 'W')].shape[1]
    n_dim = params[join(prefix+'_lstm', 'U')].shape[0]
    n_steps = state_below.shape[0]
    
    [h, c] = lstm_calc(prefix+'_lstm', params, layer_setting['_lstm'], state_below, h_init, c_init, mask, training = training)
    
    dist = feedforward_calc(prefix+'_feedforward', params, layer_setting['_feedforward'], h)
    return h, c, dist