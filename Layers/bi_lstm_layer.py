import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from lstm_layer import *

def bi_lstm_init(prefix, params, layer_setting):
    params = lstm_init(prefix+'_forward', params, layer_setting)
    params = lstm_init(prefix+'_backward', params, layer_setting)
    return params

def bi_lstm_calc(prefix, params, layer_setting, state_below, hf_init = None, cf_init = None, hb_init = None, cb_init = None, mask = None, training = True):
    [hf, cf] = lstm_calc(prefix+'_forward', params, layer_setting, state_below, hf_init, cf_init, mask, training = training)
    [hb, cb] = lstm_calc(prefix+'_backward', params, layer_setting, state_below[::-1,::], hb_init, cb_init, mask[::-1,:], training = training)
    state_below = T.concatenate([hf, hb[::-1,:,:]], axis = 2)
    memory_cell = T.concatenate([cf, cb[::-1,:,:]], axis = 2)
    return state_below, memory_cell
