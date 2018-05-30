import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from lstm_layer import *
from feedforward_layer import *
from attention_layer import *

def switcher_pointer_3way_decoder_init(prefix, params, layer_setting):
    params = lstm_init(prefix+'_lstm', params, layer_setting['_lstm'])
    params = attention_init(prefix+'_att_1', params, layer_setting['_att_1'])
    params = attention_init(prefix+'_att_2', params, layer_setting['_att_2'])
    params = feedforward_init(prefix+'_tanh', params, layer_setting['_tanh'])
    params = feedforward_init(prefix+'_softmax', params, layer_setting['_softmax'])
    params = feedforward_init(prefix+'_switcher', params, layer_setting['_switcher'])
    return params

def switcher_pointer_3way_decoder_calc(prefix, params, layer_setting, h_e, s_e, mask_below, state_below, h_init = None, c_init = None, mask = None, training = True):
    [h_d, c_d] = lstm_calc(prefix+'_lstm', params, layer_setting['_lstm'], state_below, h_init, c_init, mask, training = training)
    alpha = attention_calc(prefix+'_att_1', params, layer_setting['_att_1'], h_d, h_e)
    beta = attention_calc(prefix+'_att_2', params, layer_setting['_att_2'], h_d, s_e)
    
    context_1 = T.batched_dot(alpha.dimshuffle(1,0,2), h_e.dimshuffle(1,0,2)).dimshuffle(1,0,2)
    context_2 = T.batched_dot(beta.dimshuffle(1,0,2), s_e.dimshuffle(1,0,2)).dimshuffle(1,0,2)
    
    h_d_2 = feedforward_calc(prefix+'_tanh', params, layer_setting['_tanh'], T.concatenate([h_d, context_1, context_2], axis = 2))
    dist = feedforward_calc(prefix+'_softmax', params, layer_setting['_softmax'], h_d_2)
    switcher = feedforward_calc(prefix+'_switcher', params, layer_setting['_switcher'], T.concatenate([h_d, context_1, context_2, state_below], axis = 2))
    return h_d, c_d, dist, alpha, beta, switcher