import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from lstm_layer import *
from feedforward_layer import *
from attention_layer import *

'''
    LayerName : attention_decoder_layer
    Math:
    
    h_d_t = LSTM(h_d_(t-1), y_(t-1))
    
    e_t_i = f_att(h_d_t, h_e_i)
    alpha_t = softmax(e_t_i)
    context_t = alpha_t * e_t
    
    h_d2_t = tanh(W[h_d_t, context_t])
    Or h_d2_t = LSTM(h_d2_(t-1), [h_d_t, context_t])
    
    dist = softmax(W * h_d2_t)
    
    Parameters:
        n_emb: dimension of y_(t-1)
        n_att: dimension of attention layer
        n_h1: dimension of h_d_t
        n_h2: dimension of h_d2_t
        n_he: dimension of h_e_i
'''

def attention_decoder_init(prefix, params, layer_setting):
    params = lstm_init(prefix+'_lstm', params, layer_setting['_lstm'])
    params = attention_init(prefix+'_attention', params, layer_setting['_attention'])
    params = feedforward_init(prefix+'_tanh', params, layer_setting['_tanh'])
    params = feedforward_init(prefix+'_softmax', params, layer_setting['_softmax'])
    return params

def attention_decoder_calc(prefix, params, layer_setting, h_e, mask_below, state_below, h_init = None, c_init = None, mask = None, training = True):
    [h_d, c_d] = lstm_calc(prefix+'_lstm', params, layer_setting['_lstm'], state_below, h_init, c_init, mask, training = training)
    alpha = attention_calc(prefix+'_attention', params, layer_setting['_attention'], h_d, h_e)
    context = T.batched_dot(alpha.dimshuffle(1,0,2), h_e.dimshuffle(1,0,2)).dimshuffle(1,0,2)
    h_d2 = feedforward_calc(prefix+'_tanh', params, layer_setting['_tanh'], T.concatenate([h_d, context], axis = 2))
    dist = feedforward_calc(prefix+'_softmax', params, layer_setting['_softmax'], h_d2)
    return h_d, c_d, dist, alpha
