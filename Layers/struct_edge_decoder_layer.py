import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from lstm_layer import *
from feedforward_layer import *
from attention_layer import *
from indexing_layer import *

'''
    LayerName : attention_stanford_decoder_layer
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

def struct_edge_decoder_init(prefix, params, layer_setting):
    params = lstm_init(prefix+'_lstm', params, layer_setting['_lstm'])
    params = attention_init(prefix+'_att_1', params, layer_setting['_att_1'])
    params = attention_init(prefix+'_att_2', params, layer_setting['_att_2'])
    params = indexing_init(prefix+'parent', params, layer_setting)
    params[join(prefix,'epsilon')] = numpy_floatX(0.5)
    params = feedforward_init(prefix+'_tanh', params, layer_setting['_tanh'])
    params = feedforward_init(prefix+'_softmax', params, layer_setting['_softmax'])
    params = feedforward_init(prefix+'_switcher', params, layer_setting['_switcher'])
    return params

def struct_edge_decoder_calc(prefix, params, layer_setting, h_e, s_e, parent, mask_below, state_below, h_init = None, c_init = None, mask = None, training = True):
    [h_d, c_d] = lstm_calc(prefix+'_lstm', params, layer_setting['_lstm'], state_below, h_init, c_init, mask, training = training)
    
    alpha = attention_calc(prefix+'_att_1', params, layer_setting['_att_1'], h_d, h_e)
    beta = attention_calc(prefix+'_att_2', params, layer_setting['_att_2'], h_d, s_e)
    
    
    #alpha : T_D, N, T_E
    #beta : T_D, N, T_E
    
    alpha2 = T.concatenate([T.zeros_like(alpha[:1], dtype = 'float32'),T.cumsum(alpha, axis = 0)], axis = 0)[:-1]
    
    #alpha2 : T_D, N, T_E
    position = T.cumsum(T.ones_like(mask_below, dtype = 'int64') * mask_below, axis = 0) - 1
    #position : T_E, N
    
    cond_1 = (position < parent)
    #cond_1 = (T.neq(position, -T.ones_like(position)))
    #cond_1: T_E, N
    cond_2 = (parent < position) & (T.neq(position, -T.ones_like(position)))
    #cond_2 = (T.neq(position, -T.ones_like(position)))
    #cond_2: T_E, N
    
    cond_1_ = T.tile(cond_1, (alpha.shape[0], 1, 1)).dimshuffle(0,2,1)
    #cond_1_ : T_D, N, T_E
    cond_2_ = T.tile(cond_2, (alpha.shape[0], 1, 1)).dimshuffle(0,2,1)
    #cond_2_ : T_D, N, T_E
    
    alpha2_p = indexing_calc(prefix+'parent', params, alpha2.dimshuffle(2,1,0), parent).dimshuffle(2,1,0)
    #alpha2_p : T_D, N, T_E
    
    item_1 = beta * alpha2
    #item_1 : T_D, N, T_E
    item_2 = beta * alpha2_p
    #item_2 : T_D, N, T_E
    
    target = T.where(cond_1, parent, -T.ones_like(parent, dtype = 'int64'))
    #target: T_E, N
    
    target = T.tile(target, (alpha.shape[0], 1, 1)).dimshuffle(0,2,1)
    #target: T_D, N, T_E
    
    
    gamma = T.where(cond_2_, item_2, T.zeros_like(item_2, dtype = 'float32'))
    #gamma: T_D, N, T_E
    
    target_plus = T.concatenate([T.zeros_like(target[:,:,:1], dtype = 'int64'), target+1], axis = 2)
    #target_plus: T_D, N, T_E + 1
    item_1_plus = T.concatenate([T.zeros_like(item_1[:,:,:1], dtype = 'float32'), item_1], axis = 2)
    #target_plus: T_D, N, T_E + 1
    
    target_flat = target_plus.flatten()
    #target_plus: T_D * N * (T_E + 1)
    item_1_flat = item_1_plus.flatten()
    #item_1_flat: T_D * N * (T_E + 1)
    
    t_d, n_sample, t_e = item_1_plus.shape
    
    M = T.zeros((t_d * n_sample * t_e, t_e), dtype = 'float32')
    d_flat = T.arange(t_d * n_sample * t_e, dtype = 'int64')
    
    #M[d_flat,target_flat = item_1_flat
    M = T.set_subtensor(M[(d_flat, target_flat)], item_1_flat)
    
    M = T.reshape(M, (t_d, n_sample, t_e, t_e), ndim = 4)
    #M = M.dimshuffle(0,3,1,2)
    
    gamma += T.sum(M, axis = 2)[:,:,1:]
    
    delta = alpha + params[join(prefix,'epsilon')] * gamma
    delta = delta / T.sum(delta, axis = 2, keepdims=True)
    
    context = T.batched_dot(delta.dimshuffle(1,0,2), h_e.dimshuffle(1,0,2)).dimshuffle(1,0,2)
    
    h_d_2 = feedforward_calc(prefix+'_tanh', params, layer_setting['_tanh'], T.concatenate([h_d, context], axis = 2))
    dist = feedforward_calc(prefix+'_softmax', params, layer_setting['_softmax'], h_d_2)
    switcher = feedforward_calc(prefix+'_switcher', params, layer_setting['_switcher'], T.concatenate([h_d, context, state_below], axis = 2))
    return h_d, c_d, dist, alpha, beta, gamma, delta, switcher
