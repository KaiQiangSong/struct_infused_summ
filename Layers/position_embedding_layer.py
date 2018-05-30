import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from vocabulary.vocabulary import I2E

from dropout_layer import *
'''
Layer Name: Absolutely Position Embedding
init_method:
    random: random initialize the matrix of embedding
    clock: use clock embedding to initialize the embedding matrix
'''

def absolutely_position_embedding_init(prefix, params, layer_setting):
    # n_in should equal to the maximum position of inputs
    n_in = layer_setting['n_in']
    n_out = layer_setting['n_out']

    
    if layer_setting['init_method'] == 'random':
        params[join(prefix,'_abs_emb')] = random_weights(n_in, n_out)
    elif layer_setting['init_method'] == 'clock':
        threshold = layer_setting['threshold']
        posi = np.asarray(range(0, n_in), dtype = np.float32)
        scale = 2 * np.asarray(range(0, n_out/2) , dtype = np.float32) / n_out
        scale = np.power(threshold, scale)
        value = posi[:,None] * scale[None,:]
        emb = np.concatenate([np.sin(value), np.cos(value)], axis = 1)
        params[join(prefix,'_abs_emb')] = emb
    
    return params

def absolutely_position_embedding_calc(prefix, params, layer_setting, state_below):
    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]
    n_dim = layer_setting['n_out']
    
    result = params[join(prefix,'_abs_emb')][state_below.flatten()].reshape([n_steps, n_samples, n_dim])
    return result

'''
Layer Name: Relative Position Embedding
Method:
    embedding : trainable
        init_method: random, clock
    function: non-trainable
        mehtod: clock
'''

def relative_position_embedding_init(prefix, params, layer_setting):
    if layer_setting['calc_method'] == 'embedding':
        n_in = layer_setting['n_in']
        n_out = layer_setting['n_out']
        if layer_setting['init_method'] == 'random':
            params[join(prefix,'_rel_emb')] = random_weights(n_in, n_out)
        elif layer_setting['init_method'] == 'clock':
            threshold = layer_setting['threshold']
            posi = (0.5 + np.asarray(range(0, n_in), dtype = np.float32)) / n_in
            scale = 2 * np.asarray(range(0, n_out/2) , dtype = np.float32) / n_out
            scale = np.power(threshold, scale)
            value = posi[:,None] * scale[None,:]
            emb = np.concatenate([np.sin(value), np.cos(value)], axis = 1)
            params[join(prefix,'_rel_emb')] = emb
    return params
        
def relative_position_embedding_calc(prefix, params, layer_setting, state_below, mask_below):
    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1]
    n_dim = layer_setting['n_out']
    n_in = layer_setting['n_in']

    if layer_setting['calc_method'] == 'embedding':
        len_below = T.cast(T.sum(mask_below.dimshuffle(1,0), axis = 1), dtype = 'int64')
        state_below = T.minimum(n_in-1, state_below * n_in / len_below[None,:])
        result = params[join(prefix,'_rel_emb')][state_below.flatten()].reshape([n_steps, n_samples, n_dim])
    else:
        threshold = theano.shared(layer_setting['threshold'], dtype = 'float32')
        scale = 2 * T.arange(n_dim/2 , dtype = 'float32') / n_dim
        scale = T.power(threshold, scale)
        
        len_below = T.sum(mask_below.dimshuffle(1,0), axis = 1)
        posi = (state_below / len_below[None,:]).flatten()
        value = posi[:,None] * scale[None,:]
        result = T.concatenate([T.sin(value), T.cos(value)], axis = 1)
        result = T.reshape(result, [n_stesp, n_samples, n_dim])
    return result
        
'''
Layer Name: Position_Embedding_Layer
Type:
    abs: absolutely position embedding
    rel: relative position embedding
    both: abs and rel
'''

def position_embedding_init(prefix, params, layer_setting):
    if layer_setting['type'] == 'abs':
        params = absolutely_position_embedding_init(prefix+'_abs', params, layer_setting['_abs'])
    elif layer_setting['type'] == 'rel':
        params = relative_position_embedding_init(prefix+'_rel', params, layer_setting['_rel'])
    elif layer_setting['type'] == 'both':
        params = absolutely_position_embedding_init(prefix+'_abs', params, layer_setting['_abs'])
        params = relative_position_embedding_init(prefix+'_rel', params, layer_setting['_rel'])
        
    return params

def position_embedding_calc(prefix, params, layer_setting, mask_below, training = True):
    n_steps = mask_below.shape[0]
    n_samples = mask_below.shape[1]
    state_below = T.tile(T.arange(n_steps, dtype = 'int64'), (n_samples,1)).dimshuffle(1,0)
    if layer_setting['type'] == 'abs':
        emb = absolutely_position_embedding_calc(prefix+'_abs', params, layer_setting['_abs'], state_below)
    elif layer_setting['type'] == 'rel':
        emb = relative_position_embedding_calc(prefix+'_rel', params, layer_setting['_rel'], state_below, mask_below)
    elif layer_setting['type'] == 'both':
        emb_abs = absolutely_position_embedding_calc(prefix+'_abs', params, layer_setting['_abs'], state_below)
        emb_rel = relative_position_embedding_calc(prefix+'_rel', params, layer_setting['_rel'], state_below, mask_below)
        emb = T.concatenate([emb_abs, emb_rel], axis = 2)
    
    if training:
        emb = dropout_calc(emb, layer_setting['dropout'])
    return emb