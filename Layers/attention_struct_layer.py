import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from activation_layer import *

def attention_struct_init(prefix, params, layer_setting):
    n_in = layer_setting['n_in']
    n_out = layer_setting['n_out']
    n_att = layer_setting['n_att']
    
    params[join(prefix,'W')] = np.concatenate([random_weights(n_in, n_out),
                                               random_weights(n_in, n_out),
                                               random_weights(n_in, n_out),
                                               random_weights(n_in, n_out)],
                                              axis = 1)
    
    params[join(prefix,'U')] = np.concatenate([orthogonal_weights(n_out),
                                               orthogonal_weights(n_out),
                                               orthogonal_weights(n_out),
                                               orthogonal_weights(n_out)],
                                              axis = 1)
    params[join(prefix,'b')] = zero_weights(n_out * 4)
    
    params[join(prefix,'W_s')] = random_weights(n_in, n_att)
    params[join(prefix,'W_h')] = random_weights(n_out, n_att)
    params[join(prefix,'b_f1')] = zero_weights(n_att)
    
    params[join(prefix,'W_f2')] = random_weights(n_att, 1)
    params[join(prefix,'b_f2')] = zero_weights(1)
    return params

def attention_struct_calc(prefix, params, reference, inputMask, n_steps, h_init = None, c_init = None, s_init = None, mask = None):
    n_in = params[join(prefix,'W')].shape[0]
    n_out = params[join(prefix,'U')].shape[0]
    n_att = params[join(prefix,'W_f2')].shape[0]
    
    if (reference.ndim == 3):
        n_sample = reference.shape[1]
    else:
        n_sample = 1
    
    if h_init == None:
        h_init = T.alloc(numpy_floatX(0.), n_sample, n_out)
    
    if c_init == None:
        c_init = T.alloc(numpy_floatX(0.), n_sample, n_out)
    
    if s_init == None:
        s_init = T.alloc(numpy_floatX(0.), n_sample, n_in)
        
    if mask == None:
        mask = T.alloc(numpy_floatX(1.), n_steps, n_sample)
    
    alpha_init = T.alloc(numpy_floatX(0.), n_sample, reference.shape[0])
    
    state_below = T.dot(reference, params[join(prefix,'W_s')]) + params[join(prefix,'b_f1')]
    
    def step(mask, alpha_pre, s_pre, h_pre, c_pre):
        score = T.dot(h_pre, params[join(prefix,'W_h')])
        score = state_below + score[None,:,:]
        score = T.dot(T.tanh(score), params[join(prefix,'W_f2')]) + params[join(prefix,'b_f2')]
        shp = score.shape
        alpha = softmax_mask(T.reshape(score,[shp[1], shp[0]], ndim=2), inputMask.dimshuffle(1,0))
        context = T.batched_dot(alpha.dimshuffle(0,'x',1), reference.dimshuffle(1,0,2)).dimshuffle(0,2,)
        
        activation = T.dot(h_pre, params[join(prefix,'U')])
        activation += T.dot(context, params[join(prefix,'W')]) + params[join(prefix,'b')]
        
        activation_i = slice(activation, 0, n_out)
        activation_f = slice(activation, 1, n_out)
        activation_c = slice(activation, 2, n_out)
        activation_o = slice(activation, 3, n_out)
        
        i = sigmoid(activation_i)
        f = sigmoid(activation_f)
        o = sigmoid(activation_o)
        
        c = f * c_pre + i * tanh(activation_c)
        c = mask[:, None] * c + (1 - mask)[:, None] * c_pre
        
        h = o * tanh(c)
        h = mask[:, None] * h + (1 - mask)[:, None] * h_pre
        
        return alpha, context, h, c
        
    rval, updates = theano.scan(fn = step,
                                sequences = [mask],
                                outputs_info = [alpha_init, s_init, h_init, c_init],
                                name = join(prefix,'layer'),
                                n_steps = n_steps)
    
    return rval