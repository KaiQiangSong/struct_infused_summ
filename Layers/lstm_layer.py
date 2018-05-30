import theano
import theano.tensor as T
import numpy as np

from utility.utility import *

from dropout_layer import *

'''
Layer Name: LSTM_Layer
Math: 
    i = sigmoid( x * W_i + h * U_i + b_i)
    f = sigmoid( x * W_f + h * U_f + b_h)
    c = f * c_ + i_t * tanh( x * W_c + h * U_c + b_c)
    o = sigmoid( x * W_o + h * U_o + b_o)
    h = o * tanh(c)           
--------------------------------------------------------------
'''
def lstm_init(prefix, params, layer_setting):
    n_in = layer_setting['n_in']
    n_out = layer_setting['n_out']
    
    #Parameters Initialization 
    params[join(prefix, 'W')] =  np.concatenate([random_weights(n_in, n_out),
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
    return params

def lstm_calc(prefix, params, layer_setting, state_below, h_init = None, c_init = None, mask = None, training = True):
    
    n_steps = state_below.shape[0]
    n_dim = params[join(prefix,'U')].shape[0]
    
    if (state_below.ndim == 3):
        n_sample = state_below.shape[1]
    else:
        n_sample = 1
    
    if h_init == None:
        h_init = T.alloc(numpy_floatX(0.), n_sample, n_dim)
    if c_init == None:
        c_init = T.alloc(numpy_floatX(0.), n_sample, n_dim)
    if (mask == None):
        mask = T.alloc(numpy_floatX(1.), n_steps, n_sample)
    
    state_below = T.dot(state_below, params[join(prefix,'W')]) + params[join(prefix,'b')]
    
    def step(inputs, mask, h_previous, c_previous):
        
        activation = T.dot(h_previous,params[join(prefix,'U')])
        activation += inputs 
        
        activation_i = slice(activation, 0, n_dim)
        activation_f = slice(activation, 1, n_dim)
        activation_c = slice(activation, 2, n_dim)
        activation_o = slice(activation, 3, n_dim)
        
        i = sigmoid(activation_i)
        f = sigmoid(activation_f)
        o = sigmoid(activation_o)
        
        c = f * c_previous + i * tanh(activation_c)
        c = mask[:, None] * c + (1 - mask)[:, None] * c_previous 
        
        h = o * tanh(c)
        h = mask[:, None] * h + (1 - mask)[:, None] * h_previous
        
        return h, c
    
    rval, updates = theano.scan(fn = step,
                                sequences = [state_below, mask],
                                outputs_info = [h_init, c_init],
                                name = join(prefix,'layers'),
                                n_steps = n_steps)
    ret1, ret2 = rval[0], rval[1]
    if training:
        ret1 = dropout_calc(ret1, layer_setting['dropout'])
        ret2 = dropout_calc(ret2, layer_setting['dropout'])
    return ret1, ret2