import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from lstm_layer import *

def uni_lstm_init(prefix, params, layer_setting):
    return lstm_init(prefix+'_forward', params, layer_setting)

def uni_lstm_calc(prefix, params, layer_setting,state_below, h_init = None, c_init = None, mask = None, training = True):
    return lstm_calc(prefix+'_forward', params, layer_setting, state_below, h_init, c_init, mask, training = True)