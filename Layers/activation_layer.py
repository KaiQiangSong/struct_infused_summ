import theano
import theano.tensor as T
import numpy as np

from utility.utility import *

def activation_init(prefix, params, layer_setting):
    #do Nothing
    return params

def activation_calc(prefix, params, layer_setting, state_below, mask_below = None):
    actFunc = eval(layer_setting['activation'])
    flag = False
    shp = state_below.shape
    if state_below.ndim == 3:
        flag = True
        shp0 = shp[0]
        shp1 = shp[1]
        shp2 = shp[2]
        state_below = T.reshape(state_below, [shp0 * shp1, shp2], ndim = 2)
    if mask_below == None:
        result = actFunc(state_below)
    else:
        result = actFunc(state_below, mask_below)
    if flag:
        result = T.reshape(result, [shp0, shp1, shp2], ndim = 3)
    return result
