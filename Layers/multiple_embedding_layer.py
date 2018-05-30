import theano
import theano.tensor as T
import numpy as np

from utility.utility import *
from vocabulary.vocabulary import I2E

from embedding_layer import *
from dropout_layer import *
'''
Layer Name : Multiple_Embedding_Layer
'''

def multiple_embedding_init(prefix, params, layer_setting, I2Es):
    num = layer_setting["num"]
    for i in range(num):
        params = embedding_init(prefix+'_'+str(i), params, layer_setting["_"+str(i)], I2Es[i])
    return params

def multiple_embedding_calc(prefix, params, layer_setting, state_belows, training = True):
    emb_0 = embedding_calc(prefix+'_0', params, layer_setting["_0"], state_belows[:,:,0], training = training)
    emb_1 = embedding_calc(prefix+'_1', params, layer_setting["_1"], state_belows[:,:,1], training = training)
    emb_2 = embedding_calc(prefix+'_2', params, layer_setting["_2"], state_belows[:,:,2], training = training)
    emb_3 = embedding_calc(prefix+'_3', params, layer_setting["_3"], state_belows[:,:,3], training = training)
        
    return T.concatenate([emb_0, emb_1, emb_2, emb_3], axis = 2)
