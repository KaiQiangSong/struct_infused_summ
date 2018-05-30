import theano
import numpy as np
from mylog.mylog import mylog
from collections import OrderedDict
from Layers.Layers import *

def init_params(options, Vocab, I2Es, log):
    log.log('Start Initializing Parameters')
    
    params = OrderedDict()

    embedding_setting = options['embedding']
    encoder_setting = options['encoder']
    enc2dec_setting = options['enc2dec']
    decoder_setting = options['decoder']
    
    #Prepare Embedding Parameters
    if options["Structure_aviliable"]:
        params = get_layer('embedding')[0]('embedding', params, embedding_setting['vocab'], Vocab["i2e"])
        params = get_layer('multiple_embedding')[0]('struct_embedding', params, embedding_setting['struct'], I2Es)
        params = get_layer('position_embedding')[0]('position_embedding', params, embedding_setting['position'])
    else:
        params = get_layer('embedding')[0]('embedding',params, embedding_setting, Vocab["i2e"])
    
    # Prepare Encoder Parameters
    log.log('Encoder Parameters Initializing')
    
    if encoder_setting['type'] == 'uni_lstm':
        params = get_layer('uni_lstm')[0]('encoder', params, encoder_setting)
    elif encoder_setting['type'] == 'bi_lstm':
        params = get_layer('bi_lstm')[0]('encoder', params, encoder_setting)
    elif encoder_setting['type'] == 'stacked':
        for Index in range(0, encoder_setting['n_layer']):
            layerName = 'encoder_'+str(Index)
            settingThis = encoder_setting[layerName]
            if (settingThis['type'] == 'uni_lstm'):
                params = get_layer('uni_lstm')[0](layerName, params, settingThis)
            elif (settingThis['type'] == 'bi_lstm'):
                params = get_layer('bi_lstm')[0](layerName, params, settingThis)
            elif (settingThis['type'] == 'feedforward'):
                params = get_layer('feedforward')[0](layerName, params, settingThis)
    
    # Prepare Enc2Dec Parameters
    log.log('Enc2dec Parameters Initializing')
    
    if enc2dec_setting['method'] == 'mean':
        if enc2dec_setting['type'] == 'feedforward':
            params = get_layer('feedforward')[0]('enc2dec_h', params, enc2dec_setting)
            params = get_layer('feedforward')[0]('enc2dec_c', params, enc2dec_setting)
        elif enc2dec_setting['type'] == 'stacked':
            for Index in range(0, enc2dec_setting['n_layer']):
                settingThis = enc2dec_setting['enc2dec_'+str(Index)]
                params = get_layer(settingThis['type'])[0]('enc2dec_h_'+str(Index), params, settingThis)
                params = get_layer(settingThis['type'])[0]('enc2dec_c_'+str(Index), params, settingThis)
    elif enc2dec_setting['method'] == 'last_same':
        if enc2dec_setting['type'] == 'feedforward':
            params = get_layer('feedforward')[0]('enc2dec_h', params, enc2dec_setting)
        elif enc2dec_setting['type'] == 'stacked':
            for Index in range(0, enc2dec_setting['n_layer']):
                settingThis = enc2dec_setting['enc2dec_'+str(Index)]
                params = get_layer(settingThis['type'])[0]('enc2dec_h_'+str(Index), params, settingThis)
    elif enc2dec_setting['method'] == 'last_unique':
        if enc2dec_setting['type'] == 'feedforward':
            params = get_layer('feedforward')[0]('enc2dec_h', params, enc2dec_setting)
            params = get_layer('feedforward')[0]('enc2dec_c', params, enc2dec_setting)
        elif enc2dec_setting['type'] == 'stacked':
            for Index in range(0, enc2dec_setting['n_layer']):
                settingThis = enc2dec_setting['enc2dec_'+str(Index)]
                params = get_layer(settingThis['type'])[0]('enc2dec_h_'+str(Index), params, settingThis)
                params = get_layer(settingThis['type'])[0]('enc2dec_c_'+str(Index), params, settingThis)
    # Prepare Decoder Parameters
    log.log('Decoder Parameters Initializing')
    params = get_layer(decoder_setting['type'])[0]('decoder', params, decoder_setting)
    log.log('Stop Initializing Parameters')
    
    return params

def init_params_shared(params):
    mylog('Start Initializing Shared Parameters')
    params_shared = OrderedDict()
    for key,value in params.iteritems():
        params_shared[key] = theano.shared(params[key], name = key)
    mylog('Stop Initializing Shared Parameters')
    return params_shared

def load_params(path, params):
    pp = np.load(path)        
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params