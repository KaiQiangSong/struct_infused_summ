import codecs, json, string, re
import cPickle as Pickle

import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict

# Initializing
def zero_weights(n_in ,n_out = None):
    if (n_out == None):
        W = np.zeros(n_in)
    else:
        W = np.zeros(n_in, n_out)
    return W.astype('float32')

def orthogonal_weights(n_dim):
    W = np.random.randn(n_dim,n_dim)
    u, _ , _ =np.linalg.svd(W)
    return u.astype('float32')

def random_weights(n_in,n_out, scale = None):
    if scale is None:
        scale = np.sqrt(2.0 / (n_in + n_out))
    W = scale * np.random.randn(n_in,n_out)
    return W.astype('float32')

#Data Processing
def slice(x, n, dim):
    if (x.ndim == 3):
        return x[:,:,n * dim : (n+1) * dim]
    elif (x.ndim == 2):
        return x[:,n * dim : (n+1) * dim]
    return x[n * dim : (n+1) * dim]

def slice_neq(x, n, dims):
    if (x.ndim == 3):
        return x[:,:,dims[n] : dims[n+1]]
    elif (x.ndim == 2):
        return x[:,dims[n] : dims[n+1]]
    return x[dims[n] : dims[n+1]]

def join(pp, name):
    return '%s_%s' % (pp, name)

def numpy_floatX(data):
    return np.asarray(data, dtype='float32')

def itemlist(params):
    return [vv for kk,vv in params.iteritems()]

def zip_params(params, params_shared):
    for kk, vv in params.iteritems():
        params_shared[kk].set_value(vv)
        
def unzip_params(params_shared):
    params = OrderedDict()
    for kk, vv in params_shared.iteritems():
        params[kk] = vv.get_value()
    return params

def numpy_2D_softmax(x):
    e_x = np.exp(x - x.max(axis = 1))
    return e_x / e_x.sum(axis = 1)

# activation_Function
def tanh(x):
    return T.tanh(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def softmax(x):
    return T.nnet.softmax(x)

def relu(x):
    return T.nnet.relu(x)

def linear(x):
    return x

def softmax_mask(x, mask):
    x = softmax(x)
    x = x * mask
    x = x / x.sum(0,keepdims=True)
    return x

# IO
def loadFromJson(filename):
    f = codecs.open(filename,'r',encoding = 'utf-8')
    data = json.load(f,strict = False)
    f.close()
    return data

def saveToJson(filename, data):
    f = codecs.open(filename,'w',encoding = 'utf-8')
    json.dump(data, f, indent=4)
    f.close()
    return True

def saveToPKL(filename, data):
    with open(filename,'wb')as f:
        Pickle.dump(data, f)
    return 

def loadFromPKL(filename):
    with open(filename,'rb') as f:
        data = Pickle.load(f)
    return data

def writeFile(filename, massage):
    with codecs.open(filename,'w',encoding = 'utf-8') as f:
        f.write(massage)
    return True

def saveModel(params_shared, options, log, epoch_index, batch_index,  bestScore, batch_count, lRate, rate_count, method = 'best'):
    log.log('Start Saving Options')
    
    if method == 'best':
        strId = '_best'
    elif method == 'best_epoch':
        strId = '_best_epoch_'+str(epoch_index)
    elif method == 'best_epoch_batch':
        strId = '_best_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
    elif method == 'epoch':
        strId = '_epoch_'+str(epoch_index)
    elif method == 'check_epoch_batch':
        strId = '_check_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
    elif method == 'check2':
        strId = '_check2_epoch_'+str(epoch_index)+'_batch_'+str(batch_index)
    elif method == 'check2_best':
        strId = '_check2_best'
        
    options['start_epoch'] = epoch_index+1
    options['bestScore'] = bestScore
    options['batch_count'] = batch_count
    options['lRate'] = lRate
    options['rate_count'] = rate_count
    
    
    options['reload'] = True
    options['dataset_loading_method'] = 'load'
    options['reload_options'] = 'options'+strId+'.json'
    options['reload_model'] = 'model'+strId+'.npz'
    saveToJson(options['model_path']+'options'+strId+'.json', options)
    
    log.log('Start Saving Parameters')
    np.savez(options['model_path']+'model'+strId, **unzip_params(params_shared))
    
def get_nGram(l, n = 2):
    l = list(l)
    return set(zip(*[l[i:] for i in range(n)]))

def remove_digits(parse):
    return re.sub(r'\d', '#', parse)

def RougeTrick(parse):
    '''
    parse = re.sub(r'#','XXX',parse)
    parse = re.sub(r'XXX-','XXXYYY',parse)
    parse = re.sub(r'-XXX','YYYXXX',parse)
    parse = re.sub(r'XXX.','XXXWWW',parse)
    parse = re.sub(r'.XXX','WWWXXX',parse)
    parse = re.sub(r'<unk>','ZZZZZ',parse)
    '''
    parse = re.sub(r'#','T',parse)
    parse = re.sub(r'T-','TD',parse)
    parse = re.sub(r'-T','DT',parse)
    parse = re.sub(r'TX.','TB',parse)
    parse = re.sub(r'.T','BT',parse)
    parse = re.sub(r'<unk>','UNK',parse)
    
    return parse
    