from mylog.mylog import mylog
from utility.utility import *
from Layers.Layers import *
import theano.tensor as T

def build_model(params_shared, options, log):
    '''
    inputs order:
        input
        inputMask
        batch_vocab (optional)
        pointer (optional)
        feature (optional)
        padded_output
        outputMask
    '''
    log.log('Start Building Model')
    
    # settings of layers
    modelName = options['model_name']
    embedding_setting = options['embedding']
    encoder_setting = options['encoder']
    enc2dec_setting = options['enc2dec']
    decoder_setting = options['decoder']
    
    inps = []
    inps_enc = []
    outps_add = []
    
    input = T.matrix('input', dtype = 'int64')
    inputMask = T.matrix('inputMask', dtype = 'float32')
    
    inps += [input, inputMask]
    inps_enc += [input, inputMask]
    
    if options["LVT_aviliable"]:
        batch_vocab = T.vector('batch_vocab', dtype = 'int64')
        pointer = T.matrix('pointer', dtype = 'int64')
        inps += [batch_vocab, pointer]
    else:
        inps += [None, None]
        
    updates = []
    
    # Embedding Layer
    if options["Structure_aviliable"]:
        vocab_setting = embedding_setting['vocab']
        input_embedding = get_layer('embedding')[1]('embedding', params_shared, vocab_setting, input, training = options['training'])
        
        feature = T.tensor3('feature', dtype = 'int64')
        struct_setting = embedding_setting['struct']
        struct_embedding = get_layer('multiple_embedding')[1]('struct_embedding', params_shared, struct_setting, feature, training = options['training'])
        
        position_setting = embedding_setting['position']
        position_embedding = get_layer('position_embedding')[1]('position_embedding', params_shared, position_setting, inputMask, training = options['training'])
        
        struct_below = T.concatenate([struct_embedding, position_embedding], axis = 2)
        
        inps += [feature]
        inps_enc += [feature]
        outps_add += [input_embedding, struct_below]
        
        if options['Parent_aviliable']:
            parent = T.matrix('parent', dtype = 'int64')
            inps += [parent]
        else:
            inps += [None]
    else:
        input_embedding = get_layer('embedding')[1]('embedding', params_shared, embedding_setting, input, training = options['training'])
        
        inps += [None, None]
        
        
    if modelName == 'struct_one_v1':
        state_below = T.concatenate([input_embedding, struct_below], axis = 2)
    else:
        state_below = input_embedding
    
    # Encoder Layer
    if encoder_setting['type'] == 'uni_lstm':
        state_below, memory_cell = get_layer('uni_lstm')[1]('encoder', params_shared, encoder_setting, state_below, mask = inputMask, training = options['training'])
    elif encoder_setting['type'] == 'bi_lstm':
        state_below, memory_cell = get_layer('bi_lstm')[1]('encoder', params_shared, encoder_setting, state_below, mask = inputMask, training = options['training'])
    elif encoder_setting['type'] == 'stacked':
        states_below, memory_cells = [], []
        for Index in range(0, encoder_setting['n_layer']):
            layerName = 'encoder_' + str(Index)
            settingThis = encoder_setting[layerName]
            if settingThis['type'] == 'uni_lstm':
                state_below, memory_cell = get_layer('uni_lstm')[1](layerName, params_shared, settingThis, state_below, mask = inputMask, training = options['training'])
            elif settingThis['type'] == 'bi_lstm':
                state_below, memory_cell = get_layer('bi_lstm')[1](layerName, params_shared, settingThis, state_below, mask = inputMask, training = options['training'])
            elif settingThis['type'] == 'feedforward':
                state_below, memory_cell = get_layer('feedforward')[1](layerName, params_shared, settingThis, state_below)
            #states_below.append(state_below)
            #memory_cells.append(memory_cell)
        #state_below = T.concatenate(states_below, axis = 2)
        #memory_cell = T.concatenate(memory_cells, axis = 2)
    
    
    # Intermediate Layer  
    if enc2dec_setting['method'] == 'mean':
        h_init = T.mean(state_below, axis = 0)
        c_init = T.mean(memory_cell, axis = 0)
        if enc2dec_setting['type'] == 'feedforward':
            h_init = get_layer('feedforward')[1]('enc2dec_h', params_shared, enc2dec_setting, h_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
            c_init = get_layer('feedforward')[1]('enc2dec_c', params_shared, enc2dec_setting, c_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
        elif enc2dec_setting['type'] == 'mlp':
            for Index in range(0, enc2dec_setting['n_layer']):
                settingThis = enc2dec_setting['enc2dec_'+str(Index)]
                h_init = get_layer(settingThis['type'])[1]('enc2dec_h_'+str(Index),params_shared, settingThis, h_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
                c_init = get_layer(settingThis['type'])[1]('enc2dec_c_'+str(Index),params_shared, settingThis, c_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
    
    elif enc2dec_setting['method'] == 'last_same':
        if encoder_setting['type'] == 'uni_lstm':
            h_init = state_below[-1]
        elif encoder_setting['type'] == 'bi_lstm':
            shp = state_below.shape
            h_init = T.concatenate([state_below[0,:,:shp[2]/2], state_below[-1,:,shp[2]/2:]], axis = 1)
        elif encoder_setting['type'] == 'stacked':
            Index = encoder_setting['n_layer'] - 1
            layerName = 'encoder_' + str(Index)
            settingThis = encoder_setting[layerName]
            if settingThis['type'] == 'uni_lstm':
                h_init = state_below[-1]
            elif settingThis['type'] == 'bi_lstm':
                shp = state_below.shape
                h_init = T.concatenate([state_below[0,:,:shp[2]/2], state_below[-1,:,shp[2]/2:]], axis = 1)
        if enc2dec_setting['type'] == 'feedforward':
            h_init = get_layer('feedforward')[1]('enc2dec_h', params_shared, enc2dec_setting, h_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
        elif enc2dec_setting['type'] == 'mlp':
            for Index in range(0, enc2dec_setting['n_layer']):
                settingThis = enc2dec_setting['enc2dec_'+str(Index)]
                h_init = get_layer(settingThis['type'])[1]('enc2dec_h_'+str(Index),params_shared, settingThis, h_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
        c_init = h_init
        
    elif enc2dec_setting['method'] == 'last_unique':
        if encoder_setting['type'] == 'uni_lstm':
            h_init = state_below[-1]
            c_init = memory_cell[-1]
        elif encoder_setting['type'] == 'bi_lstm':
            shp = state_below.shape
            h_init = T.concatenate([state_below[0,:,:shp[2]/2], state_below[-1,:,shp[2]/2:]], axis = 1)
            shp = memory_cell.shape
            c_init = T.concatenate([memory_cell[0,:,:shp[2]/2], memory_cell[-1,:,shp[2]/2:]], axis = 1)
        elif encoder_setting['type'] == 'stacked':
            Index = encoder_setting['n_layer'] - 1
            layerName = 'encoder_' + str(Index)
            settingThis = encoder_setting[layerName]
            if settingThis['type'] == 'uni_lstm':
                h_init = state_below[-1]
                c_init = memory_cell[-1]
            elif settingThis['type'] == 'bi_lstm':
                shp = state_below.shape
                h_init = T.concatenate([state_below[0,:,:shp[2]/2], state_below[-1,:,shp[2]/2:]], axis = 1)
                shp = memory_cell.shape
                c_init = T.concatenate([memory_cell[0,:,:shp[2]/2], memory_cell[-1,:,shp[2]/2:]], axis = 1)
        if enc2dec_setting['type'] == 'feedforward':
            h_init = get_layer('feedforward')[1]('enc2dec_h', params_shared, enc2dec_setting, h_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
            c_init = get_layer('feedforward')[1]('enc2dec_c', params_shared, enc2dec_setting, c_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
        elif enc2dec_setting['type'] == 'mlp':
            for Index in range(0, enc2dec_setting['n_layer']):
                settingThis = enc2dec_setting['enc2dec_'+str(Index)]
                h_init = get_layer(settingThis['type'])[1]('enc2dec_h_'+str(Index),params_shared, settingThis, h_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
                c_init = get_layer(settingThis['type'])[1]('enc2dec_c_'+str(Index),params_shared, settingThis, c_init.dimshuffle('x',0,1)).dimshuffle(1,2,)
        
    
    encoder = theano.function(inputs = inps_enc,
                              outputs = [state_below, memory_cell, h_init, c_init] + outps_add,
                              on_unused_input='ignore')
    
    # Decoder Layer
    padded_output = T.matrix('padded_output', dtype = 'int64')
    goldStd = padded_output[:-1,:]
    if options["Structure_aviliable"]:
        goldStd_embedding = get_layer('embedding')[1]('embedding', params_shared, embedding_setting['vocab'], goldStd, training = options['training'])
    else:
        goldStd_embedding = get_layer('embedding')[1]('embedding', params_shared, embedding_setting, goldStd, training = options['training'])
        
    inps += [padded_output]
    
    if modelName == 'baseline':
        h_next, c_next, vocab_ = get_layer('baseline_decoder')[1]('decoder', params_shared, decoder_setting, goldStd_embedding, h_init, c_init, training = options['training'])
    elif modelName == 'attention':
        h_next, c_next, vocab_, posi_ = get_layer('attention_decoder')[1]('decoder', params_shared, decoder_setting, state_below, inputMask, goldStd_embedding,  h_init, c_init, training = options['training'])
    elif modelName == 'switcher_pointer':
        h_next, c_next, vocab_, posi_, switcher_ = get_layer('switcher_pointer_decoder')[1]('decoder', params_shared, decoder_setting, state_below, inputMask, goldStd_embedding,  h_init, c_init, training = options['training'])
    elif modelName == 'struct_one_v1':
        h_next, c_next, vocab_, posi_, switcher_ = get_layer('switcher_pointer_decoder')[1]('decoder', params_shared, decoder_setting, state_below, inputMask, goldStd_embedding,  h_init, c_init, training = options['training'])
    elif modelName == 'struct_one_v2':
        one_info = T.concatenate([state_below, struct_below], axis = 2)
        h_next, c_next, vocab_, posi_, switcher_ = get_layer('switcher_pointer_decoder')[1]('decoder', params_shared, decoder_setting, one_info, inputMask, goldStd_embedding, h_init, c_init, training = options['training'])
    elif modelName == 'struct_node':
        struct_info = T.concatenate([input_embedding, struct_below], axis = 2)
        h_next, c_next, vocab_, alpha_, beta_, posi_, switcher_ = get_layer('struct_node_decoder')[1]('decoder', params_shared, decoder_setting, state_below, struct_info, inputMask, goldStd_embedding, h_init, c_init, training = options['training'])
    elif modelName == 'struct_edge':
        struct_info = T.concatenate([input_embedding, struct_below], axis = 2)
        parent_info = indexing_calc('parent', params_shared, struct_info, parent)
        struct_info = T.concatenate([struct_info, parent_info], axis = 2)
        h_next, c_next, vocab_, alpha_, beta_, gamma_, posi_, switcher_ = get_layer('struct_edge_decoder')[1]('decoder', params_shared, decoder_setting, state_below, struct_info, parent, inputMask, goldStd_embedding, h_init, c_init, training = options['training'])

    #Loss Function
    select_std = padded_output[1:,:]
    outputMask = T.matrix('outputMask', dtype = 'float32')
    inps += [outputMask]
    
    outputMask_flat = outputMask[1:,:].flatten()
    select_std_flat = select_std.flatten()
    
    if options["LVT_aviliable"]:
        if decoder_setting['_switcher']['n_out'] < 2:
            prob_vocab = switcher_[:,:,None] * T.concatenate([vocab_, T.zeros((vocab_.shape[0], vocab_.shape[1], batch_vocab.shape[0] - vocab_.shape[2]), dtype = 'float32')], axis = 2)
            shp = pointer.shape
            pointer = T.reshape(T.extra_ops.to_one_hot(pointer.flatten(), batch_vocab.shape[0]),[shp[0], shp[1], batch_vocab.shape[0]], ndim = 3)
            prob_posi = (1 - switcher_[:,:,None]) * T.batched_dot(posi_.dimshuffle(1,0,2), pointer.dimshuffle(1,0,2)).dimshuffle(1,0,2)
            prob = prob_vocab + prob_posi
            
        elif decoder_setting['_switcher']['n_out'] ==  3:
            switcher_vocab = switcher_[:,:,0].dimshuffle(0,1,)
            switcher_alpha = switcher_[:,:,1].dimshuffle(0,1,)
            switcher_beta = switcher_[:,:,2].dimshuffle(0,1,)
            prob_vocab = switcher_vocab[:,:,None] * T.concatenate([vocab_, T.zeros((vocab_.shape[0], vocab_.shape[1], batch_vocab.shape[0] - vocab_.shape[2]), dtype = 'float32')], axis = 2)
            shp = pointer.shape
            pointer = T.reshape(T.extra_ops.to_one_hot(pointer.flatten(), batch_vocab.shape[0]),[shp[0], shp[1], batch_vocab.shape[0]], ndim = 3)
            prob_alpha = switcher_alpha[:,:,None] * T.batched_dot(alpha_.dimshuffle(1,0,2), pointer.dimshuffle(1,0,2)).dimshuffle(1,0,2)
            prob_beta = switcher_beta[:,:,None] * T.batched_dot(beta_.dimshuffle(1,0,2), pointer.dimshuffle(1,0,2)).dimshuffle(1,0,2)
            prob = prob_vocab + prob_alpha + prob_beta
    else:
        prob = vocab_
    
    p_flat = prob.flatten()
    
    cost = -T.log(p_flat[T.arange(select_std_flat.shape[0]) * prob.shape[2] + select_std_flat] + 1e-8)
    
    maskedCost = cost * outputMask_flat
    cost = maskedCost.sum() / outputMask_flat.sum()
    
    # Regularization
    if options["Coverage_aviliable"]:
        weight_lambda = theano.shared(np.float32(options['weight_lambda']), name = 'weight_lambda')
    
        def prefix_sum(data):
            '''
            n_steps = data.shape[0]
            n_sample = data.shape[1]
            n_dim = data.shape[2]
        
            total_init = T.alloc(numpy_floatX(0.), n_sample, n_dim)
            def step(data_i, total):
                total = total + data_i
                return total
            result, _ = theano.scan(step,
                                    sequences=[data],
                                    outputs_info = [total_init],
                                    name = 'prefix_sum',
                                    n_steps = n_steps)
            '''
            result = T.concatenate([T.zeros_like(data[:1], dtype = 'float32'),T.cumsum(data, axis = 0)], axis = 0)[:-1]
            return result
        
        if (options["LVT_aviliable"]) and (decoder_setting['_switcher']['n_out'] ==  3):
            alpha_sum = prefix_sum(alpha_)
            beta_sum = prefix_sum(beta_)
            cost_alpha = T.minimum(alpha_, alpha_sum)
            cost_beta = T.minimum(beta_, beta_sum)
            masks = (T.ones_like(cost_alpha, dtype = 'float32') * outputMask[1:,:,None]) * inputMask.dimshuffle(1,0)[None,:,:]
            masks_flat = masks.flatten()
            maskedCost_alpha = cost_alpha.flatten() * masks_flat
            maskedCost_beta = cost_beta.flatten() * masks_flat
            cost += weight_lambda * ((maskedCost_alpha.sum()+ maskedCost_beta.sum()) / masks_flat.sum())
        else:
            posi_sum = prefix_sum(posi_)
            cost_posi = T.minimum(posi_sum, posi_)
            masks = (T.ones_like(cost_posi, dtype = 'float32') * outputMask[1:,:,None]) * inputMask.dimshuffle(1,0)[None,:,:]
            masks_flat = masks.flatten()
            maskedCost_posi = cost_posi.flatten() * masks_flat
            cost += weight_lambda * (maskedCost_posi.sum()/ masks_flat.sum())
        
    # L2 Regularization
    
    weight_L2 = theano.shared(np.float32(options['weight_L2']), name='weight_L2')
    value_L2 = 0
    for kk, vv in params_shared.iteritems():
        value_L2 += (vv ** 2).sum()
    value_L2 *= weight_L2
    
    
    cost += value_L2
    
    # posi Regularization
    #cost_posi = posi_.dimshuffle(1,2,0)
    #cost_posi = T.minimum(abs(cost_posi.sum()- 1), 0)
    #cost+= cost_posi.sum()
    
    log.log('Stop Building Model')
    return inps, prob, cost, updates, encoder

def build_sampler(params_shared, options):
    
    modelName = options['model_name']
    embedding_setting = options['embedding']
    decoder_setting = options['decoder']
    
    inps = []
    
    word_input = T.scalar('word_input', dtype = 'int64')
    word_input_matrix = T.reshape(word_input, [1, 1], ndim = 2)
    
    inps += [word_input]
    
    if options["Structure_aviliable"]:
        emb_input = get_layer('embedding')[1]('embedding', params_shared, embedding_setting['vocab'], word_input_matrix, training = options['training'])
        struct_below = T.tensor3('struct_below', dtype = 'float32')
        input_embedding = T.tensor3('input_embedding', dtype = 'float32')
        inps += [input_embedding, struct_below]
        if options['Parent_aviliable']:
            parent = T.matrix('parent', dtype = 'int64')
            inps += [parent]
        else:
            inps += [None]
    else:
        emb_input = get_layer('embedding')[1]('embedding', params_shared, embedding_setting, word_input_matrix, training = options['training'])
        inps += [None, None, None]
    
    h_init = T.matrix('h_init', dtype = 'float32')
    c_init = T.matrix('c_init', dtype = 'float32')
    inps += [h_init, c_init]
    
    if options["Attention_aviliable"]:
        state_below =  T.tensor3('state_below', dtype = 'float32')
        mask_below = T.matrix('mask_below', dtype = 'float32')
        
        inps += [state_below, mask_below]
    else:
        inps += [None, None]
    
    if options["LVT_aviliable"]:
        batch_vocab = T.vector('batch_vocab', dtype = 'int64')
        pointer = T.matrix('pointer', dtype = 'int64')
        inps+= [batch_vocab, pointer]
    else:
        inps += [None, None]
        
    if modelName == 'struct_fei_decoder':
        hs_init = T.matrix('hs_init', dtype = 'float32')
        cs_init = T.matrix('cs_init', dtype = 'float32')
        inps += [hs_init, cs_init]
    else:
        inps += [None, None]
        
    inps_avil = [item for item in inps if item is not None]
    
    # Decoder Layer
    outps = []
    
    if modelName == 'baseline':
        h_next, c_next, vocab_ = get_layer('baseline_decoder')[1]('decoder', params_shared, decoder_setting, emb_input, h_init, c_init, training = options['training'])
    elif modelName == 'attention':
        h_next, c_next, vocab_, posi_ = get_layer('attention_decoder')[1]('decoder', params_shared, decoder_setting, state_below, mask_below, emb_input,  h_init, c_init, training = options['training'])
    elif modelName == 'switcher_pointer':
        h_next, c_next, vocab_, posi_, switcher_ = get_layer('switcher_pointer_decoder')[1]('decoder', params_shared, decoder_setting, state_below, mask_below, emb_input,  h_init, c_init, training = options['training'])
    elif modelName == 'struct_one_v1':
        h_next, c_next, vocab_, posi_, switcher_ = get_layer('switcher_pointer_decoder')[1]('decoder', params_shared, decoder_setting, state_below, mask_below, emb_input,  h_init, c_init, training = options['training'])
    elif modelName == 'struct_one_v2':
        one_info = T.concatenate([state_below, struct_below], axis = 2)
        h_next, c_next, vocab_, posi_, switcher_ = get_layer('switcher_pointer_decoder')[1]('decoder', params_shared, decoder_setting, one_info, mask_below, emb_input, h_init, c_init, training = options['training'])
    elif modelName == 'struct_node':
        struct_info = T.concatenate([input_embedding, struct_below], axis = 2)
        h_next, c_next, vocab_, alpha_, beta_, posi_, switcher_ = get_layer('struct_node_decoder')[1]('decoder', params_shared, decoder_setting, state_below, struct_info, mask_below, emb_input, h_init, c_init, training = options['training'])
    elif modelName == 'struct_edge':
        struct_info = T.concatenate([input_embedding, struct_below], axis = 2)
        parent_info = indexing_calc('parent', params_shared, struct_info, parent)
        struct_info = T.concatenate([struct_info, parent_info], axis = 2)
        h_next, c_next, vocab_, alpha_, beta_, gamma_, posi_, switcher_ = get_layer('struct_edge_decoder')[1]('decoder', params_shared, decoder_setting, state_below, struct_info, parent, mask_below, emb_input, h_init, c_init, training = options['training'])

    outps += [h_next, c_next]
    if modelName == 'struct_fei_decoder':
        outps += [hs_next, cs_next] 
    else:
        outps += [None, None]
        
        
    if options["LVT_aviliable"]:
        if decoder_setting['_switcher']['n_out'] < 2:
            prob_vocab = switcher_[:,:,None] * T.concatenate([vocab_, T.zeros((vocab_.shape[0], vocab_.shape[1], batch_vocab.shape[0] - vocab_.shape[2]), dtype = 'float32')], axis = 2)
            shp = pointer.shape
            pointer = T.reshape(T.extra_ops.to_one_hot(pointer.flatten(), batch_vocab.shape[0]),[shp[0], shp[1], batch_vocab.shape[0]], ndim = 3)
            prob_posi = (1 - switcher_[:,:,None]) * T.batched_dot(posi_.dimshuffle(1,0,2), pointer.dimshuffle(1,0,2)).dimshuffle(1,0,2)
            prob = prob_vocab + prob_posi
            
        elif decoder_setting['_switcher']['n_out'] ==  3:
            switcher_vocab = switcher_[:,:,0].dimshuffle(0,1,)
            switcher_alpha = switcher_[:,:,1].dimshuffle(0,1,)
            switcher_beta = switcher_[:,:,2].dimshuffle(0,1,)
            prob_vocab = switcher_vocab[:,:,None] * T.concatenate([vocab_, T.zeros((vocab_.shape[0], vocab_.shape[1], batch_vocab.shape[0] - vocab_.shape[2]), dtype = 'float32')], axis = 2)
            shp = pointer.shape
            pointer = T.reshape(T.extra_ops.to_one_hot(pointer.flatten(), batch_vocab.shape[0]),[shp[0], shp[1], batch_vocab.shape[0]], ndim = 3)
            prob_alpha = switcher_alpha[:,:,None] * T.batched_dot(alpha_.dimshuffle(1,0,2), pointer.dimshuffle(1,0,2)).dimshuffle(1,0,2)
            prob_beta = switcher_beta[:,:,None] * T.batched_dot(beta_.dimshuffle(1,0,2), pointer.dimshuffle(1,0,2)).dimshuffle(1,0,2)
            prob = prob_vocab + prob_alpha + prob_beta
    else:
        prob = vocab_
        
    outps += [prob, posi_]
    outps_avil = [item for item in outps if item is not None]

    decoder = theano.function(inputs = inps_avil,
                              outputs = outps_avil,
                              on_unused_input='ignore')
    return inps, outps, decoder
