import theano.tensor as T
import numpy as np
import copy

from options_loader import *

from mylog.mylog import mylog
from utility.utility import *

from Layers.Layers import *
from data_processor.data_manager import *
from data_processor.data_loader import data_loader

from build_model.build_model import build_model, build_sampler

def bfs_beam_search(encoder, encoderInputs, decoder, otherInputs, Vocab, options, log):
    
    #Get Some Basic Options
    maxStep = options['max_len']
    decoder_setting = options['decoder']
    
    #input Format
    if ('batch_vocab' in otherInputs) and (otherInputs['batch_vocab'] is not None):
        batch_vocab = otherInputs['batch_vocab']
    else:
        batch_vocab = np.arange(options['decoder']['_softmax']['n_out'], dtype = np.int64)
    
    # Go Through Encoder
    if options["Structure_aviliable"]:
        state_below, _, h_init, c_init, input_embedding, struct_below = encoder(*encoderInputs)
    else:
        state_below, _, h_init, c_init = encoder(*encoderInputs)
    
    input, inputMask = encoderInputs[0], encoderInputs[1]
        
    bi_in = get_nGram(input.flatten())
    
    start_state_pass = [h_init, c_init]
    
    if decoder_setting['type'] == 'struct_fei_decoder':
        hs_init = np.zeros((h_init.shape[0], decoder_setting['_att_2']['n_out']), dtype = np.float32)
        cs_init = np.zeros((c_init.shape[0], decoder_setting['_att_2']['n_out']), dtype = np.float32)
        start_state_pass += [hs_init, cs_init]
    
    # Generate Natural Language
    if options['apply_bigram_trick']:
        startState = (0, [(0,'<s>')], start_state_pass, set())
    else:
        startState = (0, [(0,'<s>')], start_state_pass)
        
    n_states, n_cand, n_top = (1, 1, 0)
    
    TopKStates = []
    Candidates = [[startState]]
    flag = False
    
    for step in range(0, maxStep):
        bestNextStates = []
        for currentState in Candidates[step]:
            
            if options['apply_bigram_trick']:
                score, sequence, previous_state, bi_old = currentState
            else:
                score, sequence, previous_state = currentState
            
            if options["Structure_aviliable"]:
                inps = getInputs_new(sequence[-1], previous_state, state_below, inputMask, otherInputs, options, input_embedding = input_embedding, struct_below = struct_below)
            else:
                inps = getInputs_new(sequence[-1], previous_state, state_below, inputMask, otherInputs, options)
                
            outps = decoder(*inps)
            state_pass, dist, posi = outps[:-2], outps[-2], outps[-1]
            dist = dist.flatten()
            state_pass = [s[-1] for s in state_pass]
            
            dist = np.log(dist + 1e-8)
            if options['apply_bigram_trick']:
                dist, indexes, bi_new = biGramTrick_new(dist, sequence[-1][0], bi_in, bi_old, options, batch_vocab)
            else:
                indexes = topKIndexes(dist, options['beam_size'])
                dist = dist[indexes].flatten()
            dist = -dist + score
            att = posi.flatten()
                
            '''
            # BiGram Trick
            if options['apply_bigram_trick']:
                dist, indexes, bi_new = biGramTrick(dist, sequence[-1], bi_in, bi_old, options, batch_vocab)
            else:
                indexes = np.asarray(range(dist.shape[0]), dtype = np.int64)
                
            newIndexes = topKIndexes(dist, options['beam_size'])
            dist = dist[newIndexes]
            indexes = indexes[newIndexes].flatten()
            
            if options['apply_bigram_trick']:
                bi_new = [bi_new[id] for id in newIndexes.flatten().tolist()]
            
            dist = -np.log(dist + 1e-8) + score
            '''
            
            if options['apply_bigram_trick']:        
                nextStates = [(dist[Index], sequence+[(batch_vocab[int(indexes[Index])], att)], state_pass, copy.deepcopy(bi_new[Index])) for Index in range(dist.shape[0])]
            else:
                nextStates = [(dist[Index], sequence+[(batch_vocab[int(indexes[Index])], att)], state_pass) for Index in range(dist.shape[0])]
            
            bestNextStates += nextStates
            n_states += len(nextStates)
            
        bestNextStates = sorted(bestNextStates, key = lambda x:x[0])
        n_new = 0
        newCandidates = []
        for i in range(0, len(bestNextStates)):
            if (bestNextStates[i][1][-1][0] == 1):
                n_top += 1
                TopKStates.append(bestNextStates[i])
                if n_top >= options['beam_size']:
                    flag = True
                    break
            else:
                n_new += 1
                newCandidates.append(bestNextStates[i])
                if n_new >= options['beam_size']:
                    break
        if flag:
            break
        n_cand += len(newCandidates)
        Candidates.append(newCandidates)
    
    lastCandidates = Candidates[-1]
    
    while n_top < options['beam_size']:
        #print n_top
        n_top += 1
        TopKStates.append(lastCandidates[0])
        lastCandidates = lastCandidates[1:]
        
    return TopKStates, (n_states, n_cand, n_top, TopKStates[0][0])
