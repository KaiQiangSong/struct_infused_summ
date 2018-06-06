import scipy
import theano
import theano.tensor as T
import numpy as np
import copy

from sets import Set
from utility.utility import *

#from mylog.mylog import mylog

'''
    Functionality:
        In this file, our goal is to manage the format of data in different level
        
        Assume that all the document are pre-processed as format below:
            lower letters:
                all words are forced to be lowercase
            unknown words:
                using a symbol <unk> to represent the unknown words
            number pattern:
                #.# for numbers like 1.5
                ### for number like 123
                ...
            no </s> in the text
        
        
    Word level:
        Word(string)
        Index
        Embedding
    Sentence level:
        Sentence(string)
        sequence of Word(string)
        sequence of Word Index
        Embedding matrix
    Document level:
        Document(string)
        sequence of Sentence(string)
        sequence of sequence of Word(string)
        sequence of sequence of Word Index
        Embedding matrix (add sentence boundary at the end)
'''

'''
Word Level Operations:
    Index2Word
    Index2Embedding
    Word2Index
    Word2Embedding
'''

def Index2Word(Index, Vocab, options):
    # For some corner cases
    return Vocab['i2w'][Index]

def Index2Embedding(Index, Vocab, options):
    # For some corner cases
    return Vocab['i2e'][Index]

def Word2Index(Word, Vocab, options, flag):
    if not (Word in Vocab['w2i']):
        Word = '<unk>'
    elif flag and (Vocab['w2i'][Word] >= options['vocab_size']):
        Word = '<unk>'
    return Vocab['w2i'][Word]

def Word2Embedding(Word, Vocab,options, flag):
    return Index2Embedding(Word2Index(Word, Vocab, options, flag), Vocab, options)

'''
    Sentence Level Operations:
        Sentence2ListOfWord
        ListOfWord2ListOfIndex
        Sentence2ListOfIndex
        ListOfIndex2Embedding
        Sentence2Embedding 
        
        ListOfIndex2ListOfWord
        ListOfWord2Sentence
        ListOfIndex2Sentence
inps
    Options:
        endOfSentence:
            (add </s> after sentence)
'''

def Sentence2ListOfWord(sentence):
    listOfWord = sentence.split()
    return listOfWord

def ListOfWord2ListOfIndex(listOfWord, Vocab, options, flag):
    listOfIndex = []
    for w in listOfWord:
        listOfIndex.append(Word2Index(w, Vocab, options, flag))
    return listOfIndex
    
def Sentence2ListOfIndex(sentence, Vocab, options, flag):
    return ListOfWord2ListOfIndex(Sentence2ListOfWord(sentence),Vocab, options, flag)

def ListOfIndex2Embedding(listOfIndex, Vocab, options):
    emb = np.empty((0, options['emb_dim']), dtype = eval(options['dtype_float_numpy']))
    for wi in listOfIndex:
        emb = np.append(emb, Index2Embedding(wi, Vocab, options)[None,:], axis = 0)
    return emb

def Sentence2Embedding(sentence, Vocab, options, flag):
    return ListOfIndex2Embedding(Sentence2ListOfIndex(sentence, Vocab, options, flag), Vocab, options)

def ListOfIndex2ListOfWord(listOfIndex, Vocab, options):
    return [Index2Word(Index, Vocab, options) for Index in listOfIndex]

def ListOfWord2Sentence(listOfWord):
    first = True
    sentence = ''
    for word in listOfWord:
        if first:
            first = False
            sentence += word
        else:
            sentence += ' ' + word
    return sentence

def ListOfIndex2Sentence(listOfIndex, Vocab, options):
    return ListOfWord2Sentence(ListOfIndex2ListOfWord(listOfIndex, Vocab, options))

def cutDown(listOfIndex, maxStep = None):
    if maxStep == None:
        if 1 in listOfIndex:
            maxStep = listOfIndex.index(1)
        else:
            maxStep = len(listOfIndex)
    result = listOfIndex[0:min(len(listOfIndex), maxStep)]
    
    return result

def convertWord2Position(document, summary, options):
    document = document.split()
    summary = summary.split()
    dict = {}
    length = len(document)
    for i in range(0, length):
        if not (document[i] in dict):
            dict[document[i]] = i
    
    position = []
    length = len(summary)
    for i in range(0, length):
        if summary[i] in dict:
            position.append(dict[summary[i]])
        else:
            position.append(options['max_posi'])
    return position

def cosSim(word1, word2, Vocab, options):
    emb1 = Word2Embedding(word1, Vocab, options, False)
    emb2 = Word2Embedding(word2, Vocab, options, False)
    return 1 - scipy.spatial.distance.cosine(emb1, emb2)

def convertWord2Position_LCS(document, summary, Vocab, options):
    document = document.split()
    summary = summary.split()
    N = len(document)
    M = len(summary)
    F = np.zeros((N + 1,M + 1), dtype = eval(options['dtype_float_numpy']))
    G = np.zeros((N + 1,M + 1), dtype = eval(options['dtype_int_numpy']))
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            Sim = cosSim(document[i-1],summary[j-1], Vocab, options)
            if F[i-1][j] > F[i][j-1]:
                F[i][j] = F[i-1][j]
                G[i][j] = 1
            else:
                F[i][j] = F[i][j-1]
                G[i][j] = 2
            if (Sim > 0.5) and (F[i-1][j-1] + Sim > F[i][j]):
                F[i][j] = F[i-1][j-1] + Sim
                G[i][j] = 3
    
    position = []
    i = N
    j = M
    
    while (j>0):
        if (i > 0):
            if G[i][j] == 1:
                i -= 1
            elif G[i][j] == 2:
                position.append(options['max_posi'])
                j -= 1
            else:
                position.append(i-1)
                i -= 1
                j -= 1
        else:
            position.append(options['max_posi'])
            j -= 1
    position.reverse()
    return position

'''
    Batched Data Preparation
'''


def ListOfEmbedding2BatchedEmbedding(listOfEmbedding, options, maxStep = None):
    if maxStep == None:
        n_steps = [emb.shape[0] for emb in listOfEmbedding]
        maxStep = max(n_steps)
        
    n_dim = listOfEmbedding[0].shape[1]
    data = np.empty((maxStep, 0 , n_dim), dtype = eval(options['dtype_float_numpy']))
    mask = np.empty((maxStep, 0), dtype = eval(options['dtype_float_numpy']))
    
    for emb in listOfEmbedding:
        data_i = np.append(emb, np.zeros((maxStep - emb.shape[0], n_dim), dtype = eval(options['dtype_float_numpy'])), axis = 0)
        data = np.append(data, data_i.reshape(data_i.shape[0], 1, data_i.shape[1]), axis = 1)
        
        mask_i = np.append(np.ones((emb.shape[0],1), dtype = eval(options['dtype_float_numpy'])),
                           np.zeros((maxStep - emb.shape[0],1), dtype = eval(options['dtype_float_numpy'])),
                           axis = 0)
        
        mask = np.append(mask, mask_i, axis = 1)
    return data, mask

def listOfListOfIndex2BatchedEmbedding(listOfListOfIndex, Vocab, options, maxStep = None):
    if maxStep == None:
        n_steps = [len(it) for it in listOfListOfIndex]
        maxStep = max(n_steps)
        
    listOfEmbedding = []
    for listOfIndex in listOfListOfIndex:
        if len(listOfIndex) > maxStep:
            listOfEmbedding.append(ListOfIndex2Embedding(listOfIndex[0:maxStep], Vocab, options))
        else:        
            listOfEmbedding.append(ListOfIndex2Embedding(listOfIndex, Vocab, options))
    return ListOfEmbedding2BatchedEmbedding(listOfEmbedding, options, maxStep)

def listOfSentence2BatchedEmbedding(listOfSentence, Vocab, options, flag):
    listOfEmbedding = []
    for sentence in listOfSentence:
        listOfEmbedding.append(Sentence2Embedding(sentence, Vocab, options, flag))
    return ListOfEmbedding2BatchedEmbedding(listOfEmbedding, options)

def listOfListOfIndex2BatchedAnnotation(listOfListOfIndex, options ,maxStep = None):
    if maxStep == None:
        n_steps = [len(it) for it in listOfListOfIndex]
        maxStep = max(n_steps)
        
    data = np.empty((maxStep, 0), dtype = eval(options['dtype_int_numpy']))
    mask = np.empty((maxStep, 0), dtype = eval(options['dtype_float_numpy']))
    
    for listOfIndex in listOfListOfIndex:
        data_i = np.array(listOfIndex, dtype = eval(options['dtype_int_numpy']))
        data_i = data_i.reshape(data_i.shape[0], 1)
        data_i = np.append(data_i, np.zeros((maxStep - len(listOfIndex),1), dtype = eval(options['dtype_int_numpy'])), axis = 0)
        data = np.append(data, data_i, axis = 1)
        
        mask_i = np.append(np.ones((len(listOfIndex),1), dtype = eval(options['dtype_float_numpy'])),
                           np.zeros((maxStep - len(listOfIndex),1), dtype = eval(options['dtype_float_numpy'])),
                           axis = 0)
        mask = np.append(mask, mask_i, axis = 1)
    return data, mask


'''
    Generate Batches
'''

def paddingList(list, max_length, patch):
    length = len(list)
    for i in range(length, max_length):
        list.append(patch)
    return list

def Generate_Batches(dataset, batch_size):
    data = dataset[0]
    anno = dataset[1]
    posi = dataset[2]
    number = len(anno)
    
    number_batch = number / batch_size
    batches = []
    for bid in range(0, number_batch):
        batches.append((data[bid * batch_size : (bid+1) * batch_size],
                        anno[bid * batch_size : (bid+1) * batch_size],
                        posi[bid * batch_size : (bid+1) * batch_size]))
    if (number % batch_size > 0):
        batches.append((data[number-batch_size:number],
                        anno[number-batch_size:number],
                        posi[number-batch_size:number]))
        number_batch += 1
    return number_batch, batches

def Get_Kth_Batch(batches, K, RandomIndex):
    return batches[RandomIndex[K]]

def get_Kth_Instance(K, batchedData):
        return ([batchedData[0][K]] , [batchedData[1][K]], [batchedData[2][K]])

def reverseListOfList(listOfListOfElement):
    result = []
    for listOfElement in listOfListOfElement:
        reversedListOfElement = copy.deepcopy(listOfElement)
        reversedListOfElement.reverse()
        result.append(reversedListOfElement)
    return result

def ListOfListOfElement_LeftShift(listOFListOfElement):
    result = []
    for listOfElement in listOFListOfElement:
        if len(listOfElement) > 0:
            shiftedListOfElement = copy.deepcopy(listOfElement[1:])
        else:
            shiftedListOfElement = []
        result.append(shiftedListOfElement)
    return result

'''
    New way to get Switch List
                       inVinD    inVoutD       outVinD    outVoutD      </s>
    VocabIndex         Index     Index         0          0             1
    DocumentPosition   Posi      max_posi      Posi       max_posi      max_posi+1
    Switch             0         0             1          0             0
'''


def listOfListOfIndex2ListOfListAnnotation(listOfListOfIndex, listOfListOfPosition, listOfVocab, options):
    n_instances = len(listOfListOfIndex)
    n_vocab = len(listOfVocab)
    
    switch = []
    select = []
    for i in range(0,n_instances):
        switch_i = []
        select_i = []
        n_steps = len(listOfListOfIndex[i])
        for j in range(0,n_steps):
            switch_i_j = options['unk_switch']
            select_i_j = options['unk_select']
            for k in range(0, n_vocab):
                if  listOfListOfIndex[i][j] in listOfVocab[k]:
                    switch_i_j = k
                    select_i_j = listOfVocab[k].index(listOfListOfIndex[i][j])
            if (select_i_j == options['unk_switch']) and (select_i_j == options['unk_select']) and (options['use_position_prediction']):
                if listOfListOfPosition[i][j] < options['max_posi']:
                    switch_i_j = n_vocab
                    select_i_j =listOfListOfPosition[i][j]
            switch_i.append(switch_i_j)
            select_i.append(select_i_j)
        switch.append(switch_i)
        select.append(select_i)
    
    return switch, select, n_instances, n_vocab + int(options['use_position_prediction'])

def listOfListOfAnnotation2BatchedAnnotation(listOfListOfSwitch, listOfListOfSelect, n_instances, n_vocab, n_step, bias, options):
    floatType = eval(options['dtype_float_numpy'])
    intType = eval(options['dtype_int_numpy'])
    
    
    #switch_std = np.empty((n_step, 0), dtype = intType)
    switch_std = np.empty((n_step, 0, n_vocab), dtype = floatType)
    switch_bias = np.empty((n_step, 0), dtype = intType)
    select = np.empty((n_step, 0), dtype = intType)
    mask = np.empty((n_step, 0), dtype = floatType)
    
    for i in range(0, n_instances):
        data_length = len(listOfListOfSwitch[i])
        padding_length = n_step - data_length
        
        '''
        switch_std_i = np.append(np.asarray(listOfListOfSwitch[i], dtype = intType)[:, None],
                             np.zeros((padding_length, 1), dtype = intType),
                             axis = 0)
        '''
        switch_std_i = np.zeros((n_step, 1, n_vocab), dtype = floatType)
        switch_std_i[np.arange(0, data_length), 0 , listOfListOfSwitch[i]] = 1.
        listOfBias = [bias[d] for d in listOfListOfSwitch[i]]
        #print listOfListOfSwitch[i]
        #print listOfBias
        switch_bias_i = np.append(np.asarray(listOfBias, dtype = intType)[:, None],
                                  np.zeros((padding_length, 1), dtype = intType),
                                  axis = 0)
        
        select_i = np.append(np.asarray(listOfListOfSelect[i], dtype = intType)[:,None],
                             np.zeros((padding_length, 1), dtype = intType),
                             axis = 0)
        
        mask_i = np.append(np.ones((data_length, 1), dtype = floatType),
                           np.zeros((padding_length, 1), dtype = floatType),
                           axis = 0)
        
        switch_std = np.append(switch_std, switch_std_i, axis = 1)
        switch_bias = np.append(switch_bias, switch_bias_i, axis = 1)
        select = np.append(select, select_i, axis = 1)
        mask = np.append(mask, mask_i, axis = 1)
    
    return switch_std, switch_bias, select, mask
    
def sharpVocab(listOfListOfElement, vocab_size):
    result = []
    for listOfElement in listOfListOfElement:
        result.append([(int(item < vocab_size) * item) for item in listOfElement])
    return result

'''
def batch2Inputs(batch, Vocab, options):
    
    x = batch[0]
    y = batch[1]
    z = batch[2]
    batch_size = len(y)
    
    if options['reverseDecoder']:
        y = reverseListOfList(y)
        z = reverseListOfList(z)
    
    y = sharpVocab(y, options['vocab_size'])
    
    inps = []
    
    input, inputMask = listOfListOfIndex2BatchedEmbedding(x, Vocab, options)
    goldStd, _ = listOfListOfIndex2BatchedEmbedding(y, Vocab, options, options['max_len']-1)
    
    inps += [input, inputMask, goldStd]
    
    y = ListOfListOfElement_LeftShift(y)
    output, outputMask = listOfListOfIndex2BatchedAnnotation(y, options, options['max_len']-1)
    inps += [output, outputMask]
    
    
    #z = ListOfListOfElement_LeftShift(z)
    
    
    return inps
'''
def LVT(x, options):
    vocab_LVT = Set(range(0,options['decoder']['_softmax']['n_out']))
    for l in x:
        vocab_LVT = vocab_LVT.union(Set(l))
    vocab_LVT = list(vocab_LVT)
    vocab_LVT = sorted(list(vocab_LVT))
    dict_LVT = {}
    Index = 0
    for w in vocab_LVT:
        dict_LVT[w] = Index
        Index += 1
    return vocab_LVT, dict_LVT

def get_pointer(x, dict):
    result = []
    for l in x:
        result.append([dict[w] for w in l])
    return result

def sharpLVT(x, dict):
    result = []
    for l in x:
        temp = []
        for w in l:
            if w in dict:
                temp.append(dict[w])
            else:
                temp.append(0)
        result.append(temp)
    return result
        
def batch2Inputs(batch, options):
    x = batch[0]
    y = batch[1]
    
    batch_size = len(y)
    batch_vocab, batch_dict = LVT(x, options)
    batch_vocab = np.asarray(batch_vocab, dtype=np.int64)
    pointer = get_pointer(x, batch_dict)
    
    inps = []
    input, inputMask = listOfListOfIndex2BatchedAnnotation(x, options)
    pointer, _ = listOfListOfIndex2BatchedAnnotation(pointer, options)
    inps += [input, inputMask, batch_vocab, pointer]
    
    y = sharpLVT(y, batch_dict)
    output, outputMask = listOfListOfIndex2BatchedAnnotation(y, options)
    
    inps += [output, outputMask]
    return inps

def feature2BatchFeature(listOfSentenceFeatures, options, maxStep = None):
    
    intType = eval(options['dtype_int_numpy'])
    featList = options["featList"]
    if maxStep == None:
        lengths = []
        for sent in listOfSentenceFeatures:
            flen = [len(data) for data in sent.values()]
            if (len(Set(flen)) != 1):
                print 'Error'
                return 'Error'
            lengths.append(flen[0])
        maxStep = max(lengths)
    
    Features = {}
    for feat in featList:
        Features[feat] = np.empty((maxStep, 0), dtype = intType)
    
    for sent in listOfSentenceFeatures:
        flen = [len(data) for data in sent.values()][0]
        for feat in featList:
            feature = np.append(np.asarray(sent[feat], dtype = intType)[:,None],
                                np.zeros((maxStep - flen, 1), dtype = intType),
                                axis = 0)
            Features[feat] = np.append(Features[feat], feature, axis = 1)
    
    result = np.empty((maxStep, len(listOfSentenceFeatures), 0), dtype = intType)
    for feat in featList:
        result = np.append(result, Features[feat][:,:,None], axis = 2)
    '''
            
    result = []
    for feat in featList:
        result.append(Features[feat])
    '''     
    return result
        

def batch2Inputs_new(batch, options):
    x = batch[0]
    y = batch[1]
    z = batch[2]
    inps = []
    input, inputMask = listOfListOfIndex2BatchedAnnotation(x, options)
    inps += [input, inputMask]
    if options['LVT_aviliable']:
        batch_vocab, batch_dict = LVT(x, options)
        batch_vocab = np.asarray(batch_vocab, dtype=np.int64)
        pointer = get_pointer(x, batch_dict)
        pointer, _ = listOfListOfIndex2BatchedAnnotation(pointer, options)
        y = sharpLVT(y, batch_dict)
        inps += [batch_vocab, pointer]
    else:
        inps += [None, None]
        
    if options['Structure_aviliable']:
        feat = feature2BatchFeature(z, options)
        inps += [feat]
        if feat.shape[0] != input.shape[0]:
            print 'Feature Length Error', feat.shape, input.shape
        if options['Parent_aviliable']:
            parent, _ = listOfListOfIndex2BatchedAnnotation([d["parent"] for d in z],options)
            inps += [parent]
        else:
            inps += [None]
    else:
        inps += [None, None]
    

    
    output, outputMask = listOfListOfIndex2BatchedAnnotation(y, options)
    
    inps += [output, outputMask]
    
    return inps

def batch_merge(batch1, batch2):
    return (batch1[0] + batch2[0], batch1[1] + batch2[1], batch1[2] + batch2[2])


def translateWord(data, bias, options):
    switch = 0
    while (switch + 1 < len(bias)) and (bias[switch + 1] <= data):
        switch += 1
    select = data - bias[switch]
    return switch, select

def checkEnd(switch, select, n_vocab, options):
    if (switch == 0) and (select == 1):
        return True
    if (switch == n_vocab) and (options['use_position_prediction']) and (select == options['max_posi']+1):
        return True
    return False

def translateSequence(sequence, Vocab, options):
    sentence = ''
    for item in sequence:
        if item == 1:
            break
        word = Vocab['i2w'][item]
        sentence += word + ' '
    return sentence

def translateSequence_new(sequence, OriginalText, Vocab, options):
    sentence = ''
    for item in sequence:
        if item[0] == 1:
            break
        if item[0] == 0:
            att = int(item[1].argmax())
            word = OriginalText[att]
        else:  
            word = Vocab['i2w'][item[0]]
            if '#' in word:
                cands = []
                Index = 0
                for token in OriginalText:
                    if remove_digits(token).lower() == word:
                        cands += [Index]
                    Index += 1
                if Index != 0:
                    bestScore = 0.0
                    bestCand = 0
                    for cand in cands:
                        if float(item[1][cand]) > bestScore:
                            bestScore = float(item[1][cand])
                            bestCand = cand
                    word = OriginalText[bestCand]
        word = word.lower()
        sentence += word + ' '
    return sentence

def getInputs(word_input, previous_state, state_below, mask_below, otherInputs, options, input_embedding = None, struct_below = None):
    decoder_setting = options['decoder']
    
    # Deal with inputs
    inps = []
    # word_input
    inps += [word_input]
    # feature
    if options["Structure_aviliable"]:
        inps += [input_embedding, struct_below]
        if options["Parent_aviliable"]:
            inps += [otherInputs['parent']]
    # h_init, c_init
    inps += [previous_state[0], previous_state[1]]
    # state_below, mask_below:
    if options["Attention_aviliable"]:
        inps += [state_below, mask_below]
    # batch_vocab, pointer
    if options["LVT_aviliable"]:
        inps+= [otherInputs['batch_vocab'], otherInputs['pointer']]
    # hs_init, cs_init
    if decoder_setting['type'] == 'struct_fei_decoder':
        inps += [previous_state[2], previous_state[3]]
        
    return inps

def getInputs_new(word_input, previous_state, state_below, mask_below, otherInputs, options, input_embedding = None, struct_below = None):
    decoder_setting = options['decoder']
    
    # Deal with inputs
    inps = []
    # word_input
    inps += [word_input[0]]
    # feature
    if options["Structure_aviliable"]:
        inps += [input_embedding, struct_below]
        if options["Parent_aviliable"]:
            inps += [otherInputs['parent']]
    # h_init, c_init
    inps += [previous_state[0], previous_state[1]]
    # state_below, mask_below:
    if options["Attention_aviliable"]:
        inps += [state_below, mask_below]
    # batch_vocab, pointer
    if options["LVT_aviliable"]:
        inps+= [otherInputs['batch_vocab'], otherInputs['pointer']]
    # hs_init, cs_init
    if decoder_setting['type'] == 'struct_fei_decoder':
        inps += [previous_state[2], previous_state[3]]
        
    return inps

def topKIndexes(dist, K, gamma = 0):
    indexes = np.argpartition(dist, -K)[-K:]
    if gamma == 0:
        return indexes.flatten()
    threshold = np.min(dist[indexes]) - gamma
    indexes = np.argwhere(dist > threshold)
    return indexes.flatten()

def biGramTrick(dist, word_input, bi_in, bi_old, options, batch_vocab = None):
    indexes = topKIndexes(dist, options['beam_size'], options['gamma'])
    
    if batch_vocab is None:
        batch_vocab = np.arange(options['decoder']['_softmax']['n_out'], dtype = np.int64)
        
    bi_new = [(word_input, word) for word in batch_vocab[indexes].flatten().tolist()]
    
    trick_item = [int((biGram in bi_in) & (biGram not in bi_old)) for biGram in bi_new]
    trick_item = np.asarray(trick_item, dtype = np.float32) 
    bi_next = [copy.deepcopy(bi_old).union(biGram) if ((biGram in bi_in) & (biGram not in bi_old)) else copy.deepcopy(bi_old) for biGram in bi_new]
    dist = dist[indexes]
    dist = dist + options['gamma'] * trick_item
    return dist, indexes, bi_next
'''
def biGramTrick_new(dist, word_input, bi_in, bi_old, options, batch_vocab = None):
    ratio = options['gamma'] / (len(bi_in) + 1e-8)
    indexes = topKIndexes(dist, options['beam_size'], ratio)
    
    if batch_vocab is None:
        batch_vocab = np.arange(options['decoder']['_softmax']['n_out'], dtype = np.int64)
    
    #print type(indexes), type(batch_vocab)
    #print indexes
    #print batch_vocab
    
    bi_new = [(word_input, word) for word in batch_vocab[indexes].flatten().tolist()]
    
    trick_item = [int((biGram in bi_in) & (biGram not in bi_old)) for biGram in bi_new]
    trick_item = np.asarray(trick_item, dtype = np.float32) 
    
    dist = dist[indexes].flatten()
    dist += ratio * trick_item
    
    newIndexes = topKIndexes(dist, options['beam_size'])
    indexes = indexes[newIndexes].flatten()
    dist = dist[newIndexes].flatten()
    bi_new = [bi_new[id] for id in newIndexes.tolist()]
    bi_next = [copy.deepcopy(bi_old).union(biGram) if ((biGram in bi_in) & (biGram not in bi_old)) else copy.deepcopy(bi_old) for biGram in bi_new]
    
    return dist, indexes, bi_next 
'''
def biGramTrick_new(dist, word_input, bi_in, bi_old, options, batch_vocab = None):
    ratio = options['gamma'] / (len(bi_in) + 1e-8)
    indexes = topKIndexes(dist, options['beam_size'], ratio)
    
    if batch_vocab is None:
        batch_vocab = np.arange(options['decoder']['_softmax']['n_out'], dtype = np.int64)
    
    #print type(indexes), type(batch_vocab)
    #print indexes
    #print batch_vocab
    
    bi_new = [(word_input, word) for word in batch_vocab[indexes].flatten().tolist()]
    
    trick_item = [int((biGram in bi_in) & (biGram not in bi_old)) for biGram in bi_new]
    trick_item = np.asarray(trick_item, dtype = np.float32) 
    
    dist = dist[indexes].flatten()
    dist += ratio * trick_item
    
    newIndexes = topKIndexes(dist, options['beam_size'])
    indexes = indexes[newIndexes].flatten()
    dist = dist[newIndexes].flatten()
    bi_new = [bi_new[id] for id in newIndexes.tolist()]
    bi_next = [copy.deepcopy(bi_old).union(biGram) if ((biGram in bi_in) & (biGram not in bi_old)) else copy.deepcopy(bi_old) for biGram in bi_new]
    
    return dist, indexes, bi_next 

def opt_k(states, k = 1, key = lambda x:x[0]):
    newStates = sorted(states, key = key)
    if (len(newStates) > k):
        newStates = newStates[:k]
    return newStates