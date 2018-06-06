import sys
import theano.tensor as T
from mylog.mylog import mylog
from utility.utility import *

from data_processor.data_manager import *
from data_processor.data_loader import data_loader

from build_model.build_model import build_model, build_sampler
from build_model.parameters import *

from generation.generation import *
from vocabulary.vocabulary import Vocabulary
from evaluation.evaluation import evalFile

from options_loader import *
from optimizer import *


def summarize(encoder, encoderInputs, decoder,  otherInputs, OriginalText, Vocab, options, log):
    result, time_data = gen_sample(encoder, encoderInputs, decoder, otherInputs, Vocab, options, log)
    result = sorted(result, key = lambda x:x[0])
    #print result[0][1][1:]
    sentence = translateSequence_new(result[0][1][1:], OriginalText, Vocab, options)
    #print sentence
    return sentence, time_data

def test_once(dataset, encoder, decoder, OriginalText, Vocab, options, log):
    data = dataset[0]
    print len(dataset), len(dataset[0]), len(dataset[1]), len(dataset[2])
    
    
    number = len(data)
    summary = ''
    reference = ''
    document = ''
    time_data = ''
    time_sum = (0.0,0.0,0.0,0.0)
    log.log('Start Beam Searching')
    
    for i in range(0, number):
        #log.log('Dealing with the %d-th Instance'%(i))
        batchedData = get_Kth_Instance(i, dataset)
        #print batchedData
        inps = batch2Inputs_new(batchedData, options)
        
        
        encoderInputs = [inps[0], inps[1], inps[4]]
        
        otherInputs = {}
        otherInputs['batch_vocab'] = inps[2]
        otherInputs['pointer'] = inps[3]
        otherInputs['parent'] = inps[5]
        
        summary_i, time_data_i = summarize(encoder, [inp for inp in encoderInputs if inp is not None], decoder, otherInputs, OriginalText[i], Vocab, options, log)
        reference_i = ListOfIndex2Sentence(cutDown(batchedData[1][0][1:]),Vocab,options)
        document_i = ListOfIndex2Sentence(batchedData[0][0], Vocab, options)
        
        document += document_i + '\n'
        summary += summary_i + '\n'
        reference += reference_i + '\n'
        time_data += str(time_data_i) + '\n'
        time_sum = [sum(x) for x in zip(time_sum, time_data_i)]
        
    time_sum = [(x+0.0) / (number+1e-8) for x in time_sum]
    return document, reference, summary, time_data, time_sum

def evaluate(hyp_fileName, ref_fileName, metrics, log, Show = True):
    Eval = evalFile(hyp_fileName, ref_fileName, metrics)    
    result = Eval.eval()
    if Show:
        for kk,vv in result.items():
            print kk
            print vv
    return result


def prepare(optionName, modelName, dataset, testSet, Vocab, I2Es, log):
    options = optionsLoader(log, False, optionName)

    
    params = init_params(options, Vocab, I2Es, log)
    if options['reload'] == True:
        log.log('Start reloading Parameters.')
        params = load_params(modelName, params)
        log.log('Finish reloading Parameters.')
    
    options["training"] = False
    options["test"] = True
    
    if 'decoder_epsilon' in params:
        log.log('Decoder Epsilon:'+str(params['decoder_epsilon']))
    
    params_shared = init_params_shared(params)
    
    inps_all, dist, cost, updates, encoder = build_model(params_shared, options, log)
    inps_aviliable = [item for item in inps_all if item is not None]
    
    
    inps_dec, outps_dec, decoder = build_sampler(params_shared, options)
    testData = dataset.get_first_K_instances(4096, testSet)
    
    return testData, encoder, decoder, options

def generate(prefix, testData, encoder, decoder, OrignialText, Vocab, options, log, beam_size = 5, bigramTrick = False, gamma = 7):
    log.log('Using Beam_Search')
    if len(testData[0]) != 500:
        options['generation_method'] = 'bfs_beam'
    else:
        options['generation_method'] = 'bfs_beam_75'
    options['beam_size'] = beam_size
    options['gamma'] = gamma
    options['apply_bigram_trick'] = bigramTrick
    options["training"] = False
    options["test"] = True
    
    document, reference, summary, time_data, time_avg = test_once(testData, encoder, decoder, OrignialText, Vocab, options, log)
    
    writeFile(prefix + '.document', document)
    writeFile(prefix + '.reference', reference)
    writeFile(prefix + '.summary', summary)
    writeFile(prefix + '.counts', time_data)

def loadFromText(fName):
    f = codecs.open(fName,'r',encoding = 'utf-8')
    result = []
    for l in f:
        line = l.strip().split()
        result.append(line)
    return result
    
    
if __name__ == '__main__':
    log = mylog()
    dataoptions = optionsLoader(log, True)
    # Load the Vocabulary and Features and Dataset First
    Vocab_Giga = loadFromPKL('giga_new.Vocab')
    Vocab = {
        'w2i':Vocab_Giga.w2i,
        'i2w':Vocab_Giga.i2w,
        'i2e':Vocab_Giga.i2e
    }
    
    Features_Giga = loadFromPKL('features.Embedding')
    I2Es = []
    for feat in dataoptions["featList"]:
        I2Es.append(Features_Giga[feat].i2e)
    
    dataset = data_loader(Vocab, dataoptions, log)
    
    Index = 0
    optionName = './model/struct_edge/options_check2_best.json'
    modelName = './model/struct_edge/model_check2_best.npz'
    for part in dataoptions['subsets']:
        OrignialText = loadFromText(dataoptions['primary_dir']+dataoptions[part]+'.Ndocument') 
        Index += 1
        log.log('Testing %d th model'%(Index))
        testData, encoder, decoder, options = prepare(optionName, modelName, dataset, part , Vocab, I2Es, log)
        generate(part+'.result', testData, encoder, decoder, OrignialText, Vocab, options, log, beam_size = 5, bigramTrick=True, gamma = 13.284)
        
    log.log('Finish Testing')
