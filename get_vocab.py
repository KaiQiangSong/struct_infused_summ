import codecs, sys

from mylog.mylog import mylog
from vocabulary.vocabulary import Vocabulary, I2E
from utility.utility import *
from options_loader import *

if __name__ == '__main__':
    #Vocab = loadFromPKL('../../vocab/gigaword.pkl')
    
    log_vocab = mylog(logFile = 'log/log_vocab')
    options = optionsLoader(log_vocab, True)
    
    fileName = options['primary_dir'] 
    
    inputCorpus = [fileName + options['trainSet'] + '.Ndocument']
    
    outputCorpus= [fileName + options['trainSet'] + '.Nsummary']
    
    Vocab = Vocabulary(options, inputCorpus = inputCorpus, outputCorpus = outputCorpus)
    
    log_vocab.log(str(Vocab.full_size)+', '+str(Vocab.n_in) + ', ' + str(Vocab.n_out))
    
    saveToPKL(fileName+sys.argv[1]+'.Vocab',Vocab)
    
    f = codecs.open(fileName+sys.argv[1]+'.i2w','w',encoding ='utf-8')
    for item in Vocab.i2w:
        if (item == '<unk>' or item == '<s>'):
            print >> f, item, 'NAN'
        else:
            print >> f, item, Vocab.typeFreq[item]
    
    FeatureEmbedding = {}
    for feat in options["featList"]:
        FeatureEmbedding[feat] = I2E(options, feat)
    
    log_vocab.log(str(options["featList"]))
    saveToPKL(fileName+'features.Embedding',FeatureEmbedding)
    
    options_vocab = loadFromJson('settings/vocabulary.json')
    options_vocab['vocab_size'] = Vocab.full_size
    options_vocab['vocab_full_size'] = Vocab.full_size
    options_vocab['vocab_input_size'] = Vocab.n_in
    options_vocab['vocab_output_size'] = Vocab.n_out
    saveToJson('settings/vocabulary.json', options_vocab)
    
    