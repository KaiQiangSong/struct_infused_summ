import theano.tensor as T
from options_loader import *
from optimizer import *

from mylog.mylog import mylog
from utility.utility import *
from vocabulary.vocabulary import Vocabulary
from data_processor.data_manager import *
from data_processor.data_loader import data_loader

from build_model.parameters import *
from build_model.build_model import build_model, build_sampler

from evaluation.evaluation import *

if __name__ == '__main__':
    log = mylog()
    optionName ='./model/options_best.json'
    modelName = './model/model_best.npz'
    
    options = optionsLoader(log, True, optionName)
    
    options['Coverage_aviliable'] = True
    options['weight_lambda'] = 1.0
    
    options["training"] = True
    options["test"] = False
    
    Vocab_Giga = loadFromPKL('../../dataset/gigaword_eng_5/giga_new.Vocab')
    Vocab = {
        'w2i':Vocab_Giga.w2i,
        'i2w':Vocab_Giga.i2w,
        'i2e':Vocab_Giga.i2e
    }
    
    Features_Giga = loadFromPKL(options['primary_dir']+'features.Embedding')
    I2Es = []
    for feat in options["featList"]:
        I2Es.append(Features_Giga[feat].i2e)
        
    dataset = data_loader(Vocab, options, log)
        
    params = init_params(options, Vocab, I2Es, log)
    
    if options['reload'] == True:
        log.log('Start reloading Parameters.')
        params = load_params(modelName, params)
        log.log('Finish reloading Parameters.')
        
    params_shared = init_params_shared(params)
    
    inps_all, dist, cost, updates, encoder = build_model(params_shared, options, log)
    inps_aviliable = [item for item in inps_all if item is not None]
    
    log.log('Compiling Gradient Functions')
    f_log_probs = theano.function(inputs = inps_aviliable,
                                  outputs = cost,
                                  updates = updates,
                                  on_unused_input='ignore')
    
    
    grads = T.grad(cost, wrt = itemlist(params_shared))
    grads_clipped = []
    for item in grads:
        item_clipped = theano.gradient.grad_clip(item, -5.0, 5.0)
        grads_clipped.append(item_clipped)
    grads = grads_clipped
    
    log.log('Compiling Optimizers')
    lr = T.scalar(name = 'lr')
    f_grad_shared, f_update = eval(options['optimizer'])(lr, params_shared, grads, inps_aviliable ,cost, updates)
    
    first = True
    avg_cost = 0
    lRate = options['lRate']
    bestScore = 1e99
    bestCheckPoint = 0
    
    maxEpoch = 5
    
    for epoch_id in range(maxEpoch):
        dataset.batchShuffle()
    
        for batch_index in range(0, dataset.Subsets['trainSet'].n_batches()):
            Index = (epoch_id * dataset.Subsets['trainSet'].n_batches()) + batch_index + 1
            batchedData = dataset.get_Kth_Batch(batch_index)
            inps = batch2Inputs_new(batchedData, options)
            inps_avil = [item for item in inps if item is not None]
        
            costValue = f_grad_shared(*inps_avil)
            
            if first:
                avg_cost = costValue
                first = False
            else:
                avg_cost = 0.95 * avg_cost + 0.05 * costValue
            
            f_update(lRate)
        
            log.log('Batch %d: Cost %f, AvgCost %f'%(batch_index, costValue, avg_cost))
        
            if (options['sample'] and (Index % options['sampleFreq']) == 0):
                score = get_cost(dataset,'validNewSet', f_log_probs, Vocab, options)
                saveModel(params_shared, options, log, epoch_id, batch_index, None, None, None, None, 'check2')
                if score < bestScore:
                    log.log('Find a better model')
                    bestScore = score
                    bestCheckPoint = batch_index
                    saveModel(params_shared, options, log, epoch_id, batch_index, None, None, None, None, 'check2_best')
    
    saveModel(params_shared, options, log, epoch_id, batch_index, None, None, None, None, 'check2')
    log.log('Best Check Point: %d'%(bestCheckPoint))