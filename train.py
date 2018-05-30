import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


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

srng = RandomStreams(seed = 19940505)


if __name__ == '__main__':
    log = mylog()
    
    options = optionsLoader(log, True)
    if options['reload'] == True:
        options = optionsLoader(log, True, options['model_path']+options['reload_options'])
    else:
        options['start_epoch'] = 0
    
    options["training"] = True
    options["test"] = False
    
    Vocab_Giga = loadFromPKL('../../dataset/gigaword_eng_5/giga_new.Vocab')
    log.log(str(Vocab_Giga.full_size)+', '+str(Vocab_Giga.n_in) + ', ' + str(Vocab_Giga.n_out))
    
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
    
    '''
    for part in dataset.Subsets:
        print part, dataset.Subsets[part].number()
    '''
    '''
    if options['dataset_loading_method'] == 'load':
        log.log('Start Loading Dataset')
        dataset = loadFromPKL(options['dataset_saving_address'])
        log.log('Stop Building Dataset')
    else: 
        log.log('Start Building Dataset')
        dataset = data_loader(Vocab, options, log)
        saveToPKL(options['dataset_saving_address'], dataset)
        log.log('Stop ')
    '''

    '''
    for Id in range(dataset.n_train):
        data = (dataset.train[0][Id], dataset.train[1][Id], dataset.train[2][Id])
        length = len(data[0])
        lengths = [len(feat) for feat in data[2].values()]
        if (len(Set(lengths + [length])) > 1):
            log.log('DataError '+str(Id)+' '+str(length)+' '+str(lengths))                    
    '''
    params = init_params(options, Vocab, I2Es, log)
    
    if options['reload'] == True:
        log.log('Start reloading Parameters.')
        params = load_params(options['model_path']+options['reload_model'], params)
        log.log('Finish reloading Parameters.')
    
    params_shared = init_params_shared(params)
    
    inps_all, dist, cost, updates, encoder = build_model(params_shared, options, log)
    inps_aviliable = [item for item in inps_all if item is not None]
    
    
    #inps_dec, decoder = build_sampler(params_shared, options)
    
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
    
    if options['start_epoch'] == 0:
        log.log('Start Training')
    else:
        log.log('Continue Training at epoch %d '%(options['start_epoch']))
    
    first = True
    avg_cost = 0
    
    if options['reload']:
        bestScore = options['bestScore']
        batch_count = options['batch_count']
        lRate = options['lRate']
        rate_count = options['rate_count']
    else:
        bestScore = 1e99
        batch_count = 0
        lRate = options['LearningRate']
        rate_count = 0
        
    flag = False
    
    for epoch_index in range(options['start_epoch'],options['max_epochs']):
        log.log('Epoch %d'%(epoch_index))
        dataset.batchShuffle()
        
        for batch_index in range(0, dataset.Subsets['trainSet'].n_batches()):
            
            Index = (epoch_index * dataset.Subsets['trainSet'].n_batches()) + batch_index + 1
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
            
            log.log('Epoch %d, Batch %d: Cost %f, AvgCost %f'%(epoch_index, batch_index, costValue, avg_cost))
            
            if (options['sample'] and (Index >= options['sampleMin']) and (Index % options['sampleFreq']) == 0):
                log.log("This is a check point")
                if options['earlyStop']:
                    log.log('Do early Stopping')
                    if options['earlyStop_method'] == 'valid_err':
                        score = get_cost(dataset,'validSet', f_log_probs, Vocab, options)
                        
                    log.log('Score is %f, bestScore is %f'%(score,bestScore))
                    log.log('Current learning Rate is %f'%(lRate))
                    
                    if score < bestScore:
                        log.log('Find a better model')
                        bestScore = score
                        batch_count = 0
                        rate_count = 0
                        
                        log.log('Update Best Model')
                        
                        saveModel(params_shared, options, log, epoch_index, batch_index, bestScore, batch_count, lRate, rate_count, 'best')
                        saveModel(params_shared, options, log, epoch_index, batch_index, bestScore, batch_count, lRate, rate_count, 'best_epoch')
                        saveModel(params_shared, options, log, epoch_index, batch_index, bestScore, batch_count, lRate, rate_count, 'best_epoch_batch')
                        
                    else:
                        batch_count  += options['sampleFreq']
                        rate_count += options['sampleFreq']
                        
                        log.log('batch_count = %d'%(batch_count))
                        log.log('rate_count = %d'%(rate_count))
                        
                        if (batch_count >= options['earlyStop_bound']):
                            log.log('Early Stopping')
                            flag = True
                            break
                        
                        if (rate_count >= options['rate_bound']):
                            log.log('Half Learning Rate')
                            lRate *= 0.5
                            rate_count = 0
                
                saveModel(params_shared, options, log, epoch_index, batch_index, bestScore, batch_count, lRate, rate_count, 'check_epoch_batch')
                
            if flag:
                break
        if flag:
            break
            
        if options['SaveEachEpoch']:
            saveModel(params_shared, options, log, epoch_index, batch_index, bestScore, batch_count, lRate, rate_count, 'epoch')
