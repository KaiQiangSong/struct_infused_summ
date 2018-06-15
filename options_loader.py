from utility.utility import *
from mylog.mylog import mylog
from collections import OrderedDict

'''
    The Goal of this file is to setup all the options of the experiment
    and prepare for experiments
'''

optionsFrame = {
    'vocabulary': 'settings/vocabulary.json',
    'dataType':'settings/dType.json',
    'training':'settings/training.json',
    'test':'settings/test.json',
    'dataset':'settings/my_test_settings.json',
    'network':'settings/network_struct_edge.json',
    'saveLoad':'settings/saveLoad.json',
    'earlyStop':'settings/earlyStop.json',
    'evaluation':'settings/evaluation.json',
    'structure':'settings/structure_embedding.json'
    }

def optionsLoader(log, disp = False, reload = None):
    if reload == None:
        log.log('Start Loading Options')
        options = OrderedDict()
        for k,v in optionsFrame.items():
            log.log(k)
            option = loadFromJson(v)
            for kk,vv in option.items():
                if not kk in options:
                    options[kk] = vv
                else:
                    log.log('Options Error: conflict with ' + kk)
        log.log('Stop Loading Options')
    else:
        log.log('Start Reloading Options')
        options = loadFromJson(reload)
        log.log('Stop Reloading Options')
    if disp:
        print 'Options:'
        for kk,vv in options.items():
            print '\t',kk,':',vv
    return options
