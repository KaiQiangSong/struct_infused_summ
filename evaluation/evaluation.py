from utility.utility import *
from Rouge import Rouge
from Bleu import Bleu

from data_processor.data_manager import *
from generation.generation import gen_sample

def get_cost(dataset, part, f_log_probs, Vocab, options):
    number = dataset.Subsets[part].n_batches()
    cost_total = 0
    for i in range(0, number):
        #print i
        batchedData = dataset.get_Kth_Batch(i, part)
        inps = batch2Inputs_new(batchedData, options)
        inps_avil = [item for item in inps if item is not None]
        #print batchedData
        cost = f_log_probs(*inps_avil)
        cost_total += cost
    return cost_total/number

class evalFile(object):
    def __init__(self, hyp_fileName, ref_fileName, metrics):
        self.hyp_fileName = hyp_fileName
        self.ref_fileName = ref_fileName
        self.metrics = metrics
        
    def eval(self):
        hypFile = open(self.hyp_fileName,'r')
        refFile = open(self.ref_fileName,'r')
        
        hypList = hypFile.readlines()
        refList = refFile.readlines()
        
        result = {}
        for metricName, metricSetting in self.metrics.items():
            Obj = eval(metricName)(metricSetting)
            result[metricName] = Obj.eval(hypList, refList)
            
        return result
    
class evalList(object):
    def __init__(self, metrics):
        self.metrics = metrics
    
    def eval(self, hypList, refList):
        result = {}
        for metricName, metricSetting in self.metrics.items():
            Obj = eval(metricName)(metricSetting)
            result[metricName] = Obj.eval(hypList, refList)
        return result
     