import codecs, random, time

from utility.utility import *
from mylog.mylog import mylog

from data_manager import *

'''
    In this file, our goal is to build an interface of different dataset:
    Dataset:
        gigawords
        cnn
        dailymail
    
    dataset_loading_method:
        load
        build
    dataset_saving_address:
'''

class Subset:
    def __init__(self, name, Vocab, options, log):
        self.name = name
        self.log = log
        self.Vocab = Vocab
        self.options = options
        self.path = options['primary_dir'] + options[name]
        self.applied = False
        self.Data = {}
        
    
    def loadFromFile(self, fName, Vocab, options):
        documentName = fName + '.Ndocument'
        summaryName = fName + '.Nsummary'
        featureName = fName + '.dfeature'
        
        featList = options["featList"]
        df = codecs.open(documentName,'r', encoding = 'utf-8')
        sf = codecs.open(summaryName,'r', encoding = 'utf-8')
        ff = codecs.open(featureName,'r', encoding = 'utf-8')
        
        cf = codecs.open(fName + '.Error_List','w', encoding = 'utf-8')
        
        data = []
        anno = []
        feat = []
        
        Index = 0
        total = 0
        
        while (True):
            Index += 1
            dLine = df.readline()
            sLine = sf.readline()
            fLine = ff.readline()
            if (not dLine) or (not sLine) or (not fLine):
                break
            
            dLine = remove_digits(dLine.strip()).lower()
            sLine = remove_digits(sLine.strip()).lower()
            
            if (len(dLine.split()) < 1) or(len(sLine.split()) < 1):
                continue
            
            document = Sentence2ListOfIndex(dLine, Vocab, options, False)
            if len(document) > options['max_posi']:
                document = cutDown(document, options['max_posi'])
                
            summary = Sentence2ListOfIndex(sLine, Vocab, options, False)
            if len(summary) > options['max_len'] - 2:
                summary = cutDown(summary, options['max_len']-2)
            summary = [1] + summary + [1]
            
            feature = eval(fLine.strip())
            
            length = len(document)
            lengths = [len(fet) for fet in feature.values()]
            if (len(Set(lengths + [length])) > 1):
                total += 1
                print >> cf, 'DataError', total, Index, length, lengths
                print >> cf, dLine
                print >> cf, fLine
                #time.sleep(2)
                continue
            else:
                data.append(document)
                anno.append(summary)
                feat.append(feature)
        print len(data), len(anno), len(feat)
        return len(anno), (data, anno, feat)
   

    def sortByLength(self):
        self.log.log('Start sorting by length')
        data = self.Data['data'][0]
        anno = self.Data['data'][1]
        feat = self.Data['data'][2]
        number = len(anno)
        
        lengths =  [(len(data[Index]), Index) for Index in range(0,number)]
        sorted_lengths = sorted(lengths)
        sorted_Index = [d[1] for d in sorted_lengths]
        self.Data['data'] = ([data[sorted_Index[Index]] for Index in range(0, number)],
                             [anno[sorted_Index[Index]] for Index in range(0, number)],
                             [feat[sorted_Index[Index]] for Index in range(0, number)])
        self.log.log('Stop sorting by length')
        return
          

    def apply(self):
        if self.options['dataset_loading_method'] == 'build':
            self.log.log('Building Subset %s from original text documents'%(self.name))
            number, data = self.loadFromFile(self.path, self.Vocab, self.options)
            self.Data['number'] = number
            self.Data['data'] = data
            
            if ('train' in self.path) and (self.options['sortByLength']):
                self.sortByLength()

            self.log.log('Start Generating Batches')
            n_batches, batches = Generate_Batches(self.Data['data'], self.options['batch_size'])
            self.Data['n_batches'] = n_batches
            self.Data['batches'] = batches
            self.log.log('Stop Generating Batches')
            self.Data['index'] = range(n_batches)
            saveToPKL(self.path+'.data',self.Data)
            self.applied = True
            self.log.log('Finish Building Subset %s'%(self.name))
            
        elif self.options['dataset_loading_method'] == 'load':
            self.log.log('Loading Subset %s from PKL File'%(self.name))
            self.Data = loadFromPKL(self.path+'.data')
            self.applied = True
            self.log.log('Finish Loading Subset %s'%(self.name))
        return
    
    def number(self):
        if not self.applied:
            self.apply()
        return self.Data['number']
    
    def data(self):
        if not self.applied:
            self.apply()
        return self.Data['data']
    
    def n_batches(self):
        if not self.applied:
            self.apply()
        return self.Data['n_batches']
    
    def batches(self):
        if not self.applied:
            self.apply()
        return self.Data['batches']        
        
    def batchShuffle(self):
        if not self.applied:
            self.apply()
        random.shuffle(self.Data['index'])
        
    def get_Kth_Batch(self, K):
        if not self.applied:
            self.apply()
        return Get_Kth_Batch(self.Data['batches'],K, self.Data['index'])
    
    def get_first_K_Batches(self, K):
        if not self.applied:
            self.apply()
        batchedData = ([],[],[])
        K = min(K, self.Data['n_batches'])
        
        for Index in range(0, K):
            batchedData = batch_merge(batchedData, self.get_Kth_Batch(Index, part))
        
        return batchedData
    
    def get_K_random_instances(self, K):
        if not self.applied:
            self.apply()
        data = self.Data['data']
        n_data = self.Data['number']
        randomIndex = range(0, n_data)
        random.shuffle(randomIndex)
        batchedData = ([],[],[])
        K = min(n_data, K)
        for Index in range(0, K):
            data_Index = get_Kth_Instance(randomIndex[Index], data)
            batchedData = batch_merge(batchedData, data_Index)
        return batchedData
    
    def get_first_K_instances(self, K):
        if not self.applied:
            self.apply()
        data = self.Data['data']
        n_data = self.Data['number']
        batchedData = ([],[],[])
        K = min(n_data, K)
        for Index in range(0, K):
            data_Index = get_Kth_Instance(Index, data)
            batchedData = batch_merge(batchedData, data_Index)
        return batchedData
    
    
            

class data_loader:
    def __init__(self, Vocab, options, log):
        self.name = options['datasetName']
        self.log = log
        self.Subsets = {}
        for name in options['subsets']:
            self.Subsets[name] = Subset(name, Vocab, options, log)
            
    def batchShuffle(self, part = 'trainSet'):
        self.Subsets[part].batchShuffle()
        return
    
    def get_Kth_Batch(self, K, part = 'trainSet'):
        return self.Subsets[part].get_Kth_Batch(K)
    
    def get_first_K_Batches(self, K, part = 'trainSet'):
        return self.Subsets[part].get_first_K_Batches(K)
    
    def get_K_random_instances(self, K, part = 'trainSet'):
        return self.Subsets[part].get_K_random_instances(K)
    
    def get_first_K_instances(self, K, part = 'trainSet'):
        return self.Subsets[part].get_first_K_instances(K)
    
        