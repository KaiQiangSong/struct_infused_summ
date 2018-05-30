import sys, os, re, json
import cPickle as Pickle
from shutil import copyfile
from collections import OrderedDict


if __name__ == '__main__':
    
    datasetSettings = {
        "primary_dir":"",
        "datasetName":sys.argv[2],
        "subsets":[],
        "dataset_loading_method":"build",
        }
    
    datasetSettings = OrderedDict(datasetSettings)
    
    for fName_ in open(sys.argv[1],'r'):
        
        a = fName_.strip()
        b = a+'.feature'
        copyfile(a+'.Ndocument', a+'.Nsummary')
        copyfile(b, a+'.dfeature')
        
        datasetSettings[a]=a
        datasetSettings["subsets"].append(a)
    print datasetSettings
    f = open(datasetSettings['datasetName']+'.json','w')
    json.dump(datasetSettings, f)
    f.write('\n')