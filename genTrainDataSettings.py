import sys, os, re, json
import cPickle as Pickle
from shutil import copyfile
from collections import OrderedDict


if __name__ == '__main__':
    
    datasetSettings = {
        "primary_dir":"",
        "datasetName":sys.argv[3],
        "subsets":['trainSet','validSet'],
        "trainSet":sys.argv[1],
        "validSet":sys.argv[2],
        "dataset_loading_method":"build",
        }
    
    datasetSettings = OrderedDict(datasetSettings)
    
    print datasetSettings
    f = open(datasetSettings['datasetName']+'.json','w')
    json.dump(datasetSettings, f)
    f.write('\n')