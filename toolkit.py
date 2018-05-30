import sys, os, re, json, codecs
import cPickle as Pickle

from PyCoreNLP.PyCoreNLP import PyCoreNLP

def saveToPKL(filename, data):
    with open(filename,'wb')as f:
        Pickle.dump(data, f)
    return 

def loadFromPKL(filename):
    with open(filename,'rb') as f:
        data = Pickle.load(f)
    return data

def Annotate(fileName):
    Annotator = PyCoreNLP()
    f_in = open(fileName,'r')
    f_out = open(fileName+'.json','w')
    f_p = open(fileName+".Ndocument","w")
    Index = 0
    for line in f_in:
        text = line.strip()
        anno = Annotator.annotate(text)
        json.dump(anno,f_out)
        f_out.write('\n')
        print Index
        Index += 1
        newText = ""
        first = True
        for token in anno["sentences"][0]["tokens"]:
            if first:
                first = False
                newText += token["originalText"].encode('ascii','ignore')
            else:
                newText += " " + token["originalText"].encode('ascii','ignore')
        
        f_p.write(newText)
        f_p.write('\n')
        
    return fileName

def extractFromConstituency(strParse):
    number = 0
    stack = []
    name = []
    edge = []
    tmp = ""
    
    for ch in strParse:
        if ch == '(':
            # New Node
            if (len(stack) > 0) and (name[stack[-1]] == ""):
                name[stack[-1]] = tmp.strip()
                tmp = ""
                
            stack.append(number)
            number +=1
            
            name.append("")
            edge.append([])
            if len(stack) > 1:
                edge[stack[-2]].append(stack[-1])
                
        elif ch == ')':
            if tmp != "":
                name[stack[-1]] = tmp.strip()
                tmp = ""
            done = stack.pop()
            if edge[done] == []:
                strs = name[done].split()
                name[done] = strs[0]
                name.append(strs[1])
                edge.append([])
                edge[done].append(number)
                number += 1
        else:
            tmp += ch
    
    data = {
        "number":number,
        "name":name,
        "edge":edge
        }
        
    return data

def BFS_ConstituencyParsing(data):
    queue = [0]
    depth = [0] * data["number"]
    pos = [""] * data["number"]
    parent = [-1] * data["number"]
    
    while (len(queue) > 0):
        current = queue.pop(0)
        for node in data["edge"][current]:
            depth[node] = depth[current] + 1
            if data["edge"][node] != []:
                queue.append(node)
            else:
                pos[node] = data["name"][current]
    pos = [pos[i] for i in range(data["number"]) if data["edge"][i] == []]
    depth = [depth[i] for i in range(data["number"]) if data["edge"][i] == []]
    return pos, depth
        
def extractFromDependency(data):
    number = len(data)
    incomeArc = [""] * number
    edge = [[]] * number
    token = [""] * number
    ROOT = -1
        
    for item in data:
        Id = item["dependent"] - 1
        father = item["governor"] - 1
        
        incomeArc[Id] = item["dep"]
        token[Id] = item["dependentGloss"]
        
        if (incomeArc[Id] != "ROOT"):
            if edge[father] != []:
                edge[father].append(Id)
            else:
                edge[father] = [Id]
        else:
            ROOT = Id
            
    Tree = {
        "number":number,
        "incomeArc":incomeArc,
        "edge":edge,
        "token":token,
        "ROOT":ROOT
        }
    
    return Tree

def BFS_Dependency(Tree):
    queue = [Tree["ROOT"]]
    path = [[]] * Tree["number"]
    depth = [0] * Tree["number"]
    parent = [-1] * Tree["number"]
    
    while (len(queue) > 0):
        current = queue.pop(0)
        for node in Tree["edge"][current]:
            depth[node] = depth[current] + 1
            parent[node] = current
            queue.append(node)
            
    outgoArc = [len(Tree["edge"][i]) for i in range(Tree["number"])]        
    return Tree["incomeArc"], outgoArc, depth, parent
    
featList = ["pos","depth_c","inArc","outArc","depth_d","parent"]

def extractFeatures(fileName, chunk_size = None):
    f = open(fileName,'r')
    Features = {}
    for feat in featList:
            Features[feat] = []
             
    Index = 0
    for l in f:
        data = json.loads(l)
        Tree_C = extractFromConstituency(data["sentences"][0]["parse"])
        pos, depth_c = BFS_ConstituencyParsing(Tree_C)
        Tree_D = extractFromDependency(data["sentences"][0]["basicDependencies"])
        
        inArc, outArc, depth_d, parent = BFS_Dependency(Tree_D)
        for feat in featList:
            Features[feat].append(eval(feat))

        Index += 1
        if (chunk_size != None) and (Index % chunk_size == 0):
            print Index, datetime.datetime.now()
            for feat in featList:
                f_out = open(fileName+'.'+feat+'.'+str(Index/chunk_size-1),'w')
                for item in Features[feat]:
                    print >> f_out, item
                Features[feat] = []
                
    if (chunk_size != None) and (Index % chunk_size != 0):
        print Index, datetime.datetime.now()
        for feat in featList:
            f_out = open(fileName+'.'+feat+'.'+str(Index/chunk_size),'w')
            for item in Features[feat]:
                print >> f_out, item
            Features[feat] = []
    
    if (chunk_size == None):
        for feat in featList:
            f_out = open(fileName+'.'+feat,'w')
            for item in Features[feat]:
                print >> f_out, item
            Features[feat] = []
    
    return fileName

def Mapping_POS(setFiles):
    w2i = {}
    i2w = []
    total = 0
    
    for fName in setFiles:
        f = open(fName,'r')
        for l in f:
            line = l.strip()
            data = eval(line)
            for path in data:
                for tag in path[:-1]:
                    if tag not in w2i:
                        i2w.append(tag)
                        w2i[tag] = total
                        total += 1
    map = {
        'w2i':w2i,
        'i2w':i2w
        }
    return map

def Mapping_inArc(setFiles):
    w2i = {}
    i2w = []
    total = 0
    for fName in setFiles:
        f = open(fName,'r')
        for l in f:
            line = l.strip()
            data = eval(line)
            for arc in data:
                if arc not in w2i:
                    i2w.append(arc)
                    w2i[arc] = total
                    total += 1
    map = {
        'w2i':w2i,
        'i2w':i2w
        }
    return map

def Compress_Pos(setFiles, map):
    for fName in setFiles:
        f_in = open(fName,'r')
        f_out = open(fName+'_map','w')
        for l in f_in:
            data = eval(l.strip())
            newData = []
            for item in data:
                newData.append(map["w2i"][item])
            print >> f_out, newData
    return 'Compress_POS'

def Compress_inArc(setFiles, map):
    for fName in setFiles:
        f_in = open(fName,'r')
        f_out = open(fName+'_map','w')
        for l in f_in:
            data = eval(l.strip())
            newData = []
            for item in data:
                newData.append(map["w2i"][item])
            print >> f_out, newData
    return 'Compress_inArc'

def Merge(setFiles, featList):
    for fName in setFiles:
        ff = {}
        fl = {}
        data = {}
        f_out = open(fName+'.feature','w')
        for feat in featList:
            ff[feat] = open(fName+'.'+feat,'r')
        Done = False
        while (True):
            for feat in featList:
                fl[feat] = ff[feat].readline()
                if not fl[feat]:
                    Done = True
                    break
                data[feat] = eval(fl[feat].strip())
            if (Done):
                break
            print >> f_out, data
    return True

rmList = ['.json','.json.pos','.json.pos_map','.json.parent','.json.outArc','.json.inArc','.json.inArc_map','.json.depth_c','.json.depth_d']

if __name__ == '__main__':
    if sys.argv[1] != '-f':
        fName = sys.argv[1]
        Annotate(fName)
        extractFeatures(fName+'.json')
    
        Pos_Map = loadFromPKL('Pos_Map.pkl')
        Arc_Map = loadFromPKL('Arc_Map.pkl')
    
        Compress_Pos([fName+'.json.pos'], Pos_Map)
        Compress_inArc([fName+'.json.inArc'], Arc_Map)
    
        mapped_featList = ["pos_map","depth_c","inArc_map","outArc","depth_d","parent"]
    
        Merge([fName+'.json'],mapped_featList)
    
        for ed in rmList:
            os.remove(fName+ed)
    
        os.rename(fName+'.json.feature', fName+'.feature')
    else:
        for fName_ in open(sys.argv[2],'r'):
            fName = fName_.strip()
            Annotate(fName)
            extractFeatures(fName+'.json')
    
            Pos_Map = loadFromPKL('Pos_Map.pkl')
            Arc_Map = loadFromPKL('Arc_Map.pkl')
    
            Compress_Pos([fName+'.json.pos'], Pos_Map)
            Compress_inArc([fName+'.json.inArc'], Arc_Map)
    
            mapped_featList = ["pos_map","depth_c","inArc_map","outArc","depth_d","parent"]
    
            Merge([fName+'.json'],mapped_featList)
    
            for ed in rmList:
                os.remove(fName+ed)
            
            os.rename(fName+'.json.feature',fName+'.feature')