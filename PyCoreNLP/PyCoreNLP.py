import requests, json

class PyCoreNLP(object):
    PROP_DEFAULT = {
        "annotators":"ssplit,tokenize,pos,lemma,parse",
        "outputFormat":"json"
    }
    PROP_TOKENIZE = {
        "annotators":"ssplit,tokenize,pos,lemma",
        "outputFormat":"json"
        }
    PROP_SSPLIT = {
        "annotators":"ssplit",
        "outputFormat":"json"
    }
    
    URL_DEFAULT = "http://localhost:9000"
    
    def __init__(self, url = URL_DEFAULT):
        if url[-1] == '/':
            url = url[:-1]
        self.url = url
        
    def annotate(self, text, mode = None):
        if mode != None:
            prop = eval('self.'+mode)
        else:
            prop = self.PROP_DEFAULT
            
        r = requests.post(url = self.url, params = {"properties":str(prop)}, data = text)
        output = json.loads(r.text, strict=False)
        return output
