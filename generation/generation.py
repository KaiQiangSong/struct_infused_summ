import theano.tensor as T
import numpy as np

from options_loader import *

from mylog.mylog import mylog
from utility.utility import *

from Layers.Layers import *
from data_processor.data_manager import *
from data_processor.data_loader import data_loader

from build_model.build_model import build_model, build_sampler

#Search Methods
from bfs_beam import *
from bfs_beam_75 import *

def gen_sample(encoder, encoderInputs, decoder, otherInputs, Vocab, options, log):
    return eval(options['generation_method']+'_search')(encoder, encoderInputs, decoder, otherInputs, Vocab, options, log)