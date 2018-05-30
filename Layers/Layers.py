#Embedding Layers
from embedding_layer import *
from multiple_embedding_layer import *
from position_embedding_layer import *

#Basic Layers
from indexing_layer import *
from feedforward_layer import *
from lstm_layer import *
from activation_layer import *
from attention_layer import *
from attention_struct_layer import *

#Intermediate Level Layers
from uni_lstm_layer import *
from bi_lstm_layer import * 

# Model 0 Baseline
from baseline_decoder_layer import *

# Model 1 Attention
from attention_decoder_layer import *
# Model 2 Switcher Pointer
# Model 3 Struct_One v1
# Model 4 Struct_One v2 (New)
from switcher_pointer_decoder_layer import *

# Model 5 Structure Kaiqiang v1
from struct_node_decoder_layer import *
# Model 6 Structure Kaiqiang v2 (New)
from struct_edge_decoder_layer import *


def get_layer(name):
    return (eval(name+'_init'),eval(name+'_calc'))
