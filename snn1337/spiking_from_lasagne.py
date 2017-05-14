
from snn1337.layers import *
import lasagne


def spiking_from_lasagne(input_net, threshold=[1.]):
    input_layers = lasagne.layers.get_all_layers(input_net)
    weights = lasagne.layers.get_all_param_values(input_net)
    spiking_net = NNet(input_layers[0].shape[-3:], threshold)
    if(len(threshold) == 1):
        threshold = np.ones(len(input_layers)-1) * threshold
    convert_layers = {lasagne.layers.conv.Conv2DLayer : spiking_net.add_convolution,\
                      lasagne.layers.dense.DenseLayer : spiking_net.add_dense}

    i = 0
    
    for l in input_layers[1:]:
        if(type(l) == lasagne.layers.pool.Pool2DLayer or type(l) == lasagne.layers.pool.MaxPool2DLayer):
            spiking_net.add_subsampling(l.pool_size)
        else:

            convert_layers[type(l)](weights[i], threshold=threshold[i])
            i+=1
            

    return spiking_net
