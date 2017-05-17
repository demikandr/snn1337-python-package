import numpy as np
import theano.tensor as T
import lasagne
from snn1337.spiking_from_lasagne import spiking_from_lasagne
from mnist import load_dataset
import pytest

def prepare_lasagne_network():
    input_X = T.tensor4("X")
    input_shape = [None,1,28,28]
    target_y = T.vector("target Y integer",dtype='int32')

    input_layer = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X)
    dense_1 = lasagne.layers.DenseLayer(input_layer, num_units=128,
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                       name='dense_1', b=None)
    dense_2 = lasagne.layers.DenseLayer(dense_1, num_units=256,
                                        nonlinearity=lasagne.nonlinearities.sigmoid,
                                       name='dense_2', b=None)
    dense_3 = lasagne.layers.DenseLayer(dense_2, num_units=64,
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                       name='dense_2', b=None)
    dense_output = lasagne.layers.DenseLayer(dense_3, num_units = 10,
                                            nonlinearity = lasagne.nonlinearities.softmax,
                                            name='output', b=None)

    with np.load('dense_weights.npz') as f:
        param_values = [f['arr_%d' % i] * 10 for i in range(len(f.files))]
        for i in param_values[-1]:
            for j in i:
                j = abs(j)
        for i in param_values:
            for j in i:
                for k, n in enumerate(j):
                    if (n < 0.000001):
                        j[k] = 0.0

    lasagne.layers.set_all_param_values(dense_output, param_values)

def test_dense():
    X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
    lasagne_net = prepare_lasagne_network()
    spiking_net = spiking_from_lasagne(lasagne_net, 1.375)
    output = spiking_net.get_output_for(X_train[1], 39)