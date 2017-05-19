import numpy as np
import theano.tensor as T
import lasagne
from snn1337.spiking_from_lasagne import spiking_from_lasagne
import pytest
from tests.data.mnist import load_dataset
import time

@pytest.fixture
def lasagne_dense_network():
    input_X = T.tensor4("X")
    input_shape = [None,1,28,28]
    target_y = T.vector("target Y integer",dtype='int32')

    net = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X, name='input')
    net = lasagne.layers.DenseLayer(net, num_units = 32, nonlinearity = lasagne.nonlinearities.rectify, name='hidden', b=None)
    net = lasagne.layers.DenseLayer(net, num_units = 16, nonlinearity = lasagne.nonlinearities.rectify, name='hidden1', b=None)
    net = lasagne.layers.DenseLayer(net, num_units = 10, nonlinearity = lasagne.nonlinearities.softmax, name='output', b=None)

    with np.load('tests/data/dense_weights.npz') as f:
        param_values = [f['arr_%d' % i] * 10 for i in range(len(f.files))]
        for i in param_values[-1]:
            for j in i:
                j = abs(j)

    lasagne.layers.set_all_param_values(net, param_values)
    return net

def test_create_dense(lasagne_dense_network):
    X_train = load_dataset()
    spiking_net = spiking_from_lasagne(lasagne_dense_network, [1.5])
    print(spiking_net.get_output_for(X_train[2], 90))
