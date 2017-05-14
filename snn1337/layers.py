from snn1337.connection import *
import numpy as np

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

def get_fixed_frequency_spike_train(frequency, t_max):
    actual_frequency = float(frequency)
    result = [1 if frequency > 0 else 0]
    for i in range(t_max - 1):
        if actual_frequency >= 1:
            result.append(1)
            actual_frequency -= int(actual_frequency)
        else:
            result.append(0)
        actual_frequency += frequency
    return result

from functools import reduce
class InputLayer(object):
    def __init__(self, nnet, shape):
        self.type = "InputLayer"
        self.net = nnet
        self.shape = shape 
        self.neur_size = reduce(lambda res, x: res*x, self.shape, 1)
        self.neurons = np.ndarray(shape=self.shape, dtype=InputNeuron, buffer=np.array([InputNeuron(self.net, []) for i in np.arange(self.neur_size)]))
            
    def get_random_spike_train(freq, t_max):
        return sps.bernoulli.rvs(0.25*freq +0.5, size=t_max)
    
    def new_input(self, arg, t_max=1000, make_spike_train=get_fixed_frequency_spike_train):
        for i, f in enumerate(arg):
            for j, l in enumerate(f):
                for k, m in enumerate(l):
                    self.neurons[i][j][k].set_spike_train(get_fixed_frequency_spike_train(arg[i][j][k], t_max))
    
    def step(self):
        for neur in self.neurons.reshape((self.neur_size)):
            neur.step()
            
    def learning(self):
        pass

class Conv2DLayer(object):
    # Формат весов: w[nk][h][i][j], где nk - фильтр, h - номер фильтра на предыдущем слое, i, j - координаты весов в фильтре
    def __init__(self, nnet, input_layer, num_filters, filter_shape, weights):
        self.type = "Conv2DLayer"
        self.net = nnet
        self.filter_shape = filter_shape
        self.weights = self.oldweights = weights.copy() 
        if(len(self.weights.shape) < 5):
            self.weights = self.weights.reshape(np.append(weights.shape, 1))
        
        self.shape = (num_filters, input_layer.shape[1]-filter_shape[0]+1, input_layer.shape[2]-filter_shape[1]+1)
        self.neur_size = reduce(lambda res, x: res*x, self.shape, 1)
        self.neurons = np.array([Neuron(self.net) for i in np.arange(self.neur_size)]).reshape(self.shape)
        
        self.connections = []
        
        for nk, kernel in enumerate(self.neurons):
            for i, row in enumerate(kernel):
                for j, neuron in enumerate(row):   # соединяем с предыдущим слоем
                    self.connections += [Connection(self.net, input_layer.neurons[l][i+p][j+q],neuron, self.weights[nk][l][p][q])\
                                         for l in np.arange(input_layer.shape[0]) for p in np.arange(filter_shape[0])\
                                         for q in np.arange(filter_shape[1])] 
    
    # возвращает веса слоя
    def get_weights(self):
        weights = []
        for conn in self.connections:
            weights.append(conn.weights)
        return weights
        
    # устанавливает веса слоя
    def set_weights(self, weights):
        assert(len(self.connections) == len(weights))
        for i in range(len(weights)):
            self.connections[i].weights = weights[i]
                    
    def restart(self):
        for i, neur in enumerate(self.neurons.reshape((self.neur_size))):
            neur.restart()
    
    def step(self):
        for conn in self.connections:
            conn.step()
        for i, neur in enumerate(self.neurons.reshape((self.neur_size))):
            neur.step()
            
    def learning(self):
        for conn in self.connections:
            conn.STDP_step()

class SubSampling2DLayer(object):
    def __init__(self, nnet, input_layer, pool_size):
        self.type = "SubSampling2DLayer"
        self.net = nnet
        self.pool_size = pool_size
        self.shape = input_layer.shape // np.append([1], pool_size)
        self.neur_size = reduce(lambda res, x: res*x, self.shape, 1)
        self.neurons = np.array([Neuron(self.net) for i in np.arange(self.neur_size)]).reshape(self.shape)
        
        self.conn_weight = 1 / (pool_size[0] * pool_size[1])
        
        self.connections = []
        
        for nk, kernel in enumerate(self.neurons):
            for i, row in enumerate(kernel):
                for j, neuron in enumerate(row):   
                    self.connections += [Connection(self.net,input_layer.neurons[l][i*pool_size[0]+p][j*pool_size[1]+q],neuron,\
                                                    [self.conn_weight])\
                                         for l in np.arange(input_layer.shape[0]) for p in np.arange(pool_size[0])\
                                         for q in np.arange(pool_size[1])] 
                    
    def restart(self):
        for i, neur in enumerate(self.neurons.reshape((self.neur_size))):
            neur.restart()
        
    def step(self):
        for conn in self.connections:
            conn.step()
        for neur in self.neurons.reshape((self.neur_size)):
            neur.step()
            
    def learning(self):
        pass

# pool = ThreadPool(5)
class DenseLayer(object):
    #Формат весов: w[i][j],  где i - номер нейрона на предыдущем слое, j - номер нейрона на текущем слое
    def __init__(self,nnet, input_layer, num_units, weights, threshold=1.):
        self.type = "DenseLayer"
        self.net = nnet
        self.shape = [num_units]
        self.neur_size = num_units
        self.neurons = np.array([Neuron(self.net, threshold) for i in np.arange(self.neur_size)])
        self.weights = weights
        
        if(len(weights.shape) < 3):
            weights = weights.reshape(np.append(weights.shape, 1))
        
        self.connections = [Connection(self.net, input_neuron, output_neuron, weights[i][j])\
                            for i, input_neuron in enumerate(input_layer.neurons.reshape((input_layer.neur_size)))\
                            for j, output_neuron in enumerate(self.neurons)]
    # возвращает веса слоя
    def get_weights(self):
        weights = []
        for conn in self.connections:
            weights.append(conn.weights)
        return weights
        
    # устанавливает веса слоя
    def set_weights(self, weights):
        assert(len(self.connections) == len(weights))
        for i in range(len(weights)):
            self.connections[i].weights = weights[i]
        
    def restart(self):
        for neur in self.neurons:
            neur.restart()
        
    def step(self):
        #for conn in self.connections:
            #conn.step()
        list(map(lambda x: x.step(), self.connections)) # POOL
        list(map(lambda x: x.step(), self.neurons)) # POOL
        #for neur in self.neurons:
            #neur.step()
            
    def learning(self):
        for conn in self.connections:
            conn.STDP_step()

# pool1 = ThreadPool(4)
class NNet(object):
    def __init__(self, shape, threshold=1.):
        self.layers = [InputLayer(self, shape)]
        self.global_time = 0
        self.threshold = threshold
    
    def add_convolution(self, weights):
        num_filters = weights.shape[0]
        filter_shape = weights.shape[2:4]
        self.layers.append(Conv2DLayer(self, self.layers[-1], num_filters, filter_shape, weights))
        
    def add_subsampling(self, pool_size):
        self.layers.append(SubSampling2DLayer(self, self.layers[-1], pool_size))
        
    def add_dense(self, weights):
        num_units = weights.shape[1]
        self.layers.append(DenseLayer(self, self.layers[-1], num_units, weights, threshold=self.threshold))
    
    def get_output_for(self, data, t_max):
        self.global_time = 0
        self.layers[0].new_input(data, t_max)
        for l in self.layers[1:]:
            l.restart()
        for t in np.arange(t_max):
            #for layer in self.layers:
                #layer.step()
            list(map(lambda x: x.step(), self.layers)) # POOL
            self.global_time += 1
        result = [neur.get_spikes() for neur in self.layers[-1].neurons.reshape((self.layers[-1].neur_size))]
        return result
    
    def classify(self, data, t_max):
        self.global_time = 0
        self.layers[0].new_input(data)
        for l in self.layers[1:]:
            l.restart()
        ans = []
        for t in np.arange(t_max):
            #for layer in self.layers:
                #layer.step()
            list(map(lambda x: x.step(), self.layers)) # POOL
            for i, neur in enumerate(self.layers[-1].neurons):
                if len(neur.get_spikes()) > 0:
                    ans.append(i)
            if(len(ans) > 0):
                return ans, t
            self.global_time += 1
        print('not_enough_time')
        
    def learning(self):
        for layer in self.layers:
            layer.learning()
     
     # забирает веса у слоёв с параметрами
    def get_all_params_values(self):
        weights = []
        for layer in self.layers:
            if layer.type not in ["InputLayer", "Conv2DLayer"]: 
                weights.append(layer.get_weights())
        return weights
    
    # устанавливает веса для слоёв с параметрами
    def set_all_params_values(self, weights):
        layer_with_weights_index = 0
        for layer in self.layers:
            if layer.type not in ["InputLayer", "Conv2DLayer"] and layer_with_weights_index < len(weights): 
                layer.set_weights(wights[layer_with_weights_index])
                layer_with_weights_index += 1

import lasagne

def spiking_from_lasagne(input_net, threshold):
    input_layers = lasagne.layers.get_all_layers(input_net)
    weights = lasagne.layers.get_all_param_values(input_net)
    spiking_net = NNet(input_layers[0].shape[-3:], threshold)
    convert_layers = {lasagne.layers.conv.Conv2DLayer : spiking_net.add_convolution,\
                      lasagne.layers.dense.DenseLayer : spiking_net.add_dense}
    
    #номер элемента в общем массиве весов, в котором хранятся веса текущего слоя
    i = 0
    
    for l in input_layers[1:]:
        if(type(l) == lasagne.layers.pool.Pool2DLayer or type(l) == lasagne.layers.pool.MaxPool2DLayer):
            spiking_net.add_subsampling(l.pool_size)
        else:
            convert_layers[type(l)](weights[i])
            i+=1

    return spiking_net
