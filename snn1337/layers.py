#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from snn1337.connection import *
import numpy as np
from functools import reduce
import os
import errno
import json

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

class Layer(object):
    connections = None
    neurons = None

    def step(self):
        for conn in self.connections:
            conn.step()
        for i, neur in enumerate(self.neurons.reshape((self.neur_size))):
            neur.step()

    def restart(self):
        for neur in self.neurons:
            neur.restart()

    def get_nodes_description(self):
        return [{'id':neuron.get_id()} for neuron in self.neurons]

    def get_connections_description(self):
        return [{'source':conn.inp.get_id(), 'dest':conn.out.get_id()} for conn in self.connections]

class InputLayer(Layer):
    def __init__(self, nnet, shape):
        self.net = nnet
        self.shape = shape 
        self.neur_size = reduce(lambda res, x: res*x, self.shape, 1)
        self.neurons = np.ndarray(shape=self.shape, dtype=InputNeuron, buffer=np.array([InputNeuron(self.net, i, []) for i in np.arange(self.neur_size)]))
        self.connections = []

    def get_random_spike_train(freq, t_max):
        return sps.bernoulli.rvs(0.25*freq +0.5, size=t_max)
    
    def new_input(self, arg, t_max=1000, make_spike_train=get_fixed_frequency_spike_train):
        for i, f in enumerate(arg):
            for j, l in enumerate(f):
                for k, m in enumerate(l):
                    self.neurons[i][j][k].set_spike_train(get_fixed_frequency_spike_train(arg[i][j][k], t_max))

class Conv2DLayer(object):
    def __init__(self, nnet, input_layer, num_filters, filter_shape, weights, threshold=1.):
    # Формат весов: w[nk][h][i][j], где nk - фильтр, h - номер фильтра на предыдущем слое, i, j - координаты весов в фильтре
        self.net = nnet
        self.filter_shape = filter_shape
        self.weights = self.oldweights = weights.copy() 
        if(len(self.weights.shape) < 5):
            self.weights = self.weights.reshape(np.append(weights.shape, 1))
        
        self.shape = (num_filters, input_layer.shape[1]-filter_shape[0]+1, input_layer.shape[2]-filter_shape[1]+1)
        self.neur_size = reduce(lambda res, x: res*x, self.shape, 1)
        self.neurons = np.array([Neuron(self.net, i, threshold=threshold) for i in np.arange(self.neur_size)]).reshape(self.shape)
        
        self.connections = []
        
        for nk, kernel in enumerate(self.neurons):
            for i, row in enumerate(kernel):
                for j, neuron in enumerate(row):   # соединяем с предыдущим слоем
                    self.connections += [Connection(self.net, input_layer.neurons[l][i+p][j+q],neuron, self.weights[nk][l][p][q])\
                                         for l in np.arange(input_layer.shape[0]) for p in np.arange(filter_shape[0])\
                                         for q in np.arange(filter_shape[1])]

class SubSampling2DLayer(Layer):
    def __init__(self, nnet, input_layer, pool_size, threshold=1.):
        self.net = nnet
        self.pool_size = pool_size
        self.shape = input_layer.shape // np.append([1], pool_size)
        self.neur_size = reduce(lambda res, x: res*x, self.shape, 1)
        self.neurons = np.array([Neuron(self.net, i, threshold=threshold) for i in np.arange(self.neur_size)]).reshape(self.shape)
        
        self.conn_weight = 1 / (pool_size[0] * pool_size[1])
        
        self.connections = []
        
        for nk, kernel in enumerate(self.neurons):
            for i, row in enumerate(kernel):
                for j, neuron in enumerate(row):   
                    self.connections += [Connection(self.net,input_layer.neurons[l][i*pool_size[0]+p][j*pool_size[1]+q],neuron,\
                                                    [self.conn_weight])\
                                         for l in np.arange(input_layer.shape[0]) for p in np.arange(pool_size[0])\
                                         for q in np.arange(pool_size[1])]

# pool = ThreadPool(5)
class DenseLayer(Layer):
    #Формат весов: w[i][j],  где i - номер нейрона на предыдущем слое, j - номер нейрона на текущем слое
    def __init__(self, nnet, input_layer, num_units, weights, threshold=1.):
        self.net = nnet
        self.shape = [num_units]
        self.neur_size = num_units
        self.neurons = np.array([Neuron(self.net, i, threshold) for i in np.arange(self.neur_size)])
        self.weights = weights
        
        if(len(weights.shape) < 3):
            weights = weights.reshape(np.append(weights.shape, 1))
        
        self.connections = [Connection(self.net, input_neuron, output_neuron, weights[i][j])\
                            for i, input_neuron in enumerate(input_layer.neurons.reshape((input_layer.neur_size)))\
                            for j, output_neuron in enumerate(self.neurons)]


# pool1 = ThreadPool(4)
class NNet(object):
    def __init__(self, shape, threshold=1.):
        self.layers = [InputLayer(self, shape)]
        self.global_time = 0
        self.threshold = threshold

    def add_convolution(self, weights, threshold=-1):
        if(threshold == -1):
            threshold = self.threshold
        num_filters = weights.shape[0]
        filter_shape = weights.shape[2:4]
        self.layers.append(Conv2DLayer(self, self.layers[-1], num_filters, filter_shape, weights, threshold=threshold))
        
    def add_subsampling(self, pool_size, threshold=-1):
        if(threshold == -1):
            threshold = self.threshold
        self.layers.append(SubSampling2DLayer(self, self.layers[-1], pool_size, threshold=self.threshold))
        
    def add_dense(self, weights, threshold=-1):
        if(threshold == -1):
            threshold = self.threshold
        num_units = weights.shape[1]
        self.layers.append(DenseLayer(self, self.layers[-1], num_units, weights, threshold=threshold))
    
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

    def output_logs(self, path='data/'):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        nodes = []
        connections = []
        for layer in self.layers:
            pass
        graph = {'nodes': nodes, 'connections': connections}
        json.dump(graph, os.path.join(path, 'graph.json'))

