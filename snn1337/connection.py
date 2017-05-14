#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from snn1337.neuron import *
import numpy as np

class InputNeuron(object):

    def __init__(self, nnet, spike_train):
        self.spike_train = spike_train
        self.output_spikes_times = []
        self.net = nnet

    def set_spike_train(self, spike_train):
        self.spike_train = spike_train
        self.output_spikes_times = []

    def step(self):
        if self.spike_train[self.net.global_time] == 1:
            self.output_spikes_times.append(self.net.global_time)

    def get_spikes(self):
        return self.output_spikes_times


class Connection(object):
    def __init__(self, nnet, input_neuron, output_neuron, weights=[1], delays=[1],
                etta=4, tau_min=6, tau_plus=3, sigma=0.1):  # weights and delays are scaled
        self.weights = weights
        self.delays = delays
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.net = nnet
        self.last_conducted_spike_index = 0

        #for STDP
        self.etta = etta
        self.tau_min = tau_min
        self.tau_plus = tau_plus
        self.sigma = sigma

    def step(self):
        spikes = self.input_neuron.get_spikes()
        for i in range(self.last_conducted_spike_index, len(spikes)):
            spike_time = spikes[i]
            for j in range(len(self.weights)):
                if spike_time + self.delays[j] == self.net.global_time:
                    self.last_conducted_spike_index += 1
                    self.output_neuron.receive_spike(self.weights[j])
    
    ############################
    def STDP_step(self):
        output_spikes = np.array(self.output_neuron.get_spikes())
        input_spikes = np.array(self.input_neuron.get_spikes())
        weights = np.array(self.weights)
        w_max = weights.max()
        w_min = weights.min()
        out_spikes_num = len(output_spikes)
        
        if out_spikes_num <= 0:
            return
        
        tau_post_before = 0
        tau_post_after = output_spikes[0]
        out_ind = 0
        
        etta = self.etta
        tau_min = self.tau_min
        tau_plus = self.tau_plus
        sigma = self.sigma
        
        for j in range(len(weights)):
            for i, tau_pre in enumerate(input_spikes):
                #####################
                if (tau_pre > tau_post_after):
                    tau_post_before = tau_post_after
                    if (out_ind < out_spikes_num - 1):
                        out_ind += 1
                        tau_post_after = output_spikes[out_ind]
                    else:
                        tau_post_after = 0
                
                if (0 <= tau_pre-tau_post_after <= tau_min):
                    delta_w = -1.0*etta*(tau_min-(tau_pre-tau_post_after))
                elif (-1.*tau_plus <= tau_pre-tau_post_after <= 0):
                    delta_w = etta*(tau_plus+(tau_pre-tau_post_after))
                else:
                    delta_w = 0
                    
                if (0 <= tau_pre-tau_post_before <=tau_min):
                    delta_w -= etta*(tau_min-(tau_pre-tau_post_before))
                elif (-1.*tau_plus <= tau_pre-tau_post_before <= 0):
                    delta_w += etta*(tau_plus+(tau_pre-tau_post_before))

                w_old = self.weights[j]
                if (delta_w > 0):
                    self.weights[j] = w_old + sigma*delta_w*(w_max - w_old)
                else:
                    self.weights[j] = w_old + sigma*delta_w*(w_old - w_min)
