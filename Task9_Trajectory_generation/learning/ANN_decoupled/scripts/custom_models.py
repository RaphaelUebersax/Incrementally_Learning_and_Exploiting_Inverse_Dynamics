# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:25:19 2020

@author: Amanhoud Walid
"""

import torch
from torch import nn
from torch.nn import functional as F
import yaml

# learning params
with open('./../param.yaml',"r") as learningStream:
	paramLoaded = yaml.safe_load(learningStream)

totalJoints = paramLoaded["totalJoints"]

class InverseDynamicModel(nn.Module):
    def __init__(self, nb_hidden_layers, nb_hidden_neurons, input_size=3*totalJoints,
                 output_size=1):
    
        super(InverseDynamicModel, self).__init__()
        
        self.nb_hidden_layers = nb_hidden_layers
        
        self.nb_hidden_neurons = nb_hidden_neurons
        
        self.input_size = input_size
        
        self. output_size = output_size
        
        self.fc_h1 = nn.Linear(self.input_size, self.nb_hidden_neurons)
        
        self.fc_hblock = nn.Sequential(*(nn.Sequential(nn.Linear(self.nb_hidden_neurons, 
                                                                 self.nb_hidden_neurons), 
                                                       nn.ReLU()) 
                                         for _ in range(self.nb_hidden_layers-1)))
    
        self.fc_out = nn.Linear(self.nb_hidden_neurons, self.output_size)
        
    def forward(self, x):
        
        y = F.relu(self.fc_h1(x))
        
        y = self.fc_hblock(y)
        
        y = self.fc_out(y)
        
        return y

