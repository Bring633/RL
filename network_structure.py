# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:01:16 2022

@author: MSI-NB
"""


import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self,input_size,output_size):

        super(Net,self).__init__()        

        self.layer1 = nn.Linear(input_size,32)
        self.layer2 = nn.Linear(32,16)
        self.layer3 = nn.Linear(16,output_size)
        
    def forward(self,X,):

        layer1 = F.relu(self.layer1(X))
        layer2 = F.relu(self.layer2(layer1))
        
        return self.layer3(layer2)
    
class Net_wolf(nn.Module):
    
    def __init__(self,input_size,output_size):

        super(Net_wolf,self).__init__()        

        self.layer1 = nn.Linear(input_size,32)
        self.layer2 = nn.Linear(32,128)
        self.layer3 = nn.Linear(128,output_size)
        
    def forward(self,X,):

        layer1 = F.relu(self.layer1(X))
        layer2 = F.relu(self.layer2(layer1))
        
        return self.layer3(layer2)
        

