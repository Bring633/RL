#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 22:12:14 2022

@author: bring
"""

from Environment import Env
from QLearningobj import QLearningobj
from network_structure import Net
from tqdm import trange

import torch
import numpy as np
import matplotlib.pyplot as plt

NUM_SHEEP = 1
NUM_WOLF = 1
NN = 1
MAP_SIZE = (16,16)
action_space = 4
wolf_action_space = 8

alpha = 0.01
gamma = 0.9
epsilon = 0.9

#wolf_net = Net(MAP_SIZE[0]*MAP_SIZE[1],wolf_action_space)

def main(net_wolf):
    
    env = Env(MAP_SIZE)  
    
    sheep_list = []
    wolfs_list = [] 
    
    for i in range(NUM_WOLF):
        
        wolf = QLearningobj(alpha,gamma,epsilon,wolf_action_space,MAP_SIZE[0]*MAP_SIZE[1],True)
        wolf.init_qtable(NN,net_wolf)
        wolfs_list.append(wolf)
        
        
    for i in range(NUM_SHEEP):
        
        sheep = QLearningobj(alpha,gamma,epsilon,action_space,MAP_SIZE[0]*MAP_SIZE[1],True)
        net = Net(sheep.total_state,sheep.action_space)
        sheep.init_qtable(NN,net)
        sheep_list.append(sheep)

    env.main(sheep_list,wolfs_list,NN)
    
    mean_loss = np.array(wolfs_list[0].sum_loss).mean()
    
    if len(env.killed_sheep) == NUM_SHEEP:
        print(env.killed_sheep)
        return 1,mean_loss
    else:
        return 2,mean_loss

if __name__ ==  '__main__':
    
    wolf_success = 0
    sheep_success = 0
    
    loss = []
    
    for i in trange(5000):#trange有时候会打印不出来函数中的print
    
        wolf_net = torch.load(r'./wolf_net.pt')
    
        flags,sing_los = main(wolf_net)
        loss.append(sing_los)
        
        if flags == 1:
            wolf_success = wolf_success+1
        else:
            sheep_success = sheep_success+1
            
        torch.save(wolf_net,'./wolf_net.pt')
        
    plt.figure()
    plt.plot(loss)
        
    print('\n')
    print(wolf_success)
    print(sheep_success)