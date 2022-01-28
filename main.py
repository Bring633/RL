#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 22:12:14 2022

@author: bring
"""

from Environment import Env
from QLearningobj import QLearningobj
from network_structure import Net,Net_wolf
from tqdm import trange

import torch
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Process,Pool

NUM_SHEEP = 1
NUM_WOLF = 1
NN = 1
MAP_SIZE = (32,32)
action_space = 4
wolf_action_space = 8

alpha = 0.01
gamma = 0.9
epsilon = 0.9
    


def main_train(net_wolf):
    
    env = Env(MAP_SIZE)  
    
    sheep_list = []
    wolfs_list = [] 
    
    for i in range(NUM_WOLF):
        
        wolf = QLearningobj(alpha,gamma,epsilon,action_space,wolf_action_space,MAP_SIZE[0]*MAP_SIZE[1],True)
        wolf.init_qtable(NN,net_wolf)
        wolfs_list.append(wolf)
        
        
    for i in range(NUM_SHEEP):
        
        sheep = QLearningobj(alpha,gamma,epsilon,action_space,wolf_action_space,MAP_SIZE[0]*MAP_SIZE[1],True)
        net = Net(sheep.total_state,sheep.action_space)
        sheep.init_qtable(NN,net)
        sheep_list.append(sheep)

    env.main(sheep_list,wolfs_list,NN)
    
    mean_loss = np.array(wolfs_list[0].sum_loss).mean()
    
    if len(env.killed_sheep) == NUM_SHEEP:
        return 1,mean_loss
    else:
        return 2,mean_loss
    
def main_exec(net_wolf):
    
    env = Env(MAP_SIZE)  
    
    sheep_list = []
    wolfs_list = [] 
    
    for i in range(NUM_WOLF):
        
        wolf = QLearningobj(alpha,gamma,epsilon,action_space,wolf_action_space,MAP_SIZE[0]*MAP_SIZE[1],True)
        wolf.init_qtable(NN,net_wolf)
        wolfs_list.append(wolf)
        
        
    for i in range(NUM_SHEEP):
        
        sheep = QLearningobj(alpha,gamma,epsilon,action_space,wolf_action_space,MAP_SIZE[0]*MAP_SIZE[1],True)
        net = Net(sheep.total_state,sheep.action_space)
        sheep.init_qtable(NN,net)
        sheep_list.append(sheep)

    env.main_without_update(sheep_list,wolfs_list,NN)
    
    if len(env.killed_sheep) == NUM_SHEEP:
        return 1,
    else:
        return 2,

def experiments(exp,pretrain = None):
    
    wolf_success = 0
    sheep_success = 0
    
    if pretrain != None:
        
        wolf_net = Net_wolf(MAP_SIZE[0]*MAP_SIZE[1],wolf_action_space)
        
        for i in range(pretrain):
        
            flags,sing_los = main_train(wolf_net)
            
            if flags == 1:
                wolf_success = wolf_success+1
            else:
                sheep_success = sheep_success+1
                
            torch.save(wolf_net,r'./wolf_net_{}'.format(pretrain))
              
        
        for i in range(exp):
            
            wolf_net = torch.load(r'./wolf_net_{}.pt'.format(pretrain))
            
            flags = main_exec(wolf_net)
        
            if flags == 1:
                wolf_success = wolf_success+1
            else:
                sheep_success = sheep_success+1
 
        
    else:
        
        wolf_net = Net_wolf(MAP_SIZE[0]*MAP_SIZE[1],wolf_action_space)
        
        for i in range(exp):
            
            flags = main_exec(wolf_net)
        
            if flags == 1:
                wolf_success = wolf_success+1
            else:
                sheep_success = sheep_success+1
                
    return (exp,pretrain,wolf_success,sheep_success)


if __name__ ==  '__main__':
    
    exp_list = [(100,None),(100,50),(100,100),(100,300),(100,500),(100,1000)]
    data = []
    p_list = []
    pool = Pool(processes = 10)
        
    for i in exp_list:
        p_list.append(pool.apply_async(experiments,(i)))
    
    pool.close()
    pool.join()
    
    for i in p_list:
        data.append(i.get())
    
    