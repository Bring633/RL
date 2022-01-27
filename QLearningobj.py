# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:15:51 2022

@author: MSI-NB
"""


import numpy as np
import pandas as pd
import torch
from torch import FloatTensor

class QLearningobj():
    
    def __init__(self,alpha,gamma,epsilon,action_space,total_state,nn=False):
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.total_state = total_state
        self.action_space = action_space
        
        if nn:
            
            self.loss = torch.nn.MSELoss()
            self.sum_loss = []
        
        return None
    
    def init_qtable(self,nn,net=None):
        
        if not nn:
            #横坐标是回报，纵坐标是状态
            self.qtable = np.zeros((self.total_state,self.action_space))
        else:
            self.qtable = net
            self.optimizer = torch.optim.Adam(net.parameters(),0.01)
        return None
    
    def take_action(self,state,nn):
        
        if np.random.random()>self.epsilon:
            action = np.random.randint(self.action_space)
        
        if not nn:
            action = self.qtable[state,:].argmax()
        else:
            action = self.find_max_action_nn(state)               
        return action
    
    def find_max_action_nn(self,state):
        
        value,action = 0,0
        for i in range(self.action_space):
            now_value = self.qtable(FloatTensor(state).reshape(-1)).max()
            if now_value>value:
                value = now_value
                action = self.qtable(FloatTensor(state).reshape(-1)).argmax()

        return action
            
    def update_qtable(self,r,now_state,last_state,action,nn):
        
        if not nn:
            
            now_max_action = self.qtable[now_state,:].argmax()
            now_max_q = self.qtable[now_state,now_max_action]
            
            last_q = self.qtable[last_state,action]
            
            self.qtable[last_state,action] = now_max_q+self.alpha*(r+self.gamma*(now_max_q-last_q))
                
        else:
            
            now_max_action = self.find_max_action_nn(now_state)
            now_max_q = r+self.gamma*self.qtable(FloatTensor(now_state).reshape(-1))
            
            last_q = self.qtable(FloatTensor(last_state).reshape(-1))
            
            l = self.loss(now_max_q,last_q)
            l.backward()
            self.optimizer.step()
            self.sum_loss.append(l.cpu().item())
        
        return None
        
        
        
        