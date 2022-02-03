#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:44:30 2022

@author: bring
"""

import numpy as np
from QLearningobj import QLearningobj
from network_structure import Net

import matplotlib.pyplot as plt


class Env():
    
    def __init__(self,map_size):
        
        self.map_size = map_size
        self.killed_sheep = []
        
        return None
    
    def get_agent(self,wolfs,sheep):
        
        self.wolfs_list = wolfs
        self.sheep_list = sheep
        
        return None
    
    def init_environment(self):
        
        self.env = np.zeros(self.map_size)
        
        return None
    
    def init_state(self):
        
        self.state = self.env
        
        for i in self.wolfs_list:
            self.state[i.loc[0],i.loc[1]] = 2
            
        for i in self.sheep_list:
            self.state[i.loc[0],i.loc[1]] = 1
            
        return None

    def random_init_place(self):
        
        for i in self.wolfs_list:
            i.last_time_dis = 0
            i.loc = (np.random.randint(self.map_size[0]),np.random.randint(self.map_size[1]))
        for j in self.sheep_list:
            j.last_time_dis = 0
            j.loc = (np.random.randint(self.map_size[0]),np.random.randint(self.map_size[1]))
            
        return None
    
    def dis(self,agent,target):
        
        distance = 0
        
        for i in range(len(agent.loc)):
            distance = distance+(agent.loc[i]-target.loc[i])**2
            
        return distance
    
    def reward(self,action,agent,target):
        
        sum_dis = 0
        reward = 0
        
        if target == 1:
            
            for i in self.sheep_list:
                dis = self.dis(agent,i)
                sum_dis = sum_dis+dis
            
            if agent.last_time_dis>sum_dis:
                reward = reward
            else:
                reward = reward-0.01
            for i in range(len(self.sheep_list)):
                if self.sheep_list[i].loc == agent.loc:
                    reward = reward+10
        else:
            
            for i in self.wolfs_list:
                dis = self.dis(agent,i)
                sum_dis = sum_dis+dis
            if agent.last_time_dis<sum_dis:
                reward = reward+1
        return reward
    
    def get_next_state(self,action,agent,type_):
        
        x,y = agent.loc
        
        now_state = self.state
        now_state[x,y] = 0
        
        if action == 0:
            if y!=self.map_size[1]-1:
                y = y+1
        elif action == 1:
            if y!=0:
                y = y-1
        elif action == 2:
            if x!=self.map_size[0]-1:
                x = x+1
        
        elif action == 3:
            if x!=0:
                x = x-1

        else:
            if action == 4:
                if y==self.map_size[1]-1:
                    pass
                elif y==self.map_size[1]-2:
                    y = y+1
                else:
                    y = y+2
                        
            elif action == 5:
                if y==0:
                    pass
                elif y==1:
                    y = y-1
                else:
                    y = y-2
                        
            elif action == 6:
                if x==self.map_size[1]-1:
                    pass
                elif x==self.map_size[1]-2:
                    x = x+1
                else:
                    x = x+2
                
            else:
                if x==0:
                    pass
                elif x==1:
                    x = x-1
                else:
                    x = x-2
                        
        if type_ == 1:
            now_state[x,y] = 1
        else:
            try:
                now_state[x,y] = 2
            except:
                print(x,y)
        
        agent.loc = (x,y)
        
        return now_state
    
    def judge_sheep(self,agent):
        index_ = []
        
        for i in range(len(self.sheep_list)):
            if self.sheep_list[i].loc == agent.loc:
                index_.append(i)
                #print("killed")
        
        left_index = set(range(len(self.sheep_list)))-set(index_)
        left_ = []
        list_ = []
        
        for i in left_index:
            
            left_.append(self.sheep_list[i])
            
        for i in index_:
            
            list_.append(self.sheep_list[i])
        
        self.sheep_list = left_
        
        return list_
    
    def main(self,wolfs,sheep,nn):
        
        self.get_agent(wolfs,sheep)
        self.init_environment()
        self.random_init_place()
        self.init_state()
            
        step = 0
            
        while(True):
            
            if step>50:
                
                break
            
            for wolf in self.wolfs_list:
                    
                action = wolf.take_action(self.state,nn,2)
                now_state = self.get_next_state(action,wolf,2)
                r = self.reward(action,wolf,1)
                    
                wolf.update_qtable(r,now_state,self.state,action,2,nn)
                    
                list_ = self.judge_sheep(wolf)
                self.killed_sheep = self.killed_sheep + list_
                    
                self.state = now_state
                    
                    #print('wolf at {}'.format(wolf.loc))
                
                if len(self.sheep_list)==0:
                    
                    return None
                
            for sheep in self.sheep_list:
                    
                action = sheep.take_action(self.state,nn,1)
                r = self.reward(action,sheep,2)
                    
                now_state = self.get_next_state(action,sheep,1)
                sheep.update_qtable(r,now_state,self.state,action,1,nn)
                    
                self.state = now_state
                    
                #print('sheep at {}'.format(sheep.loc))
                
            step = step + 1
                
        return None
    
    def main_without_update(self,wolfs,sheep,nn):
        
        self.get_agent(wolfs,sheep)
        self.init_environment()
        self.random_init_place()
        self.init_state()
            
        step = 0
            
        while(True):
            
            if step>50:
                
                break
            
            for wolf in self.wolfs_list:
                    
                action = wolf.take_action(self.state,nn,2)
                now_state = self.get_next_state(action,wolf,2)
                #r = self.reward(action,wolf,1)
                    
                #wolf.update_qtable(r,now_state,self.state,action,2,nn)
                    
                list_ = self.judge_sheep(wolf)
                self.killed_sheep = self.killed_sheep + list_
                    
                self.state = now_state
                    
                    #print('wolf at {}'.format(wolf.loc))
                
                if len(self.sheep_list)==0:
                    
                    return None
                
            for sheep in self.sheep_list:
                    
                action = sheep.take_action(self.state,nn,1)
                r = self.reward(action,sheep,2)
                    
                now_state = self.get_next_state(action,sheep,1)
                #sheep.update_qtable(r,now_state,self.state,action,1,nn)
                    
                self.state = now_state
                    
                #print('sheep at {}'.format(sheep.loc))
                
            step = step + 1
                
        return None
                
                
                
                
                
                
        
        