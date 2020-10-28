# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:09:12 2020

@author: lvjf

RL brain for JSSP

LSTM for memory, thus no need for store transitions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import normalized_columns_initializer, weights_init


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32*553, 256)

        num_outputs = action_space
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 32*553)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
    
    def choose_action(self,inputs,action_dim):
        s, (hx, cx) = inputs
        value, logit, (hx, cx) = self.forward((s.unsqueeze(0),(hx, cx)))
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        
        #action = prob.multinomial(num_samples=action_dim).detach()
        action=[]
        for i in range(action_dim):
            action.append(prob.multinomial(num_samples=1).detach()[0])
        action = torch.from_numpy(np.array(action,dtype = np.int64).reshape(1,133))
        return action, log_prob, entropy, value