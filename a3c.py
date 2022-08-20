import numpy as np
import gym
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer


class Model(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_count),
            nn.LogSoftmax()
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.linear(x)
        return self.actor(x), self.critic(x)

model = Model(2).cuda()

opt = optimizer.Adam(model.parameters(), lr=0.0017)

env = gym.make('CartPole-v1')
obs = env.reset()

log = []

gamma = 0.99

while True:
    a, c = model(torch.tensor(obs).cuda())
    log_prob = a
    action = torch.exp(log_prob).multinomial(1).detach().item()
    s1, reward, done, info = env.step(action)
    log.append((obs, action, reward, s1, log_prob, c))
    obs = s1
    if done: obs = env.reset()

    if done:
        sum_loss = torch.tensor(0.0).cuda()
        for i, data in enumerate(log):
            s0, action, reward, s1, log_prob, c0 = data

            c1 = log[i+1][5].detach()[0] if i < len(log)-1 else torch.tensor(0.0).cuda()

            td_target = torch.tensor(reward).cuda() + gamma * c1
            td_error = td_target - c0.detach()[0]

            value_loss = F.mse_loss(c0, td_target)
            policy_loss = -log_prob[action] * td_error

            total_loss = value_loss + policy_loss
            sum_loss += total_loss
            
        opt.zero_grad()
        sum_loss.backward()
        opt.step()
        
        print(len(log))
        
        log = []

    