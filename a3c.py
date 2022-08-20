import numpy as np
import gym
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

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

class Environment:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.model = Model(2).cuda()
        self.obs = self.env.reset()
        self.log = []
        self.score_log = [0]
        

global_model = Model(2).cuda()

opt = optimizer.Adam(global_model.parameters(), lr=0.0013, weight_decay=1e-5)


gamma = 0.99
max_timesteps = 20

environments = [Environment() for _ in range(8)]


while True:
    for E in environments:
        a, c = E.model(torch.tensor(E.obs).cuda())
        log_prob = a
        action = torch.exp(log_prob).multinomial(1).detach().item()
        s1, reward, done, info = E.env.step(action)
        E.log.append((E.obs, action, reward, s1, log_prob, c))
        E.obs = s1
        if done: E.obs = E.env.reset()
        
        # Score logging
        E.score_log[-1] += 1
        if done:
            print(E.score_log[-1])
            E.score_log.append(0)


        if done or len(E.log) >= max_timesteps:
            sum_loss = torch.tensor(0.0).cuda()
            td_target = E.model(torch.tensor(s1).cuda())[1] if not done else torch.tensor(0.0).cuda()
            for i, data in enumerate(reversed(E.log)):
                s0, action, reward, s1, log_prob, c0 = data

                td_target = E.log[-i][5].detach()[0] if i!=0 else torch.tensor(0.0).cuda()

                td_target = torch.tensor(reward).cuda() + gamma * td_target
                td_error = td_target - c0.detach()[0]

                value_loss = F.mse_loss(c0, td_target)
                policy_loss = -log_prob[action] * td_error + 0.01 * torch.sum(log_prob * torch.exp(log_prob))

                total_loss = value_loss + policy_loss
                sum_loss += total_loss
            
            ensure_shared_grads(E.model, global_model)
            
            opt.zero_grad()
            sum_loss.backward()
            opt.step()
            
            E.log = []
            E.model.load_state_dict(global_model.state_dict())
