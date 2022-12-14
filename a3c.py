import numpy as np
import gym
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter

lr = 0.0001
hidden_dim = 128
rnn_layers = 2
process_count = 4
log_name = f'CartPole[0, 2]_lr:{lr}_hiddendim:{hidden_dim}_rnnlayers:{rnn_layers}_process:{process_count}'
writer = SummaryWriter(log_dir='runs/' + log_name)

from shared_adam import SharedAdam

def send_grad(model_from, model_to):
    isCuda = next(model_to.parameters()).is_cuda
    for param_from, param_to in zip(model_from.parameters(), model_to.parameters()):
        param_to._grad = param_from.grad.cuda() if isCuda else param_from.grad.cpu()

class Model(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.RNN(hidden_dim, hidden_dim, rnn_layers)
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_count),
            nn.LogSoftmax(dim=0)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, h):
        x = self.linear(x)
        x = x.view(1, *x.shape)
        x, h = self.rnn(x, h)
        x = x[0]
        x = self.linear2(x)
        return self.actor(x), self.critic(x), h

class Environment:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.model = Model(2).cuda()
        self.obs = self.env.reset()
        self.obs = [self.obs[0], self.obs[2]]
        self.log = []
        self.score_log = [0]
        self.h = torch.zeros(rnn_layers, hidden_dim).cuda()
        

global_model = Model(2)
global_model.share_memory()

opt = SharedAdam(global_model.parameters(), lr=lr, weight_decay=1e-6)
opt.share_memory()

params = {
    'gamma': 0.99,
    'max_timesteps': 20
}


def train(E, global_model, opt, params, process_num):
    gamma = params['gamma']
    max_timesteps = params['max_timesteps']
    while True:
        a, c, E.h = E.model(torch.tensor(E.obs).cuda(), E.h)
        log_prob = a
        action = torch.exp(log_prob).multinomial(1).detach().item()
        s1, reward, done, info = E.env.step(action)
        s1 = [s1[0], s1[2]]
        E.log.append((E.obs, action, reward, s1, log_prob, c))
        E.obs = s1
        if done:
            E.obs = E.env.reset()
            E.obs = [E.obs[0], E.obs[2]]
        
        # Score logging
        E.score_log[-1] += 1
        if done:
            if process_num == 0:
                print(E.score_log[-1])
                writer.add_scalar(f'score', E.score_log[-1], len(E.score_log))
            E.score_log.append(0)


        if done or len(E.log) >= max_timesteps:
            sum_loss = torch.tensor(0.0).cuda()
            td_target = E.model(torch.tensor(s1).cuda(), E.h)[1] if not done else torch.tensor(0.0).cuda()
            for i, data in enumerate(reversed(E.log)):
                s0, action, reward, s1, log_prob, c0 = data

                td_target = E.log[-i][5].detach()[0] if i!=0 else torch.tensor(0.0).cuda()

                td_target = torch.tensor(reward).cuda() + gamma * td_target
                td_error = td_target - c0.detach()[0]

                value_loss = F.mse_loss(c0[0], td_target)
                policy_loss = -log_prob[action] * td_error + 0.01 * torch.sum(log_prob * torch.exp(log_prob))

                total_loss = value_loss + policy_loss
                sum_loss += total_loss
            
            E.model.zero_grad()
            sum_loss.backward()
            send_grad(E.model, global_model)
            opt.step()
            
            E.log = []
            E.model.load_state_dict(global_model.state_dict())
            
            E.h = torch.zeros_like(E.h)


import time

if __name__ == '__main__':
    processes = []

    mp.set_start_method('spawn')
    for i in range(process_count):
        p = mp.Process(target=train, args=(Environment(), global_model, opt, params, i))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        time.sleep(0.1)
        p.join()
