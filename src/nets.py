from collections import namedtuple
import random

import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, n_outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def calc_feature_map_size(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride + 1

        n_conv_layers = 3
        feat_w, feat_h = w, h
        for _ in range(n_conv_layers):
            feat_w = calc_feature_map_size(feat_w)
            feat_h = calc_feature_map_size(feat_h)

        self.fc1 = nn.Linear(feat_w * feat_h * 64, 512)
        self.out = nn.Linear(512, n_outputs)

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(bs, -1)
        x = self.fc1(x)
        x = self.out(x)
        return x
