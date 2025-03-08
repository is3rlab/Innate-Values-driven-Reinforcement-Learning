import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F


class InnateValuesDUCnn(nn.Module):
    def __init__(self, input_shape, needs_size, num_actions, batch_size):
        super(InnateValuesDUCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.needs_size = needs_size
        self.batch_size = batch_size

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Generate needs weight matrix
        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.needs_size),
            nn.Softmax(dim=1)
        )

        # Generate delta utilities matrix
        self.fc2 = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions * self.needs_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        needs = self.fc1(x)

        delta_utility = self.fc2(x)
        if delta_utility.shape[0] == 1:
            delta_utility = delta_utility.reshape(self.num_actions, self.needs_size)
        else:
            delta_utility = delta_utility.reshape(self.batch_size, -1, self.needs_size)

        return needs, delta_utility

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)