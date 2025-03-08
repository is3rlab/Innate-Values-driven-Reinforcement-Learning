import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from ..utils.replay_buffer import ReplayBuffer


class DQNAgent():
    def __init__(self, input_shape, needs_size, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau,
                 update_every, replay_after, model):
        """Initialize an Agent object.

        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate
            update_every (int): how often to update the network
            replay_after (int): After which replay to be started
            model(Model): Pytorch Model
        """
        self.input_shape = input_shape
        self.needs_size = needs_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.InnateValuesDUCnn = model
        self.tau = tau
        self.loss = 0

        # Innate-Values-Network
        self.innate_values_net = self.InnateValuesDUCnn(input_shape, needs_size, action_size, batch_size).to(
            self.device)
        self.target_net = self.InnateValuesDUCnn(input_shape, needs_size, action_size, batch_size).to(self.device)
        self.optimizer = optim.Adam(self.innate_values_net.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)

        self.t_step = 0

    def numpy_softmax(self, w):
        return np.exp(w) / np.exp(w).sum()

    def step(self, state, needs_weight, action, delta_utility, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, needs_weight, action, delta_utility, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.innate_values_net.eval()
        with torch.no_grad():
            delta_utility, needs_weight = self.innate_values_net(state)
        self.innate_values_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # state = np.expand_dims(state, axis=0)
            # state = torch.from_numpy(state).float().to(self.device)
            n_tmp, u_tmp = self.innate_values_net(state)
            u_tmp = u_tmp.reshape(self.action_size, self.needs_size)
            action = torch.argmax(torch.sum(n_tmp * u_tmp, dim=1)).item()
            needs_weight = n_tmp.cpu().detach().numpy()[0]
            return needs_weight, action
            # return np.argmax(action_values.cpu().data.numpy())
        else:
            return self.numpy_softmax(np.random.rand(self.needs_size)), random.choice(range(self.action_size))

    def get_innate_values_rewards(self, utilities, needs_weights):
        return utilities @ needs_weights

    def max_delta_utility_reward(self, needs_weights, delta_utilities):
        next_delta_utilities = []
        next_innate_values = []

        for i in range(self.batch_size):
            next_delta_utilities.append(delta_utilities[i][torch.sum((needs_weights[i] * delta_utilities[i]), dim=1).argmax().item()])
            # next_innate_values.append(torch.max(torch.sum((needs_weights[i] * delta_utilities[i].detach()), dim=1)))
            next_innate_values.append(torch.max(torch.sum((needs_weights[i] * delta_utilities[i]), dim=1)))

        return torch.stack(next_delta_utilities), torch.Tensor(next_innate_values)

    def selecting(self, input, index):
        output = []

        for i in range(input.shape[0]):
            output.append(torch.flatten(torch.index_select(input[i], 0, index[i]), end_dim=1))

        return torch.stack(output)

    def learn(self, experiences):
        states, needs_weights, actions, delta_utilities, rewards, next_states, dones = experiences

        # Get expected needs weights and delta utilities from innate values model
        needs_weight_expected, delta_utility_expected_current = self.innate_values_net(states)
        delta_utility_expected = self.selecting(delta_utility_expected_current, actions)

        # innate_value_expected = torch.sum(delta_utility_expected.detach() * needs_weight_expected, dim=1)
        innate_value_expected = torch.sum(delta_utility_expected * needs_weight_expected, dim=1)

        # Get max predicted innate values (for next states) from target model
        needs_weight_next, delta_utility_targets_next_current = self.target_net(next_states)
        delta_utility_targets_next, next_innate_values = self.max_delta_utility_reward(needs_weight_next, delta_utility_targets_next_current)

        # Compute innate values targets for current states
        innate_values_targets = rewards + (self.gamma * next_innate_values.to(self.device) * (1 - dones))

        # Compute loss
        # loss1 = F.mse_loss(innate_value_expected, innate_values_targets)
        # loss2 = F.mse_loss(delta_utility_expected, delta_utility_targets_next)
        # loss = loss1 + loss2

        loss = F.mse_loss(innate_value_expected, innate_values_targets)

        self.loss = loss.item()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.innate_values_net, self.target_net, self.tau)

    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, innate_values_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), innate_values_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)