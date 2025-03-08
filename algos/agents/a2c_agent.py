import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random


class A2CAgent():
    def __init__(self, input_shape, needs_size, action_size, seed, device, gamma, zeta, alpha, beta, update_every,
                 innate_m, actor_m, critic_m):
        """Initialize an Agent object.
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            needs_size (int): dimension of each needs
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            zeta (float): Innate-Values learning rate
            alpha (float): Actor learning rate
            beta (float): Critic learning rate
            update_every (int): how often to update the network
            innate-value_m(Model): Pytorch Innate-Values Model
            actor_m(Model): Pytorch Actor Model
            critic_m(Model): PyTorch Critic Model
        """
        self.input_shape = input_shape
        self.needs_size = needs_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.zeta = zeta
        self.alpha = alpha
        self.beta = beta
        self.update_every = update_every
        self.loss = 0

        # Innate-Values-Network
        self.innate_values_net = innate_m(input_shape, needs_size).to(self.device)
        self.innate_values_optimizer = optim.Adam(self.innate_values_net.parameters(), lr=self.zeta)

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.critic_net = critic_m(input_shape, needs_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory
        self.needs_weights = []
        self.action_probs = []
        self.log_needs_probs = []
        self.log_action_probs = []
        self.delta_utilities = []
        self.critic_delta_utilities = []
        self.rewards = []
        self.masks = []
        self.entropies = []

        self.t_step = 0

    def step(self, state, needs_weight, log_needs_prob, log_prob, delta_utility, entropy, reward, done, next_state):

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)

        next_delta_utility = self.critic_net(state)

        # Save experience in  memory
        self.needs_weights.append(needs_weight)
        self.log_needs_probs.append(log_needs_prob)
        self.action_probs.append(torch.exp(log_prob))
        self.log_action_probs.append(log_prob)
        self.delta_utilities.append(torch.from_numpy(delta_utility))
        self.critic_delta_utilities.append(next_delta_utility)
        self.rewards.append(torch.from_numpy(np.array([reward])).to(self.device))
        self.masks.append(torch.from_numpy(np.array([1 - done])).to(self.device))
        self.entropies.append(entropy)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            self.learn(next_state)
            self.reset_memory()

    def innate_values_act(self, state):
        """Returns action, log_prob, entropy for given state as per current policy."""

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        needs_probs = self.innate_values_net(state)
        action_probs = self.actor_net(state)

        log_needs_prob = torch.log(needs_probs)

        action = action_probs.sample()
        log_action_prob = action_probs.log_prob(action)
        entropy = action_probs.entropy().mean()
        needs = needs_probs.cpu().detach().numpy()[0]

        return needs, needs_probs, log_needs_prob, action.item(), log_action_prob, entropy

    def get_innate_values_rewards(self, utilities, needs_weights):
        return utilities @ needs_weights.cpu().detach().numpy()[0]

    def learn(self, next_state):
        next_state = torch.from_numpy(next_state).unsqueeze(0).to(self.device)
        next_delta_utility = self.critic_net(next_state)

        returns = self.compute_returns(next_delta_utility, self.gamma)

        needs_weights = torch.cat(self.needs_weights)
        log_needs_probs = torch.cat(self.log_needs_probs)
        action_probs = torch.cat(self.action_probs)
        log_action_probs = torch.cat(self.log_action_probs)
        delta_utilities = torch.cat(self.delta_utilities)
        returns = torch.cat(returns).detach()
        critic_delta_utilities = torch.cat(self.critic_delta_utilities)

        advantage = returns - critic_delta_utilities

        # innate_values_loss = -(action_probs.detach() * log_needs_probs * advantage.detach()).mean()
        innate_values_loss = -(action_probs.detach() * (needs_weights * advantage.detach()).sum(dim=1)).mean()
        # actor_loss = - (log_action_probs * (needs_weights.detach() * advantage.detach()).sum(dim=1)).mean()
        actor_loss = - (action_probs * (needs_weights.detach() * advantage.detach()).sum(dim=1)).mean()
        critic_loss = advantage.pow(2).mean()

        # loss = actor_loss + 0.5 * critic_loss - 0.001 * sum(self.entropies)
        loss = innate_values_loss + actor_loss + 0.5 * critic_loss - 0.001 * sum(self.entropies)
        self.loss = loss.item()

        # Minimize the loss
        self.innate_values_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.innate_values_optimizer.step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def reset_memory(self):
        del self.needs_weights[:]
        del self.action_probs[:]
        del self.log_needs_probs[:]
        del self.log_action_probs[:]
        del self.delta_utilities[:]
        del self.critic_delta_utilities[:]
        del self.rewards[:]
        del self.masks[:]
        del self.entropies[:]

    def compute_returns(self, next_delta_utility, gamma=0.99):
        R = next_delta_utility
        returns = []
        for step in reversed(range(len(self.delta_utilities))):
            R = self.delta_utilities[step].to(self.device) + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns

    # def compute_returns(self, next_value, gamma=0.99):
    #     R = next_value
    #     returns = []
    #     for step in reversed(range(len(self.rewards))):
    #         R = self.rewards[step] + gamma * R * self.masks[step]
    #         returns.insert(0, R)
    #     return returns