import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Policy, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
        # dim_states -> 64 -> 64 -> dim_actions

        self._continuous_control = continuous_control

        self.fc1 = nn.Linear(dim_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_actions)

        if continuous_control:
            # trainable parameter
            self._log_std = torch.tensor([-0.5] * dim_actions)
            self._log_std = nn.Parameter(self._log_std)

    def forward(self, input):
        
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        
        if self._continuous_control:
            mean = self.fc3(input)
            std = torch.exp(self._log_std)
            return mean, std
        else:
            probs = F.softmax(self.fc3(input), dim = 1)
            return probs


class PolicyGradients:

    def __init__(self, dim_states, dim_actions, lr, gamma, 
                 continuous_control=False, reward_to_go=False, use_baseline=False):
        
        self._learning_rate = lr
        self._gamma = gamma
        
        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control
        self._use_reward_to_go = reward_to_go
        self._use_baseline = use_baseline

        self._policy = Policy(self._dim_states, self._dim_actions, self._continuous_control).to(device)
        # Adam optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr = 1e-3)

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_loss = self._compute_loss_continuous if self._continuous_control else self._compute_loss_discrete


    def select_action(self, observation):
        return self._select_action(observation)
        

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs = self._policy(observation).to(device) #.cpu()
            distr = Categorical(probs)
        
        action = distr.sample()
        log_prob = distr.log_prob(action) # log probability of action
        
        return action.item(), log_prob # quitar log prob
        

    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter
        
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean, std = self._policy(observation)#.to(device) #.cpu()
            mean, std = mean.to(device), std.to(device)
            distr = Normal(mean, std)
            
        action = distr.sample().squeeze(0)
        log_prob = distr.log_prob(action) # log probability of action
        
        return action.numpy(), log_prob # quitar log prob
            

    def update(self, observation_batch, action_batch, advantage_batch):
        # update the policy here
        # you should use self._compute_loss 
        
        if self._continuous_control:
            loss = self._compute_loss_continuous(observation_batch, action_batch, advantage_batch)
        else:
            loss = self._compute_loss_discrete(observation_batch, action_batch, advantage_batch)
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
    

    def _compute_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        
        observation_batch = torch.from_numpy(observation_batch).to(device)
        action_batch = torch.from_numpy(action_batch).to(device)
        
        probs = self._policy(observation_batch).to(device) #.cpu()
        distr = Categorical(probs)
        
        log_probs = distr.log_prob(action_batch) # compute log_prob for each pair mean-action
        log_probs = log_probs.squeeze().to(device)
        
        advantage_batch = torch.from_numpy(advantage_batch).to(device)
        loss = torch.multiply(-log_probs, advantage_batch)
        
        return torch.mean(loss)


    def _compute_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        
        observation_batch = torch.from_numpy(observation_batch).to(device)
        action_batch = torch.from_numpy(action_batch).to(device)
        
        mean, std = self._policy(observation_batch)
        distr = Normal(mean, std) # 200 mean, 1 std

        log_probs = distr.log_prob(action_batch) # compute log_prob for each pair mean-action
        log_probs = log_probs.squeeze().to(device)

        advantage_batch = torch.from_numpy(advantage_batch).to(device)
        loss = torch.multiply(-log_probs, advantage_batch)

        return torch.mean(loss)

    
    def estimate_returns(self, rollouts_rew):
        estimated_returns = []
        for rollout_rew in rollouts_rew:
                
            if self._use_reward_to_go:
                # only for part 2
                estimated_return = None
            else:
                estimated_return = [rollout_rew[t] * (self._gamma ** t) for t in range(len(rollout_rew))]
            
            estimated_returns = np.concatenate([estimated_returns, estimated_return])

        if self._use_baseline:
            # only for part 2
            average_return_baseline = None
            # Use the baseline:
            #estimated_returns -= average_return_baseline

        return np.array(estimated_returns, dtype=np.float32)


    # It may be useful to discount the rewards using an auxiliary function [optional]
    def _discount_rewards(self, rewards):
        pass