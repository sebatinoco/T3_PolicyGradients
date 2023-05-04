import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

import numpy as np
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using {device}!')

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
            #self._log_std = torch.tensor([-0.5] * dim_actions)
            self._log_std = torch.zeros(dim_actions)
            self._log_std = nn.Parameter(self._log_std)

    def forward(self, input):
        
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        
        #if self._continuous_control:
        #    mean = self.fc3(input)
        #    std = torch.exp(self._log_std)
        #    return mean, std
        #else:
        #    probs = F.softmax(self.fc3(input))
        #    return probs
            
        return self.fc3(input) # logits or means


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
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr = lr)

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_loss = self._compute_loss_continuous if self._continuous_control else self._compute_loss_discrete


    def select_action(self, observation):
        return self._select_action(observation)
        

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            #probs = self._policy(observation).to(device) #.cpu() # cambiar por logits?
            #distr = Categorical(probs = probs)
            logits = self._policy(observation).to(device)
            distr = Categorical(logits = logits)
        
        action = distr.sample().item()
        
        return action
        

    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter
        
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean = self._policy(observation).to(device)
            std = torch.exp(self._policy._log_std).to(device)
            
        distr = Normal(mean, std)
            
        action = distr.sample().squeeze(0).cpu().numpy()
        
        return action
            

    def update(self, observation_batch, action_batch, advantage_batch):
        # update the policy here
        # you should use self._compute_loss 
        
        loss = self._compute_loss(observation_batch, action_batch, advantage_batch)
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
    

    def _compute_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        
        observation_batch = torch.from_numpy(observation_batch).to(device)
        action_batch = torch.from_numpy(action_batch).to(device)
        
        logits = self._policy(observation_batch).to(device)
        distr = Categorical(logits = logits)
        
        log_probs = distr.log_prob(action_batch) # compute log_prob for each pair mean-action
        log_probs = log_probs.squeeze().to(device) 
        
        advantage_batch = torch.from_numpy(advantage_batch).to(device)
        assert log_probs.shape == advantage_batch.shape
        
        loss = torch.multiply(-log_probs, advantage_batch)
        
        return torch.mean(loss)

    def _compute_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        
        observation_batch = torch.from_numpy(observation_batch).to(device)
        action_batch = torch.from_numpy(action_batch).to(device)
        
        #mean, std = self._policy(observation_batch)
        
        mean = self._policy(observation_batch).to(device)
        std = torch.exp(self._policy._log_std).to(device)
        
        distr = Normal(mean, std) # n mean, 1 std

        log_probs = distr.log_prob(action_batch) # compute log_prob for each pair mean-action
        log_probs = log_probs.squeeze().to(device)

        advantage_batch = torch.from_numpy(advantage_batch).to(device)
        assert log_probs.shape == advantage_batch.shape
        
        loss = torch.multiply(-log_probs, advantage_batch)

        return torch.mean(loss)

    
    def estimate_returns(self, rollouts_rew):
        estimated_returns = []
        for rollout_rew in rollouts_rew:
                
            if self._use_reward_to_go:
                # only for part 2
                estimated_return = deque() # for efficiency
                n_steps = len(rollout_rew) # steps of rollout
                
                for t in range(n_steps)[::-1]:
                    disc_return_t = estimated_return[0] if len(estimated_return) > 0 else 0
                    estimated_return.appendleft(disc_return_t * self._gamma + rollout_rew[t]) # pi_t = pi_{t+1} + r_t
                    
                estimated_return = list(estimated_return)
                
            else:
                #estimated_return = [sum(rollout_rew) * (self._gamma ** t) for t in range(len(rollout_rew))]
                estimated_return = [self._discount_rewards(rollout_rew)] * len(rollout_rew)
            
            estimated_returns = np.concatenate([estimated_returns, estimated_return])

        if self._use_baseline:
            # only for part 2
            average_return_baseline = np.mean(estimated_returns)
            # Use the baseline:
            estimated_returns -= average_return_baseline

        return np.array(estimated_returns, dtype=np.float32)

    # It may be useful to discount the rewards using an auxiliary function [optional]
    def _discount_rewards(self, rewards: list):
        
        return sum([rewards[t] * (self._gamma ** t) for t in range(len(rewards))])
