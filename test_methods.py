import gym
from policy_gradients import PolicyGradients
from train_agent import perform_single_rollout, sample_rollouts
from utils.plot_test import plot_test

if __name__ == '__main__':
    
    environments = ['CartPole-v1', 'Pendulum-v1']
    for environment in environments:
        
        for _ in range(10):
        
            env = gym.make(environment) # initiate environment
            dim_states = env.observation_space.shape[0] # compute state space
            continuous_control = isinstance(env.action_space, gym.spaces.Box) # True if action space is continuous, False if not 
            dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n # compute action space
            
            # initiate agent
            policy_gradients_agent = PolicyGradients(dim_states=dim_states, 
                                                    dim_actions=dim_actions, 
                                                    continuous_control=continuous_control,
                                                    lr = 5e-3,
                                                    gamma = .99,
                                                    )

            # test for perform_single_rollout
            single_rollout = perform_single_rollout(env = env, agent = policy_gradients_agent, episode_nb = 1)
            
            # test for sample_rollouts
            sampled_rollouts = sample_rollouts(env = env, agent = policy_gradients_agent, training_iter = 1, min_batch_steps = 250)
            sampled_rew = [sampled_rollouts[i][2] for i in range(len(sampled_rollouts))]
            
            estimated_returns = policy_gradients_agent.estimate_returns(sampled_rew)
            
            # test for estimated returns 
            
            if _ == 0: # plot just for the first iteration
            
                sampled_rew = [list(perform_single_rollout(env = env, agent = policy_gradients_agent, episode_nb = 1)[2]) for _ in range(2)] # sample 2 rollouts, obtain rewards
                returns = policy_gradients_agent.estimate_returns(sampled_rew) # compute returns
            
                plot_test(sampled_rew, returns, environment = environment) # plot rewards vs returns
            
            
            
            
            