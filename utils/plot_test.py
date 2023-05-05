import matplotlib.pyplot as plt

def plot_test(sampled_rew, returns, environment):
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 3))

    rewards = sampled_rew[0]
    axes[0].plot([sum(rewards[:t+1]) for t in range(len(rewards))], label = 'Cumulative return')
    axes[0].plot(returns[:len(sampled_rew[0])], label = 'Estimated return')
    axes[0].set(xlabel = 'Steps', ylabel = 'Returns', title = 'Rollout 1')
    axes[0].legend()
    
    rewards = sampled_rew[1]
    axes[1].plot([sum(rewards[:t+1]) for t in range(len(rewards))], label = 'Cumulative return')
    axes[1].plot(returns[-len(sampled_rew[1]):], label = 'Estimated return')
    axes[1].set(xlabel = 'Steps', ylabel = 'Returns', title = 'Rollout 2')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'figures/test_experiments/returns_{environment}.pdf')
    plt.close()