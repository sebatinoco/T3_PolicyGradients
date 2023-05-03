import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_performance_metrics(tr_steps_vec, avg_reward_vec, std_reward_vec, exp_id, n_points = 100):
    
    plot_steps = len(tr_steps_vec) // n_points if len(tr_steps_vec) > 1 else 1
    
    tr_steps_vec = tr_steps_vec[::plot_steps]
    avg_reward_vec = avg_reward_vec[::plot_steps]
    std_reward_vec = std_reward_vec[::plot_steps]
    
    plt.errorbar(tr_steps_vec, avg_reward_vec, yerr = std_reward_vec, marker = '.', color = 'C0')
    plt.xlabel('Training Iteration')
    plt.ylabel('Avg Reward')
    plt.grid('on')
    plt.savefig(f'figures/agg_experiments/{exp_id}.pdf')
    plt.close()
    
def plot_experiment(exp_id):

    files = os.listdir('metrics')

    exp_files = [file for file in files if exp_id in file]

    exp_data = pd.DataFrame()

    for file in exp_files:
        
        file_data = pd.read_csv(f'metrics/{file}', sep = '\t')
        exp_data = pd.concat((exp_data, file_data), axis = 1)

    steps = exp_data['steps'].mean(axis = 1)
    avg_reward = exp_data['avg_reward'].mean(axis = 1)
    std_reward = exp_data['avg_reward'].std(axis = 1)

    plot_performance_metrics(steps, avg_reward, std_reward, exp_id)

if __name__ == '__main__':

    # experiment ids
    ids = ['CartPole_exp_11', 'CartPole_exp_21', 'CartPole_exp_31', 'CartPole_exp_41', 
           'CartPole_exp_12', 'CartPole_exp_22', 'CartPole_exp_32', 'CartPole_exp_42',
           'Pendulum_exp_12', 'Pendulum_exp_22', 'Pendulum_exp_32', 'Pendulum_exp_42'
           ]

    # plot for each experiment
    for exp_id in ids:
        print(exp_id)
        plot_experiment(exp_id)