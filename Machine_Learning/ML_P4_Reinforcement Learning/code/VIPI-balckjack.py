""""""
"""OMSCS2023FALL-P4: Reinforcement Learning   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

# some functions are adapted from the link below and modified
# https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import time

import os
import gym
from algorithms.planner import Planner
from examples.test_env import TestEnv
import pickle

import random
import seaborn as sns
from matplotlib.patches import Patch
from collections import defaultdict
import math
import plots
'''
Plot blackjack policy
'''
def plot_policy(V, pi, savepath):

    policy = defaultdict(int)
    for obs, action_values in enumerate(V):
        policy[obs] = int(pi(obs))

    player_count, dealer_count = np.meshgrid(np.arange(1, 30), np.arange(1, 11), )

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(lambda obs: policy[obs[0] * 10 + obs[1] - 12],  axis=2, arr=np.dstack([player_count, dealer_count]),)

    fig = plt.figure(figsize=(6, 6))
    ax2 = fig.add_subplot(1, 1, 1)
    sns.heatmap(
        policy_grid,
        linewidth=0,
        cmap="inferno",
        cbar=False,
        annot=False,
        #annot_kws={"size": 8},
        #fmt="d",
        ax=ax2,
        xticklabels=1,
        yticklabels=1,
    )
    ax2.set_title('Blackjack Policy')
    ax2.set_xlabel("Player State", fontsize=18)
    ax2.set_ylabel("Dealer state", fontsize=18)
    ax2.set_xticks(np.arange(1.5, 30, 2))
    ax2.set_xticklabels(range(1, 30, 2), fontsize=15, rotation=0, ha="right")
    ax2.set_yticks(range(1, 11), list(range(1, 11)), fontsize=15)

    legend_elements = [Patch(facecolor="lightyellow", edgecolor="black", label="Hit"),
        Patch(facecolor="black", edgecolor="black", label="Stand"),
    ]
    ax2.legend(handles=legend_elements, fontsize=14, bbox_to_anchor=(1, 1.18))
    plt.savefig(savepath)
    plt.close()
'''
1. Blackjack
'''
class Blackjack:
    def __init__(self):
        self._env = gym.make('Blackjack-v1', render_mode=None)
        # Explanation of convert_state_obs lambda:
        # def function(state, done):
        # 	if done:
		#         return -1
        #     else:
        #         if state[2]:
        #             int(f"{state[0]+6}{(state[1]-2)%10}")
        #         else:
        #             int(f"{state[0]-4}{(state[1]-2)%10}")
        self._convert_state_obs = lambda state, done: (
            -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
                f"{state[0] - 4}{(state[1] - 2) % 10}"))
        # Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
        current_dir = os.path.dirname(__file__)
        file_name = 'blackjack-envP'
        f = os.path.join(current_dir, file_name)
        try:
            self._P = pickle.load(open(f, "rb"))
        except IOError:
            print("Pickle load failed.  Check path", f)
        self._n_actions = self.env.action_space.n
        self._n_states = len(self._P)

    @property
    def n_actions(self):
        return self._n_actions

    @n_actions.setter
    def n_actions(self, n_actions):
        self._n_actions = n_actions

    @property
    def n_states(self):
        return self._n_states

    @n_states.setter
    def n_states(self, n_states):
        self._n_states = n_states

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    @property
    def convert_state_obs(self):
        return self._convert_state_obs

    @convert_state_obs.setter
    def convert_state_obs(self, convert_state_obs):
        self._convert_state_obs = convert_state_obs
'''
2. Value Iteration
'''
def value_iteration(env, gamma, theta, n_iters):
    n_states = env.n_states
    n_actions = env.n_actions
    # initialize value table with zeros
    value_table = np.zeros(n_states)

    cumulative_rewards = []
    delta_values = []

    for i in range(n_iters):
        # On each iteration, copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)
        reward = 0
        # Now we calculate Q Value for each actions in the state
        # and update the value of a state with maximum Q value
        for state in range(n_states):
            Q_value = []
            for action in range(n_actions):
                next_states_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))

                Q_value.append(np.sum(next_states_rewards))

            value_table[state] = max(Q_value)
            reward += value_table[state]
        cumulative_rewards.append(reward)
        delta = np.sum(np.fabs(updated_value_table - value_table))
        delta_values.append(delta)
        # check convergence creteria - whether the difference between value table and updated value table is small enough
        if (delta <= theta):
            #print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
        converge_iter = i+1
    return value_table, cumulative_rewards, delta_values, converge_iter

'''
3. Policy Iteration
'''
def compute_value_function(env, policy, gamma, theta):
    n_states = env.n_states
    n_actions = env.n_actions
    # initialize value table with zeros
    value_table = np.zeros(n_states)

    cumulative_rewards = []
    delta_values = []

    while True:
        # copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)
        reward = 0

        # for each state in the environment, select the action according to the policy and compute the value table
        for state in range(n_states):
            action = policy[state]

            # build the value table with the selected action
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                      for trans_prob, next_state, reward_prob, _ in env.P[state][action]])
            reward += value_table[state]

        delta = np.sum(np.fabs(updated_value_table - value_table))
        if (delta <= theta):
            break
        cumulative_rewards.append(reward)
        if delta != 0:
            delta_values.append(delta)

    return value_table

def extract_policy(env, value_table, gamma):
    n_states = env.n_states
    n_actions = env.n_actions
    # Initialize the policy with zeros
    policy = np.zeros(n_states)

    for state in range(n_states):

        # initialize the Q table for a state
        Q_table = np.zeros(n_actions)

        # compute Q value for all ations in the state
        for action in range(n_actions):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

        # Select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)

    return policy

def policy_iteration(env, gamma, theta, n_iters):
    n_states = env.n_states
    n_actions = env.n_actions
    # Initialize policy with zeros
    old_policy = np.zeros(n_states)
    V = []

    for i in range(n_iters):

        # compute the value function
        new_value_function = compute_value_function(env, old_policy, gamma, theta)
        V.append(new_value_function.mean())
        # Extract new policy from the computed value function
        new_policy = extract_policy(env, new_value_function, gamma)

        # Check convergence creteria - old_policy and new policy are the same
        if (np.all(old_policy == new_policy)):
            #print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        old_policy = new_policy
        converge_iter = i+1
    return new_policy, V, converge_iter
'''
4. Get Policy from V
'''
def get_policy(env, stateValue, lmbda):
    n_states = env.n_states
    n_actions = env.n_actions
    policy = [0 for i in range(n_states)]
    for state in range(n_states):
        action_values = []
        for action in range(n_actions):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, r, _ = env.P[state][action][i]
                action_value += prob * (r + lmbda * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy
'''
-----------------------------
        Run Algorithms
-----------------------------
'''
if __name__ == "__main__":
    seed_value = 42
    random.seed(seed_value)
    random_integers = [random.randint(1, 500) for _ in range(100)]

    np.random.seed(seed_value)
    blackjack = Blackjack()

    metric_filename = 'VIPI-blackjack.txt'
    if os.path.exists(metric_filename):
        os.remove(metric_filename)

    theta = 1e-4
    n_iters = 2000
    episodes = 1000
    '''
    Step1 Gamma change
    '''
    gamma_list = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    runtime_list_VI = []
    mean_V_list_VI = []
    runtime_list_PI = []
    mean_V_list_PI = []

    for i in range(len(gamma_list)):
        # Value Iteration
        blackjack.env.reset(seed=seed_value)
        start = time.time()
        V, V_track, pi = Planner(blackjack.P).value_iteration(gamma=gamma_list[i], n_iters=n_iters, theta=theta)
        runtime_list_VI.append(time.time() - start)

        mean_V = np.mean(V_track, axis=1)
        # Find the index of the last non-zero value
        last_non_zero_index = np.maximum.reduce(np.where(mean_V != 0, np.arange(mean_V.size), -1))
        # Fill zeros after the last non-zero value with that last non-zero value
        mean_V[last_non_zero_index + 1:] = mean_V[last_non_zero_index]
        mean_V_list_VI.append(mean_V)

        # Policy Iteration
        blackjack.env.reset(seed=seed_value)
        start = time.time()
        V, V_track, pi = Planner(blackjack.P).policy_iteration(gamma=gamma_list[i], n_iters=n_iters, theta=theta)
        runtime_list_PI.append(time.time() - start)

        mean_V = np.mean(V_track, axis=1)
        # Find the index of the last non-zero value
        last_non_zero_index = np.maximum.reduce(np.where(mean_V != 0, np.arange(mean_V.size), -1))
        # Fill zeros after the last non-zero value with that last non-zero value
        mean_V[last_non_zero_index + 1:] = mean_V[last_non_zero_index]
        mean_V_list_PI.append(mean_V)

    # Plot VI (gamma)
    plt.figure(figsize=(8, 5))
    for i in range(len(gamma_list)):
        plt.plot(np.arange(1, 6), mean_V_list_VI[i][:5], label=f'Gamma={gamma_list[i]}')

    plt.xticks(np.arange(1, 6), fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Value Iteration: Different Gamma')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Mean V', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-VI-gamma.png')
    plt.close()

    # Plot PI (gamma)
    plt.figure(figsize=(8, 5))
    for i in range(len(gamma_list)):
        plt.plot(np.arange(1, 6), mean_V_list_PI[i][:5], label=f'Gamma={gamma_list[i]}')

    plt.xticks(np.arange(1, 6), fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Policy Iteration: Different Gamma')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Mean V', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-PI-gamma.png')
    plt.close()

    # Plot VI PI time difference
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(gamma_list, runtime_list_VI, marker='o', linestyle='-', color='b', label='Value Iteration')
    plt.plot(gamma_list, runtime_list_PI, marker='*', linestyle='-', color='g', label='Policy Iteration')

    plt.xticks(gamma_list, fontsize=15)
    plt.yticks(fontsize=18)
    plt.title('Runtime for Different Gamma')
    plt.xlabel('Gamma Value', fontsize=20)
    plt.ylabel('Runtime (seconds)', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-VIPI-gamma-time.png')
    plt.close()
    '''

    plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(gamma_list, runtime_list_VI, marker='o', linestyle='-', color='b', label='Value Iteration')
    plt.title('Runtime for Different Gamma')
    plt.xlabel('Gamma Value', fontsize=20)
    plt.ylabel('VI Runtime (s)', fontsize=20)
    plt.grid(True)

    lines1, labels1 = plt.gca().get_legend_handles_labels()
    ax2 = plt.twinx()
    plt.yticks(fontsize=18)
    plt.plot(gamma_list, runtime_list_PI, marker='*', linestyle='-', color='g', label='Policy Iteration')
    ax2.set_ylabel('PI Runtime (s)', fontsize=20)
    ax2.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=18)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-VIPI-gamma-time.png')
    plt.close()

    '''
    Step2 Value iteration test
    '''
    gamma = 0.2
    blackjack.env.reset(seed=seed_value)
    start = time.time()
    V, V_track, pi = Planner(blackjack.P).value_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
    runtime = time.time() - start
    plot_policy(V, pi, 'plots/Blackjack-VI-policy.png')
    policy = get_policy(blackjack, V, gamma)
    value_table, cumulative_rewards, delta_values, converge_iter = value_iteration(blackjack, gamma, theta, n_iters)

    with open(metric_filename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'Value iteration with gamma = ' + str(gamma) + ' theta = ' + str(theta) + '\n')
        fp.write(f'Runtime = ' + str(runtime) + '\n')
        fp.write(f'Converge iteration = ' + str(converge_iter) + '\n')

    plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(np.arange(1, len(delta_values) + 1))
    iterations = np.arange(1, len(delta_values) + 1)
    plt.plot(iterations, cumulative_rewards, color='b', label='Cumulative Rewards')
    plt.title('Value Iteration with Gamma=' + str(gamma) + ' theta=' + str(theta))
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Cumulative Rewards', fontsize=20)
    plt.grid(True)

    lines1, labels1 = plt.gca().get_legend_handles_labels()
    ax2 = plt.twinx()
    plt.yticks(fontsize=18)
    ax2.plot(iterations, delta_values, color='g', label='Delta')
    ax2.set_ylabel('Delta', fontsize=20)
    ax2.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=18)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-VI-converge.png')
    plt.close()

    test_scores = TestEnv.test_env(env=blackjack.env, render=False, pi=pi, user_input=False,
                                   convert_state_obs=blackjack.convert_state_obs, n_iters=1000)
    win = np.count_nonzero(test_scores == 1)
    lose = np.count_nonzero(test_scores == -1)
    draw = np.count_nonzero(test_scores == 0)

    with open(metric_filename, 'a') as fp:
        fp.write(f'The agent has ' + str(win) + ' Win, ' + str(lose) + ' Lose, ' + str(draw) + ' Draw' + '\n')
    #print('The agent has ' + str(win) + ' Win, ' + str(lose) + ' Lose, ' + str(draw) + ' Draw')

    '''
    Varying seeds
    '''
    win_list = []
    lose_list = []
    draw_list = []

    for random_seed in random_integers:
        blackjack.env.reset(seed=random_seed)
        test_scores = TestEnv.test_env(env=blackjack.env, render=False, pi=pi, user_input=False,
                                       convert_state_obs=blackjack.convert_state_obs, n_iters=1000)
        win = np.count_nonzero(test_scores == 1)
        lose = np.count_nonzero(test_scores == -1)
        draw = np.count_nonzero(test_scores == 0)
        win_list.append(win)
        lose_list.append(lose)
        draw_list.append(draw)

    with open(metric_filename, 'a') as fp:
        fp.write(f'---------------------------' + '\n')
        fp.write(f'With 100 random seeds: \n')
        fp.write(f' Win range from ' + str(min(win_list)) + ' to ' + str(max(win_list)) + '\n')
        fp.write(f'Lose range from ' + str(min(lose_list)) + ' to ' + str(max(lose_list)) + '\n')
        fp.write(f'Draw range from ' + str(min(draw_list)) + ' to ' + str(max(draw_list)) + '\n')
        fp.write(f'\n')
        fp.write(f'Mean Win = ' + str(np.mean(win_list)) + '\n')
        fp.write(f'Mean Lose = ' + str(np.mean(lose_list)) + '\n')
        fp.write(f'Mean Draw = ' + str(np.mean(draw_list)) + '\n')
    '''
    Step3 Policy iteration test
    '''
    gamma = 0.2
    blackjack.env.reset(seed=seed_value)
    start = time.time()
    V, V_track, pi = Planner(blackjack.P).policy_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
    runtime = time.time() - start
    plot_policy(V, pi, 'plots/Blackjack-PI-policy.png')
    #policy = get_policy(blackjack, V, gamma)
    policy, V_iteration, converge_iter = policy_iteration(blackjack, gamma, theta, n_iters)

    with open(metric_filename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'Policy iteration with gamma = ' + str(gamma) + ' theta = ' + str(theta) + '\n')
        fp.write(f'Runtime = ' + str(runtime) + '\n')
        fp.write(f'Converge iteration = ' + str(converge_iter) + '\n')

    plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(np.arange(1, len(V_iteration) + 1))
    iterations = np.arange(1, len(V_iteration) + 1)
    plt.plot(iterations, V_iteration)
    plt.title('Policy Iteration with Gamma=' + str(gamma) + ' theta=' + str(theta))
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('State Value', fontsize=20)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('plots/Blackjack-PI-converge.png')
    plt.close()

    test_scores = TestEnv.test_env(env=blackjack.env, render=False, pi=pi, user_input=False,
                                    convert_state_obs=blackjack.convert_state_obs, n_iters=1000)
    win = np.count_nonzero(test_scores == 1)
    lose = np.count_nonzero(test_scores == -1)
    draw = np.count_nonzero(test_scores == 0)

    with open(metric_filename, 'a') as fp:
        fp.write(f'The agent has ' + str(win) + ' Win, ' + str(lose) + ' Lose, ' + str(draw) + ' Draw' + '\n')
    #print('The agent has ' + str(win) + ' Win, ' + str(lose) + ' Lose, ' + str(draw) + ' Draw')

    '''
    Varying seeds
    '''
    win_list = []
    lose_list = []
    draw_list = []

    for random_seed in random_integers:
        blackjack.env.reset(seed=random_seed)
        test_scores = TestEnv.test_env(env=blackjack.env, render=False, pi=pi, user_input=False,
                                   convert_state_obs=blackjack.convert_state_obs, n_iters=1000)
        win = np.count_nonzero(test_scores == 1)
        lose = np.count_nonzero(test_scores == -1)
        draw = np.count_nonzero(test_scores == 0)
        win_list.append(win)
        lose_list.append(lose)
        draw_list.append(draw)

    with open(metric_filename, 'a') as fp:
        fp.write(f'---------------------------' + '\n')
        fp.write(f'With 100 random seeds: \n')
        fp.write(f' Win range from ' + str(min(win_list)) + ' to ' + str(max(win_list)) + '\n')
        fp.write(f'Lose range from ' + str(min(lose_list)) + ' to ' + str(max(lose_list)) + '\n')
        fp.write(f'Draw range from ' + str(min(draw_list)) + ' to ' + str(max(draw_list)) + '\n')
        fp.write(f'\n')
        fp.write(f'Mean Win = ' + str(np.mean(win_list)) + '\n')
        fp.write(f'Mean Lose = ' + str(np.mean(lose_list)) + '\n')
        fp.write(f'Mean Draw = ' + str(np.mean(draw_list)) + '\n')