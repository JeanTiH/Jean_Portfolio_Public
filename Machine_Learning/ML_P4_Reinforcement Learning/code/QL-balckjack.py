""""""
"""OMSCS2023FALL-P4: Reinforcement Learning   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

import time
import random
import os
import gym
from algorithms.rl import RL
from examples.test_env import TestEnv
import pickle

import seaborn as sns
from matplotlib.patches import Patch
from collections import defaultdict
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
    ax2.set_ylabel("Dealer State", fontsize=18)
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

if __name__ == "__main__":
    seed_value = 42
    random.seed(seed_value)
    random_integers = [random.randint(1, 500) for _ in range(100)]

    np.random.seed(seed_value)
    blackjack = Blackjack()
    theta = 1e-4
    n_iters = 2000
    episodes = 1000

    metric_filename = 'QL-blackjack.txt'
    if os.path.exists(metric_filename):
        os.remove(metric_filename)

    gamma = 0.2
    epsilon = 0.5
    learning_rate = 0.99
    blackjack.env.reset(seed=seed_value)
    Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions,
                                                               blackjack.convert_state_obs, gamma=gamma,
                                                               init_alpha=learning_rate, min_alpha=0.01,
                                                               alpha_decay_ratio=0.4, init_epsilon=epsilon,
                                                               min_epsilon=0.1, epsilon_decay_ratio=0.9,
                                                               n_episodes=episodes)
    # This list present gamma, learning_rate and epsilon
    perameter_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    '''
    Step1 Learning rate change
    '''
    gamma = 0.2
    epsilon = 0.9
    max_Q = {}
    runtime_list_lr = []
    for i in range(len(perameter_list)):
        blackjack.env.reset(seed=seed_value)
        start = time.time()
        Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions,
                                                               blackjack.convert_state_obs, gamma=gamma,
                                                               init_alpha=perameter_list[i], min_alpha=0.01,
                                                               alpha_decay_ratio=0.4, init_epsilon=epsilon,
                                                               min_epsilon=0.1, epsilon_decay_ratio=0.9,
                                                               n_episodes=episodes)
        runtime_list_lr.append(time.time()-start)
        # Track the maximum Q-value for each iteration
        max_Q[perameter_list[i]] = np.max(Q_track, axis=(1, 2))

    # Plot learning rate change
    plt.figure(figsize=(8, 5))

    for learning_rate, max_q_values in max_Q.items():
        plt.plot(max_q_values, label=f'LR={learning_rate}')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Qlearning with gamma=' + str(gamma) + ' epsilon=' + str(epsilon))
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Maximum Q-Value', fontsize=20)
    plt.legend(fontsize=16, loc='center right', bbox_to_anchor=(1.35, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-QL-learningrate.png')
    plt.close()
    '''
    Step2 Epsilon change
    '''
    gamma = 0.1
    learning_rate = 0.9
    max_Q = {}
    runtime_list_ep = []
    for i in range(len(perameter_list)):
        blackjack.env.reset(seed=seed_value)
        start = time.time()
        Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions,
                                                                   blackjack.convert_state_obs, gamma=gamma,
                                                                   init_alpha=learning_rate, min_alpha=0.01,
                                                                   alpha_decay_ratio=0.4, init_epsilon=perameter_list[i],
                                                                   min_epsilon=0.1, epsilon_decay_ratio=0.9,
                                                                   n_episodes=episodes)
        runtime_list_ep.append(time.time() - start)
        # Track the maximum Q-value for each iteration
        max_Q[perameter_list[i]] = np.max(Q_track, axis=(1, 2))

    # Plot epsilon change
    plt.figure(figsize=(8, 5))

    for epsilon, max_q_values in max_Q.items():
        plt.plot(max_q_values, label=f'epsilon={epsilon}')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Qlearning with gamma=' + str(gamma) + ' learning rate=' + str(learning_rate))
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Maximum Q-Value', fontsize=20)
    plt.legend(fontsize=16, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-QL-epsilon.png')
    plt.close()
    '''
    Step3 Gamma change
    '''
    learning_rate = 0.99
    epsilon = 0.5
    max_Q = {}
    runtime_list_ga = []
    for i in range(len(perameter_list)):
        blackjack.env.reset(seed=seed_value)
        start = time.time()
        Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions,
                                                                   blackjack.convert_state_obs, gamma=perameter_list[i],
                                                                   init_alpha=learning_rate, min_alpha=0.01,
                                                                   alpha_decay_ratio=0.4, init_epsilon=epsilon,
                                                                   min_epsilon=0.1, epsilon_decay_ratio=0.9,
                                                                   n_episodes=episodes)
        runtime_list_ga.append(time.time()-start)
        # Track the maximum Q-value for each iteration
        max_Q[perameter_list[i]] = np.max(Q_track, axis=(1, 2))

    # Plot gamma change
    plt.figure(figsize=(8, 5))

    for gamma, max_q_values in max_Q.items():
        plt.plot(max_q_values, label=f'gamma={gamma}')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Qlearning with learning rate=' + str(learning_rate) + ' epsilon=' + str(epsilon))
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Maximum Q-Value', fontsize=20)
    plt.legend(fontsize=16, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-QL-gamma.png')
    plt.close()

    # Plot time difference for lr,ep,ga
    plt.figure(figsize=(8, 5))
    plt.plot(perameter_list, runtime_list_lr, marker='o', linestyle='-', color='r', label='Learning Rate')
    plt.plot(perameter_list, runtime_list_ep, marker='o', linestyle='-', color='b', label='Epsilon')
    plt.plot(perameter_list, runtime_list_ga, marker='o', linestyle='-', color='g', label='Gamma')
    plt.xticks(perameter_list, fontsize=15)
    plt.yticks(fontsize=18)
    plt.title('Runtime for Different Parameters')
    plt.xlabel('Parameter Value', fontsize=20)
    plt.ylabel('Runtime (seconds)', fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/Blackjack-QL-lr-ep-ga-time.png')

    '''
    Step4 QL test
    '''
    np.random.seed(seed_value)
    blackjack = Blackjack()
    gamma = 0.2
    epsilon = 0.5
    learning_rate = 0.99
    episodes = [100000, 1000000]
    for episode in episodes:
        blackjack.env.reset(seed=seed_value)
        start = time.time()
        Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions,
                                                                   blackjack.convert_state_obs, gamma=gamma,
                                                                   init_alpha=learning_rate, min_alpha=0.01,
                                                                   alpha_decay_ratio=0.4, init_epsilon=epsilon,
                                                                   min_epsilon=0.1, epsilon_decay_ratio=0.9,
                                                                   n_episodes=episode)
        runtime = time.time() - start
        policy = np.argmax(Q, axis=1)
        plot_policy(V, pi, 'plots/Blackjack-QL-policy' + str(episode) + '.png')

        with open(metric_filename, 'a') as fp:
            fp.write(f'***********************************' + '\n')
            fp.write(
                f'Qlearning with learning rate = ' + str(learning_rate) + ' epsilon = ' + str(
                    epsilon) + ' gamma = ' + str(
                    gamma) + ' episode = ' + str(episode) + '\n')
            fp.write(f'Runtime = ' + str(runtime) + '\n')

        msd_values = np.mean(np.square(np.diff(Q_track, axis=0)), axis=(1, 2))
        for i, msd in enumerate(msd_values):
            if msd < theta:
                pass

        # Plot mean squared difference
        plt.figure(figsize=(8, 5))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.plot(msd_values)
        plt.title('Qlearning with LR=' + str(learning_rate) + ' epsilon=' + str(epsilon) + ' gamma=' + str(gamma))
        plt.xlabel('Episode', fontsize=20)
        plt.ylabel('Mean Squared Difference', fontsize=20)
        plt.tight_layout()
        plt.savefig('plots/Blackjack-QL-converge' + str(episode) + '.png')
        plt.close()

        # test
        test_scores = TestEnv.test_env(env=blackjack.env, render=False, pi=pi, user_input=False,
                                       convert_state_obs=blackjack.convert_state_obs, n_iters=1000)
        win = np.count_nonzero(test_scores == 1)
        lose = np.count_nonzero(test_scores == -1)
        draw = np.count_nonzero(test_scores == 0)

        with open(metric_filename, 'a') as fp:
            fp.write(f'The agent has ' + str(win) + ' Win, ' + str(lose) + ' Lose, ' + str(draw) + ' Draw' + '\n')
        # print('The agent has ' + str(win) + ' Win, ' + str(lose) + ' Lose, ' + str(draw) + ' Draw')

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

        win_list.sort()
        lose_list.sort()
        draw_list.sort()

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

    gamma = 0.2
    epsilon = 0.5
    learning_rate = 0.99
    episodes = 1000
    blackjack.env.reset(seed=seed_value)
    Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions,
                                                               blackjack.convert_state_obs, gamma=gamma,
                                                               init_alpha=learning_rate, min_alpha=0.01,
                                                               alpha_decay_ratio=0.4, init_epsilon=epsilon,
                                                               min_epsilon=0.1, epsilon_decay_ratio=0.9,
                                                               n_episodes=episodes)