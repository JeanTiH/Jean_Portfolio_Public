""""""
"""OMSCS2023FALL-P4: Reinforcement Learning   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import gym
from algorithms.rl import RL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.patches import Rectangle

from algorithms.planner import Planner
from gym.envs.toy_text.frozen_lake import generate_random_map
import os
import time
import math
import random
from examples.test_env import TestEnv
import plots

def compute_value_function(policy, gamma, theta):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
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

def extract_policy(value_table, gamma):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
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
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # Initialize policy with zeros
    old_policy = np.zeros(n_states)
    V = []

    for i in range(n_iters):

        # compute the value function
        new_value_function = compute_value_function(old_policy, gamma, theta)
        V.append(new_value_function.mean())
        # Extract new policy from the computed value function
        new_policy = extract_policy(new_value_function, gamma)

        # Check convergence creteria - old_policy and new policy are the same
        if (np.all(old_policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        old_policy = new_policy
    return new_policy, V

def get_policy(env, stateValue, lmbda):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
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

def get_score(env, policy, episodes):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:
            if isinstance(observation, tuple):
                observation_index = observation[0]  # Modify based on your observation structure
            else:
                observation_index = observation
            action = policy[observation_index]

            observation, reward, done, _, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                # Reach the Target
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # Fall in a hole
                misses += 1
                break

    if len(steps_list) > 0:
        mean_step = np.mean(steps_list)
    else:
        mean_step = np.nan
    print('Agent took an average of {:.0f} steps to get the frisbee'.format(mean_step))
    print('Agent fell in the hole {:.2f} % of the times'.format((misses / episodes) * 100))

def evaluation(env, qtable, episodes):
    nb_success = 0
    # Evaluation
    for _ in range(episodes):
        state = env.reset()[0]
        done = False

        # Until the agent gets stuck or reaches the goal, keep training it
        while not done:
            # Choose the action with the highest value in the current state
            action = np.argmax(qtable[state])

            # Implement this action and move the agent in the desired direction
            new_state, reward, done, _, info = env.step(action)

            # Update our current state
            state = new_state

            # When we get a reward, it means we solved the game
            nb_success += reward

    # Let's check our success rate!
    print(f"Success rate = {nb_success / episodes * 100}%")
'''
-----------------------------
        Run Algorithms
-----------------------------
'''
if __name__ == "__main__":
    seed_value = 42
    random.seed(seed_value)
    random_integers = [random.randint(1, 500) for _ in range(100)]

    theta = 1e-4
    n_iters = 2000
    #episodes = 1000
    #proba_frozen = 0.9
    is_slippery = True

    metric_filename = 'QL-frozenlake.txt'
    if os.path.exists(metric_filename):
        os.remove(metric_filename)

    '''
    Step1 QL test
    '''
    map_sizes = [4, 16]
    for i, map_size in enumerate(map_sizes):
        if map_size == 4:
            seed_value = 123
            test_iter = 1000
            episodes = 1000
            epsilon = 0.7
            learning_rate = 0.8
            proba_frozen = 0.9
        elif map_size == 16:
            seed_value = 100
            test_iter = 10
            episodes = 100000
            epsilon = 0.7
            learning_rate = 0.9
            proba_frozen = 0.9
        np.random.seed(seed_value)
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery,
                       desc=generate_random_map(size=map_size, p=proba_frozen), )

        for gamma in [0.1, 0.9]:
            env.reset(seed=seed_value)
            planner_instance = Planner(env.P)
            start = time.time()
            Q, V, pi, Q_track, pi_track = RL(env.env).q_learning(nS=None, nA=None, gamma=gamma,
                                                                 init_alpha=learning_rate, min_alpha=0.01,
                                                                 alpha_decay_ratio=0.5, init_epsilon=epsilon,
                                                                 min_epsilon=0.1, epsilon_decay_ratio=0.8,
                                                                 n_episodes=episodes)
            runtime = time.time() - start
            policy = np.argmax(Q, axis=1)

            # plot policy map
            n_states = env.env.observation_space.n
            new_pi = list(map(lambda x: pi(x), range(n_states)))
            s = int(math.sqrt(n_states))
            savepath = 'plots/Fronzen-QL-Policy' + str(map_size) + ' gamma=' + str(gamma) + '.png'
            plots.Plots.grid_world_policy_plot(np.array(new_pi), 'QL:Grid World Policy with gamma=' + str(gamma),
                                               map_size, savepath)
            # plot convergence
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
            plt.savefig('plots/Frozenlake-QL-converge' + str(map_size) +'.png')
            plt.close()

            # test
            env.reset(seed=seed_value)
            test_scores = TestEnv.test_env(env=env.env, render=False, user_input=False, pi=pi, n_iters=test_iter)
            succ = np.count_nonzero(test_scores > 0)
            fail = np.count_nonzero(test_scores <= 0)
            #print('succ=', succ)

            with open(metric_filename, 'a') as fp:
                fp.write(f'***********************************' + '\n')
                fp.write('Frozen Lake Size: ' + str(map_size) + ' by ' + str(map_size) + '\n')
                fp.write(
                    f'Qlearning with learning rate = ' + str(learning_rate) + ' epsilon = ' + str(
                        epsilon) + ' gamma = ' + str(gamma) + ' episodes = ' + str(episodes) + '\n')
                fp.write(f'Runtime = ' + str(runtime) + '\n')
                fp.write(f'The agent reaches the goal ' + str(succ) + ' times, and fails ' + str(fail) + ' times' + '\n')

        '''
        Varying seeds
        '''
        succ_list = []
        fail_list = []

        for random_seed in random_integers:
            env.reset(seed=random_seed)
            test_scores = TestEnv.test_env(env=env.env, render=False, user_input=False, pi=pi, n_iters=test_iter)
            succ = np.count_nonzero(test_scores > 0)
            fail = np.count_nonzero(test_scores <= 0)

            succ_list.append(succ)
            fail_list.append(fail)

        with open(metric_filename, 'a') as fp:
            fp.write(f'---------------------------' + '\n')
            fp.write(f'With 100 random seeds: \n')
            fp.write(f'Succ range from ' + str(min(succ_list)) + ' to ' + str(max(succ_list)) + '\n')
            fp.write(f'Fail range from ' + str(min(fail_list)) + ' to ' + str(max(fail_list)) + '\n')
            fp.write(f'\n')
            fp.write(f'Mean Succ = ' + str(np.mean(succ_list)) + '\n')
            fp.write(f'Mean Fail = ' + str(np.mean(fail_list)) + '\n')

    # This list present gamma, learning_rate and epsilon
    perameter_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    '''
    Step2 Learning rate change
    '''
    max_Q = {}
    runtime_list_lr = []
    for map_size in [4, 16]:
        if map_size == 4:
            seed_value = 123
            gamma = 0.9
            epsilon = 0.7
            proba_frozen = 0.9
            episodes = 1000
        elif map_size == 16:
            seed_value = 100
            gamma = 0.8
            epsilon = 0.5
            proba_frozen = 0.98
            episodes = 2000
        np.random.seed(seed_value)
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery,
                       desc=generate_random_map(size=map_size, p=proba_frozen), )
        for i in range(len(perameter_list)):
            env.reset(seed=seed_value)
            planner_instance = Planner(env.P)
            start = time.time()
            Q, V, pi, Q_track, pi_track = RL(env.env).q_learning(nS=None, nA=None, gamma=gamma,
                                                                 init_alpha=perameter_list[i], min_alpha=0.01,
                                                                 alpha_decay_ratio=0.5, init_epsilon=epsilon,
                                                                 min_epsilon=0.1, epsilon_decay_ratio=0.9,
                                                                 n_episodes=episodes)
            runtime_list_lr.append(time.time() - start)
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
        plt.legend(fontsize=14, loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/Frozenlake-QL-learningrate' +str(map_size)+'.png')
        plt.close()
    '''
    Step2 Epsilon rate change
    '''
    max_Q = {}
    runtime_list_ep = []
    for map_size in [4, 16]:
        if map_size == 4:
            seed_value = 123
            gamma = 0.9
            learning_rate = 0.8
            proba_frozen = 0.9
            episodes = 1000
        elif map_size == 16:
            seed_value = 100
            gamma = 0.8
            learning_rate = 0.7
            proba_frozen = 0.98
            episodes = 2000
        np.random.seed(seed_value)
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery,
                       desc=generate_random_map(size=map_size, p=proba_frozen), )
        for i in range(len(perameter_list)):
            env.reset(seed=seed_value)
            planner_instance = Planner(env.P)
            start = time.time()
            Q, V, pi, Q_track, pi_track = RL(env.env).q_learning(nS=None, nA=None, gamma=gamma,
                                                                 init_alpha=learning_rate, min_alpha=0.01,
                                                                 alpha_decay_ratio=0.5, init_epsilon=perameter_list[i],
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
        plt.legend(fontsize=14, loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/Frozenlake-QL-epsilon' + str(map_size) + '.png')
        plt.close()
    '''
    Step3 Gamma change
    '''
    max_Q = {}
    runtime_list_ga = []
    for map_size in [4, 16]:
        if map_size == 4:
            seed_value = 123
            learning_rate = 0.8
            epsilon = 0.7
            proba_frozen = 0.9
            episodes = 1000
        elif map_size == 16:
            seed_value = 100
            learning_rate = 0.7
            epsilon = 0.5
            proba_frozen = 0.98
            episodes = 2000
        np.random.seed(seed_value)
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery,
                       desc=generate_random_map(size=map_size, p=proba_frozen), )
        for i in range(len(perameter_list)):
            env.reset(seed=seed_value)
            planner_instance = Planner(env.P)
            start = time.time()
            Q, V, pi, Q_track, pi_track = RL(env.env).q_learning(nS=None, nA=None, gamma=perameter_list[i],
                                                                 init_alpha=learning_rate, min_alpha=0.01,
                                                                 alpha_decay_ratio=0.5, init_epsilon=epsilon,
                                                                 min_epsilon=0.1, epsilon_decay_ratio=0.9,
                                                                 n_episodes=episodes)
            runtime_list_ga.append(time.time() - start)
            # Track the maximum Q-value for each iteration
            max_Q[perameter_list[i]] = np.max(Q_track, axis=(1, 2))

        # Plot gamma change
        plt.figure(figsize=(8, 5))

        for gamma, max_q_values in max_Q.items():
            plt.plot(max_q_values, label=f'gamma={gamma}')

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(0, 1.2)
        plt.title('Qlearning with learning rate=' + str(learning_rate) + ' epsilon=' + str(epsilon))
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Maximum Q-Value', fontsize=20)
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/Frozenlake-QL-gamma' + str(map_size) + '.png')
        plt.close()

    for map_size in [4, 16]:
        # Plot time difference for lr,ep,ga
        N = len(perameter_list)
        plt.figure(figsize=(8, 5))
        if map_size == 4:
            plt.plot(perameter_list, runtime_list_lr[:N], marker='o', linestyle='-', color='r', label='Learning Rate')
            plt.plot(perameter_list, runtime_list_ep[:N], marker='o', linestyle='-', color='b', label='Epsilon')
            plt.plot(perameter_list, runtime_list_ga[:N], marker='o', linestyle='-', color='g', label='Gamma')
        elif map_size == 16:
            plt.plot(perameter_list, runtime_list_lr[N:], marker='o', linestyle='-', color='r', label='Learning Rate')
            plt.plot(perameter_list, runtime_list_ep[N:], marker='o', linestyle='-', color='b', label='Epsilon')
            plt.plot(perameter_list, runtime_list_ga[N:], marker='o', linestyle='-', color='g', label='Gamma')
        plt.xticks(perameter_list, fontsize=15)
        plt.yticks(fontsize=18)
        plt.title('Runtime for Different Parameters')
        plt.xlabel('Parameter Value', fontsize=20)
        plt.ylabel('Runtime (seconds)', fontsize=20)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/Frozenlake-QL-lr-ep-ga-time' + str(map_size) + '.png')
        plt.close()






