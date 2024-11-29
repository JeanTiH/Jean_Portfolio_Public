""""""
"""OMSCS2023FALL-P4: Reinforcement Learning   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

# some functions are adapted from the link below and modified
# https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
# https://github.com/sudharsan13296/Deep-Reinforcement-Learning-With-Python/blob/master/03.%20Bellman%20Equation%20and%20Dynamic%20Programming
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

import os
import random
from algorithms.planner import Planner
from gym.envs.toy_text.frozen_lake import generate_random_map
import time
import math
from examples.test_env import TestEnv
import plots

'''
1. Value Iteration
'''
def value_iteration(env, gamma, theta, n_iters):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
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
        converge_iter = i + 1

    return value_table, cumulative_rewards, delta_values, converge_iter
'''
1. Policy Iteration
'''
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
            #print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        old_policy = new_policy
        converge_iter = i + 1
    return new_policy, V, converge_iter
'''
3. Get Policy from V
'''
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
'''
4. Get Score
'''
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
                # Reach the target
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
    episodes = 1000
    proba_frozen = 0.9
    is_slippery = True
    '''
    Map layout
    '''
    seed_value = 123
    np.random.seed(seed_value)
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, desc=generate_random_map(size=4, p=proba_frozen), )
    # Convert bytes to strings and print the layout of the FrozenLake environment
    print("4 by 4 FrozenLake Layout:")
    for row in env.desc:
        print("".join([cell.decode('utf-8') for cell in row]))

    seed_value = 100
    np.random.seed(seed_value)
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, desc=generate_random_map(size=16, p=proba_frozen), )
    # Convert bytes to strings and print the layout of the FrozenLake environment
    print("16 by 16 FrozenLake Layout:")
    for row in env.desc:
        print("".join([cell.decode('utf-8') for cell in row]))

    metric_filename = 'VIPI-frozenlake.txt'
    if os.path.exists(metric_filename):
        os.remove(metric_filename)

    # env = gym.make('FrozenLake8x8-v1')
    '''
    Step1 VI/PI test
    '''
    map_sizes = [4, 16]
    for i, map_size in enumerate(map_sizes):
        if map_size == 4:
            seed_value = 123
            test_iter = 1000
        elif map_size == 16:
            seed_value = 100
            test_iter = 100
        np.random.seed(seed_value)
        env = gym.make("FrozenLake-v1", is_slippery=is_slippery,
                       desc=generate_random_map(size=map_size, p=proba_frozen), )

        '''
        Value Iteration test
        '''
        for gamma in [0.1, 0.9]:
            env.reset(seed=seed_value)
            planner_instance = Planner(env.P)
            start = time.time()
            V, V_track, pi = planner_instance.value_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
            runtime = time.time() - start
            value_table, cumulative_rewards, delta_values, converge_iter = value_iteration(env, gamma, theta, n_iters)

            # plot policy map
            n_states = env.env.observation_space.n
            new_pi = list(map(lambda x: pi(x), range(n_states)))
            s = int(math.sqrt(n_states))
            savepath = 'plots/Fronzen-VI-Policy' + str(map_size) + ' gamma=' + str(gamma) + '.png'
            plots.Plots.grid_world_policy_plot(np.array(new_pi), 'VI:Grid World Policy with gamma=' + str(gamma),
                                               map_size, savepath)
            # plot convergence
            plt.figure(figsize=(8, 5))
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.plot(cumulative_rewards, color='b', label='Cumulative Rewards')
            plt.xlabel('Iteration', fontsize=20)
            plt.ylabel('Cumulative Rewards', fontsize=20)
            plt.grid(True)
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            ax2 = plt.twinx()
            plt.yticks(fontsize=18)
            ax2.plot(delta_values, color='g', label='Delta')
            plt.title('Value Iteration Map_size ' + str(map_size) + ' with gamma=' + str(gamma))
            ax2.set_ylabel('Delta', fontsize=20)
            ax2.grid(False)
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=18)
            plt.tight_layout()
            plt.savefig('plots/Fronzen-VI-converge' + str(map_size) + '-gamma' + str(gamma) + '.png')
            plt.close()

            # test
            env.reset(seed=seed_value)
            test_scores = TestEnv.test_env(env=env.env, render=False, user_input=False, pi=pi, n_iters=test_iter)
            succ = np.count_nonzero(test_scores > 0)
            fail = np.count_nonzero(test_scores <= 0)

            with open(metric_filename, 'a') as fp:
                fp.write(f'***********************************' + '\n')
                fp.write('Frozen Lake Size: ' + str(map_size) + ' by ' + str(map_size) + '\n')
                fp.write(f'Value iteration with gamma = ' + str(gamma) + ' theta = ' + str(theta) + '\n')
                fp.write(f'Runtime = ' + str(runtime) + '\n')
                fp.write(f'Converge iteration = ' + str(converge_iter) + '\n')
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

        '''
        Policy Iteration test
        '''
        for gamma in [0.1, 0.9]:
            env.reset(seed=seed_value)
            planner_instance = Planner(env.P)
            start = time.time()
            V, V_track, pi = planner_instance.policy_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
            runtime = time.time() - start
            policy, V_iteration, converge_iter = policy_iteration(env, gamma, theta, n_iters)

            # plot policy map
            n_states = env.env.observation_space.n
            new_pi = list(map(lambda x: pi(x), range(n_states)))
            s = int(math.sqrt(n_states))
            savepath = 'plots/Fronzen-PI-Policy' + str(map_size) + ' gamma=' + str(gamma) + '.png'
            plots.Plots.grid_world_policy_plot(np.array(new_pi), 'PI:Grid World Policy with gamma=' + str(gamma),
                                               map_size, savepath)

            # plot convergence
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
            plt.savefig('plots/Fronzen-PI-converge' + str(map_size) + '-gamma' + str(gamma) + '.png')
            plt.close()

            # test
            env.reset(seed=seed_value)
            test_scores = TestEnv.test_env(env=env.env, render=False, user_input=False, pi=pi, n_iters=test_iter)
            succ = np.count_nonzero(test_scores > 0)
            fail = np.count_nonzero(test_scores <= 0)

            with open(metric_filename, 'a') as fp:
                fp.write(f'***********************************' + '\n')
                fp.write('Frozen Lake Size: ' + str(map_size) + ' by ' + str(map_size) + '\n')
                fp.write(f'Policy iteration with gamma = ' + str(gamma) + ' theta = ' + str(theta) + '\n')
                fp.write(f'Runtime = ' + str(runtime) + '\n')
                fp.write(f'Converge iteration = ' + str(converge_iter) + '\n')
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

        '''
        Step2 Gamma change
        '''
        gamma_list = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        runtime_list_VI = []
        mean_V_list_VI = []
        runtime_list_PI = []
        mean_V_list_PI = []

        for i in range(len(gamma_list)):
            # Value Iteration
            env.reset(seed=seed_value)
            planner_instance = Planner(env.P)
            start = time.time()
            V, V_track, pi = planner_instance.value_iteration(gamma=gamma_list[i], n_iters=n_iters, theta=theta)
            runtime_list_VI.append(time.time() - start)
            mean_V = np.mean(V_track, axis=1)
            # Find the index of the last non-zero value
            last_non_zero_index = np.maximum.reduce(np.where(mean_V != 0, np.arange(mean_V.size), -1))
            # Fill zeros after the last non-zero value with that last non-zero value
            mean_V[last_non_zero_index + 1:] = mean_V[last_non_zero_index]
            mean_V_list_VI.append(mean_V)

            # Policy Iteration
            env.reset(seed=seed_value)
            planner_instance = Planner(env.P)
            start = time.time()
            V, V_track, pi = planner_instance.policy_iteration(gamma=gamma_list[i], n_iters=n_iters, theta=theta)
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
            if map_size == 4:
                plt.plot(np.arange(1, 81), mean_V_list_VI[i][:80], label=f'Gamma={gamma_list[i]}')
            elif map_size == 16:
                plt.plot(np.arange(1, 151), mean_V_list_VI[i][:150], label=f'Gamma={gamma_list[i]}')

        if map_size == 4:
            plt.xticks(np.arange(-10, 81, 5), [str(label) if label not in [-10, -5] else '' for label in np.arange(-10, 81, 5)], fontsize=18)
            plt.yticks(np.arange(0.0, 0.6, 0.1), fontsize=18)
        elif map_size == 16:
            plt.xticks(np.arange(1, 151, 30), fontsize=18)
            plt.yticks(fontsize=18)

        plt.title('Value Iteration: Different Gamma with Map_size ' + str(map_size))
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Mean V', fontsize=20)
        if map_size == 4:
            plt.legend(fontsize=12, loc='best')
        elif map_size == 16:
            plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1, 0.1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/Frozenlake-VI-gamma-' + str(map_size) +'.png')
        plt.close()

        # Plot PI (gamma)
        plt.figure(figsize=(8, 5))
        for i in range(len(gamma_list)):
            if map_size == 4:
                plt.plot(np.arange(1, 21), mean_V_list_PI[i][:20], label=f'Gamma={gamma_list[i]}')
            elif map_size == 16:
                plt.plot(np.arange(1, 11), mean_V_list_PI[i][:10], label=f'Gamma={gamma_list[i]}')

        if map_size == 4:
            plt.xticks(np.arange(1, 21, 2), fontsize=18)
            plt.yticks(fontsize=18)
        elif map_size == 16:
            plt.xticks(np.arange(1, 11, 2), fontsize=18)
            plt.yticks(np.arange(-3, 0.6, 0.5), fontsize=18)
        plt.title('Policy Iteration: Different Gamma with Map_size ' + str(map_size))
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Mean V', fontsize=20)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/Frozenlake-PI-gamma-' + str(map_size) +'.png')
        plt.close()

        # Plot VI PI time difference
        plt.figure(figsize=(8, 5))
        plt.plot(gamma_list, runtime_list_VI, marker='o', linestyle='-', color='b', label='Value Iteration')
        plt.plot(gamma_list, runtime_list_PI, marker='*', linestyle='-', color='g', label='Policy Iteration')

        plt.xticks(gamma_list, fontsize=15)
        plt.yticks(fontsize=18)
        plt.title('Runtime for Different Gamma with Map_size ' + str(map_size))
        plt.xlabel('Gamma Value', fontsize=20)
        plt.ylabel('Runtime (seconds)', fontsize=20)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/Frozenlake-VIPI-gamma-time-' + str(map_size) +'.png')
        plt.close()