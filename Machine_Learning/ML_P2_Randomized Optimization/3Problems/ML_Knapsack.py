""""""
"""OMSCS2023FALL-P2: Randomized Optimization	  	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose
import mlrose_hiive
from mlrose_hiive.runners.rhc_runner import RHCRunner
from mlrose_hiive.runners.mimic_runner import MIMICRunner
from mlrose_hiive.runners.sa_runner import SARunner
from mlrose_hiive.runners.ga_runner import GARunner

import time

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 10.5, 'axes.labelsize': 12})
'''
----------------------------------------------------------------------
                            Knapsack Problem 
----------------------------------------------------------------------
'''
def Knapsack(random_seed, size, state):
    # Store the stats
    if state:
        path = '/Users/jjhan/PycharmProjects/pythonProject/CN/ML1/0Project2/3Problems/stats/Knapsack'
    else:
        path = None
    # 1. Generate Data
    data_seed = 42
    np.random.seed(data_seed)
    weights = np.random.randint(1, 20, size=size)  # Random weights (between 1 and 20)
    values = np.random.randint(1, 30, size=size)
    max_weight_pct = 0.6
    # 3. Define the optimization problem
    fitness = mlrose_hiive.Knapsack(weights=weights, values=values, max_weight_pct=max_weight_pct)
    problem = mlrose_hiive.DiscreteOpt(length=len(values), fitness_fn=fitness, maximize=True)
    # 4. Parameters after tuning
    if size == 15:
        restart_list = [1]
        decay_list = [mlrose.ArithDecay]
        temperature_list = [7]

        population_sizesGA = [20]
        mutation_rates = [0.1]

        population_sizesMC = [45]
        keep_percent_list = [0.3]

    elif size == 30:
        restart_list = [5]
        decay_list = [mlrose.ArithDecay]
        temperature_list = [8]

        population_sizesGA = [40]
        mutation_rates = [0.1]

        population_sizesMC = [80]
        keep_percent_list = [0.35]

    elif size == 50:
        restart_list = [8]
        decay_list = [mlrose.ArithDecay]
        temperature_list = [20]

        population_sizesGA = [60]
        mutation_rates = [0.1]

        population_sizesMC = [220]
        keep_percent_list = [0.3]

    # 4.1 Randomized Hill Climbing (RHC)
    start_time = time.time()
    rhc = RHCRunner(problem=problem, restart_list=restart_list,
                    experiment_name='Knapsack-RHC' + str(size) + '_seed' + str(random_seed), output_directory=path,
                    iteration_list=[2000], seed=random_seed)
    run_stats_RHC, run_curves_RHC = rhc.run()
    time_RHC = time.time() - start_time

    # 4.2 Simulated Annealing (SA)
    # mlrose.ExpDecay or mlrose.ArithDecay

    start_time = time.time()
    sa = SARunner(problem=problem, decay_list=decay_list, temperature_list=temperature_list,
                  experiment_name='Knapsack-SA' + str(size) + '_seed' + str(random_seed),
                  output_directory=path,
                  iteration_list=[2000], seed=random_seed)
    run_stats_SA, run_curves_SA = sa.run()
    time_SA = time.time() - start_time

    # 4.3 Genetic Algorithm (GA)
    start_time = time.time()
    ga = GARunner(problem=problem, population_sizes=population_sizesGA, mutation_rates=mutation_rates,
                  experiment_name='Knapsack-GA' + str(size) + '_seed' + str(random_seed),
                  output_directory=path,
                  iteration_list=[2000], seed=random_seed)
    run_stats_GA, run_curves_GA = ga.run()
    time_GA = time.time() - start_time

    # 4.4 MIMIC (MC)
    start_time = time.time()
    mc = MIMICRunner(problem=problem, population_sizes=population_sizesMC, keep_percent_list=keep_percent_list,
                     experiment_name='Knapsack-MIMIC' + str(size) + '_seed' + str(random_seed),
                     output_directory=path,
                     iteration_list=[2000], seed=random_seed, use_fast_mimic=True)
    run_stats_MC, run_curves_MC = mc.run()
    time_MC = time.time() - start_time

    return run_stats_RHC, run_curves_RHC, run_stats_SA, run_curves_SA, run_stats_GA, run_curves_GA, run_stats_MC, run_curves_MC, time_RHC, time_SA, time_GA, time_MC
'''
----------------------------------------------------------------------
                            Postprocess
----------------------------------------------------------------------
'''
def walltime(filename, problem, size, meantime_RHC, meantime_SA, meantime_GA, meantime_MC):
    with open(filename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{problem}\n')
        fp.write(f'{size}\n')
        fp.write(f'***********************************' + '\n')
        fp.write(f'RHC wall time (s): {meantime_RHC:.5f}\n')
        fp.write(f' SA wall time (s): {meantime_SA:.5f}\n')
        fp.write(f' GA wall time (s): {meantime_GA:.5f}\n')
        fp.write(f' MC wall time (s): {meantime_MC:.5f}\n')

def std_output(filename, problem, size, std_RHC, std_SA, std_GA, std_MC):
    with open(filename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{problem}\n')
        fp.write(f'{size}\n')
        fp.write(f'***********************************' + '\n')
        fp.write(f'RHC std_min: {std_RHC[0]:>7.3f}, std_max: {std_RHC[1]:>7.3f}, std_mean: {std_RHC[2]:>7.3f}, std_median: {std_RHC[3]:>7.3f}\n')
        fp.write(f' SA std_min: {std_SA[0]:>7.3f}, std_max: {std_SA[1]:>7.3f}, std_mean: {std_SA[2]:>7.3f}, std_median: {std_SA[3]:>7.3f}\n')
        fp.write(f' GA std_min: {std_GA[0]:>7.3f}, std_max: {std_GA[1]:>7.3f}, std_mean: {std_GA[2]:>7.3f}, std_median: {std_GA[3]:>7.3f}\n')
        fp.write(f' MC std_min: {std_MC[0]:>7.3f}, std_max: {std_MC[1]:>7.3f}, std_mean: {std_MC[2]:>7.3f}, std_median: {std_MC[3]:>7.3f}\n')

def postprocess(size, random_seeds, problem, state):
    if size == 15:
        restart = 1
    elif size == 30:
        restart = 5
    elif size == 50:
        restart = 8

    curves_RHC_df = pd.DataFrame()
    curves_SA_df = pd.DataFrame()
    curves_GA_df = pd.DataFrame()
    curves_MC_df = pd.DataFrame()
    runtime_RHC = []
    runtime_SA = []
    runtime_GA = []
    runtime_MC = []

    for random_seed in random_seeds:
        run_stats_RHC, run_curves_RHC, run_stats_SA, run_curves_SA, run_stats_GA, run_curves_GA, run_stats_MC, run_curves_MC, \
        time_RHC, time_SA, time_GA, time_MC = Knapsack(random_seed, size, state)
        # Abstract the Best curve from RHC
        start = run_curves_RHC[run_curves_RHC['current_restart'] == restart].index[0]
        run_curves_RHC_best = run_curves_RHC.iloc[start:]
        run_curves_RHC_best.reset_index(drop=True, inplace=True)
        # Append the run_curves DataFrame to the combined DataFrame
        curves_RHC_df = pd.concat([curves_RHC_df, run_curves_RHC_best], axis=1)
        curves_SA_df = pd.concat([curves_SA_df, run_curves_SA], axis=1)
        curves_GA_df = pd.concat([curves_GA_df, run_curves_GA], axis=1)
        curves_MC_df = pd.concat([curves_MC_df, run_curves_MC], axis=1)

        runtime_RHC.append(time_RHC)
        runtime_SA.append(time_SA)
        runtime_GA.append(time_GA)
        runtime_MC.append(time_MC)
    '''
    Output runtime
    '''
    meantime_RHC = sum(runtime_RHC) / len(runtime_RHC)
    meantime_SA = sum(runtime_SA) / len(runtime_SA)
    meantime_GA = sum(runtime_GA) / len(runtime_GA)
    meantime_MC = sum(runtime_MC) / len(runtime_MC)
    filename = problem + '-runtime.txt'
    walltime(filename, problem, size, meantime_RHC, meantime_SA, meantime_GA, meantime_MC)

    # Reset the index
    curves_RHC_df.reset_index(drop=True, inplace=True)
    curves_SA_df.reset_index(drop=True, inplace=True)
    curves_GA_df.reset_index(drop=True, inplace=True)
    curves_MC_df.reset_index(drop=True, inplace=True)
    '''
    Output std
    '''
    # Calculate the std of Fitness
    fitness_std_RHC = curves_RHC_df['Fitness'].std(axis=1).dropna()
    fitness_std_SA = curves_SA_df['Fitness'].std(axis=1).dropna()
    fitness_std_GA = curves_GA_df['Fitness'].std(axis=1).dropna()
    fitness_std_MC = curves_MC_df['Fitness'].std(axis=1).dropna()

    # Calculate the minimum, maximum, mean, and median of fitness_std
    std_RHC = []
    std_RHC.append(np.min(fitness_std_RHC))
    std_RHC.append(np.max(fitness_std_RHC))
    std_RHC.append(np.mean(fitness_std_RHC))
    std_RHC.append(np.median(fitness_std_RHC))

    std_SA = []
    std_SA.append(np.min(fitness_std_SA))
    std_SA.append(np.max(fitness_std_SA))
    std_SA.append(np.mean(fitness_std_SA))
    std_SA.append(np.median(fitness_std_SA))

    std_GA = []
    std_GA.append(np.min(fitness_std_GA))
    std_GA.append(np.max(fitness_std_GA))
    std_GA.append(np.mean(fitness_std_GA))
    std_GA.append(np.median(fitness_std_GA))

    std_MC = []
    std_MC.append(np.min(fitness_std_MC))
    std_MC.append(np.max(fitness_std_MC))
    std_MC.append(np.mean(fitness_std_MC))
    std_MC.append(np.median(fitness_std_MC))

    filename = problem + '-std.txt'
    std_output(filename, problem, size, std_RHC, std_SA, std_GA, std_MC)
    '''
    Plot fitness and fevals
    '''
    # Calculate the mean of Fitness
    fitness_RHC = curves_RHC_df['Fitness'].mean(axis=1)
    fitness_SA = curves_SA_df['Fitness'].mean(axis=1)
    fitness_GA = curves_GA_df['Fitness'].mean(axis=1)
    fitness_MC = curves_MC_df['Fitness'].mean(axis=1)

    # Fitness vs Problem size
    f_RHC = fitness_RHC.mean()
    f_SA = fitness_SA.mean()
    f_GA = fitness_GA.mean()
    f_MC = fitness_MC.mean()

    # Calculate the mean of FEvals
    fevals_RHC = curves_RHC_df['FEvals'].mean(axis=1)
    fevals_SA = curves_SA_df['FEvals'].mean(axis=1)
    fevals_GA = curves_GA_df['FEvals'].mean(axis=1)
    fevals_MC = curves_MC_df['FEvals'].mean(axis=1)

    # Plot Fitness
    plt.plot(fitness_RHC, label='RHC', color='g')
    plt.plot(fitness_SA, label='SA', color='b')
    plt.plot(fitness_GA, label='GA', color='r')
    plt.plot(fitness_MC, label='MC', color='orange')
    plt.title(problem + ' with Size = ' + str(size))
    plt.xlabel('Iterations')
    plt.ylabel('Total Values')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/' + problem + '_' + str(size) + '_Fitness.png')
    plt.close()

    # Plot FEvals
    plt.plot(fevals_RHC/10000, label='RHC', color='g')
    plt.plot(fevals_SA/10000, label='SA', color='b')
    plt.plot(fevals_GA/10000, label='GA', color='r')
    plt.plot(fevals_MC/10000, label='MC', color='orange')
    plt.title(problem + ' with Size = ' + str(size))
    plt.xlabel('Iterations')
    plt.ylabel('FEvals (le5)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/' + problem + '_' + str(size) + '_FEvals.png')
    plt.close()

    # Plot wall clock time
    yticks = np.arange(4)
    plt.figure(figsize=(10, 5))
    plt.barh(yticks, [meantime_RHC, meantime_SA, meantime_GA, meantime_MC])
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels(['RHC', 'SA', 'GA', 'MC'])
    plt.title(problem + ' with Size = ' + str(size))
    plt.xlabel('Wall Clock Time (s)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('plots/' + problem + '_' + str(size) + '_time.png')
    plt.close()

    return f_RHC, f_SA, f_GA, f_MC

