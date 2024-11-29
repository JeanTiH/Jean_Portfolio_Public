""""""
"""OMSCS2023FALL-P2: Randomized Optimization	  	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import ML_TSP as TSP
import ML_Knapsack as KNS
import ML_Flipflop as FF

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 10.5, 'axes.labelsize': 12})
'''
----------------------------------------------------------------------
                    Plot Comparison of 3 Problem Sizes
----------------------------------------------------------------------
'''
def plot_3size(problem, sizes, fitness_RHC, fitness_SA, fitness_GA, fitness_MC):
    if problem == 'TSP':
        y_label = 'Negative Total Distance'
    elif problem == 'Flipflop':
        y_label = 'Count of Transitions'
    elif problem == 'Knapsack':
        y_label = 'Total Values'

    # Plot Fitness for 3 sizes
    plt.scatter(sizes, fitness_RHC, label='RHC', color='g')
    plt.scatter(sizes, fitness_SA, label='SA', color='b')
    plt.scatter(sizes, fitness_GA, label='GA', color='r')
    plt.scatter(sizes, fitness_MC, label='MC', color='orange')

    plt.title(problem + ' Fitness vs. Problem Size')
    plt.xlabel('Problem Size')
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/' + problem + '_Fitness_Size.png')
    plt.close()
'''
----------------------------------------------------------------------
                            Run Algorithm
----------------------------------------------------------------------
'''
if __name__ == "__main__":
    #random_seeds = [50, 53, 60, 70, 83]
    state = None    # True: output state for each run; False: no output
    random_seeds = [42, 88, 155, 230, 350]
    '''
    TSP
    '''
    problem = 'TSP'
    sizes = [15, 30, 50]

    fitness_RHC = []
    fitness_SA = []
    fitness_GA = []
    fitness_MC = []
    for size in sizes:
        f_RHC, f_SA, f_GA, f_MC = TSP.postprocess(size, random_seeds, problem, state)
        fitness_RHC.append(f_RHC)
        fitness_SA.append(f_SA)
        fitness_GA.append(f_GA)
        fitness_MC.append(f_MC)

    plot_3size(problem, sizes, fitness_RHC, fitness_SA, fitness_GA, fitness_MC)
    '''
    Flipflop
    '''
    problem = 'Flipflop'
    sizes = [30, 60, 90]

    fitness_RHC = []
    fitness_SA = []
    fitness_GA = []
    fitness_MC = []
    for size in sizes:
        f_RHC, f_SA, f_GA, f_MC = FF.postprocess(size, random_seeds, problem, state)
        fitness_RHC.append(f_RHC)
        fitness_SA.append(f_SA)
        fitness_GA.append(f_GA)
        fitness_MC.append(f_MC)

    plot_3size(problem, sizes, fitness_RHC, fitness_SA, fitness_GA, fitness_MC)
    '''
    Knapsack
    '''
    problem = 'Knapsack'
    sizes = [15, 30, 50]

    fitness_RHC = []
    fitness_SA = []
    fitness_GA = []
    fitness_MC = []
    for size in sizes:
        f_RHC, f_SA, f_GA, f_MC = KNS.postprocess(size, random_seeds, problem, state)
        fitness_RHC.append(f_RHC)
        fitness_SA.append(f_SA)
        fitness_GA.append(f_GA)
        fitness_MC.append(f_MC)

    plot_3size(problem, sizes, fitness_RHC, fitness_SA, fitness_GA, fitness_MC)