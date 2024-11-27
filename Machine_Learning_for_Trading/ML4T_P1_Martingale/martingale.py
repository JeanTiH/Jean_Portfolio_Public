""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Juejing Han
GT User ID: jhan446
GT ID: 903845311
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  	  		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt
  		  	   		  	  		  		  		    	 		 		   		 		  
def author():
    return "jhan446"  # The GT username of the student

def gtid():
    return 903845311  # The GT ID of the student
"""
---------------------------------------------------------------------
                                SPIN
---------------------------------------------------------------------
"""
def get_spin_result(win_prob):
    result = False  		  	   		  	  		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		  	  		  		  		    	 		 		   		 		  
        result = True  		  	   		  	  		  		  		    	 		 		   		 		  
    return result
"""
---------------------------------------------------------------------
                             TEST CODE
---------------------------------------------------------------------
"""
def test_code():
    win_prob = 18/38  # Betting on the black: 18 black-numbers among 38 numbers-in-total
    np.random.seed(gtid())  # do this only once
#    print(get_spin_result(win_prob))  # test the roulette spin
    Fig(win_prob)
"""
---------------------------------------------------------------------
                                SIMULATOR 
---------------------------------------------------------------------
"""
def simulator(win_prob,infinite_bankroll):
# Initialize variales
    bet_count = 0
    episode_winnings = 0
    bet_amount = 1
    bankroll = 256
# Initialize a NumPy array (winnings_output) to store the results of simulations
    winnings_output = np.full((1001), 80)
# Simulating
    while episode_winnings < 80 and bet_count < 1001:
        winnings_output[bet_count] = episode_winnings  # So winnings_output[0]=0
        bet_count = bet_count + 1
        won = get_spin_result(win_prob)
        if won:
            episode_winnings = episode_winnings + bet_amount
            bet_amount = 1  # Reset bet_amount = 1 after each win
        else:
            episode_winnings = episode_winnings - bet_amount
            bet_amount *= 2 # Double bet_amount after each loss
# ------------------------------------------------------------------------------
#                               NO INFINITE BANKROLl
# ------------------------------------------------------------------------------
            if not infinite_bankroll:                         # No infinite bankroll
                if episode_winnings + bankroll < bet_amount:  # No enough money left
                    if episode_winnings + bankroll == 0:      # Lost all bankroll
                        winnings_output[bet_count:] = -bankroll # Fix the rest winnings at -bankroll
                        return winnings_output
                    bet_amount = bankroll + episode_winnings
#------------------------------------------------------------------------------
    return winnings_output
# End of simulating
"""
--------------------------------------------------------------------
                        EXPERIMENT & PLOTTING
--------------------------------------------------------------------
"""
def Fig(win_prob):
# 1. Fig 1 Result & Plotting
# Initialize a NumPy array (output_array) to store the results of simulations
    output_array = np.full((1000, 1001), 0)

    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Winnings ($)')
    plt.title('10 Episodes of Trials with Infinite Bankroll')

    for episode in range(10):
        plt.plot(simulator(win_prob,True),label='Episode'+str(episode+1))
        plt.legend()
    plt.savefig('Fig1.png')
    plt.close()
# 2. Fig 2 & 3 Result & Plotting
# 2.1 Fig 2
    for episode in range(1000):
        output_array[episode] = simulator(win_prob,True)

    Mean = np.mean(output_array,axis=0)
    Std = np.std(output_array,axis=0)
    Mean_plu_std = Mean + Std
    Mean_min_std = Mean - Std

    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Winnings ($)')
    plt.title('1000 Episodes of Trials with Infinite Bankroll')

    plt.plot(Mean,label='Mean')
    plt.plot(Mean_plu_std,label='Mean Plus Std')
    plt.plot(Mean_min_std,label='Mean Minus Std')
    plt.legend()
    plt.savefig('Fig2.png')
    plt.close()
# 2.2 Fig 3
    Median = np.median(output_array,axis=0)
    Median_plu_std = Median + Std
    Median_min_std = Median - Std

    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Winnings ($)')
    plt.title('1000 Episodes of Trials with Infinite Bankroll')

    plt.plot(Median,label='Median')
    plt.plot(Median_plu_std,label='Median Plus Std')
    plt.plot(Median_min_std,label='Median Minus Std')
    plt.legend()
    plt.savefig('Fig3.png')
    plt.close()
# 3. Fig 4 & 5 Result & Plotting
# 3.1 Fig4
    for episode in range(1000):
        output_array[episode] = simulator(win_prob,False)

    Mean = np.mean(output_array,axis=0)
    Std = np.std(output_array,axis=0)
    Mean_plu_std = Mean + Std
    Mean_min_std = Mean - Std

    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Winnings ($)')
    plt.title('1000 Episodes of Trials with $256 Bankroll')

    plt.plot(Mean,label='Mean')
    plt.plot(Mean_plu_std,label='Mean Plus Std')
    plt.plot(Mean_min_std,label='Mean Minus Std')
    plt.legend()
    plt.savefig('Fig4.png')
    plt.close()
# 3.2 Fig 5
    Median = np.median(output_array,axis=0)
    Median_plu_std = Median + Std
    Median_min_std = Median - Std

    plt.xlim(0,300)
    plt.ylim(-256,100)
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Winnings ($)')
    plt.title('1000 Episodes of Trials with $256 Bankroll')

    plt.plot(Median,label='Median')
    plt.plot(Median_plu_std,label='Median Plus Std')
    plt.plot(Median_min_std,label='Median Minus Std')
    plt.legend()
    plt.savefig('Fig5.png')
    plt.close()
"""
--------------------------------------------------------------------
                            TESTING
--------------------------------------------------------------------
"""
if __name__ == "__main__":
    test_code()
