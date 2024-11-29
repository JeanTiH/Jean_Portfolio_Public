""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
import random as rand
import numpy as np
"""
-------------------------------------------------------------------------------  		  	   		  	  		  		  		    	 		 		   		 		  
                            This is a Q learner object.  		  	   		  	  		  		  		    	 		 		   		 		  
------------------------------------------------------------------------------- 
Parameters:
    1. num_states (int): The number of states to consider. 		  	   		  	  		  		  		    	 		 		   		 		  
    2. num_actions (int): The number of actions available..  		  	   		  	  		  		  		    	 		 		   		 		  
    3. alpha (float): The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    4. gamma (float): The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    5. rar (float): Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    6. radr (float): Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  		  		  		    	 		 		   		 		  
    7. dyna (int): The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    8. verbose (bool): If “verbose” is True, your code can print out information for debugging.
------------------------------------------------------------------------------- 		  	   		  	  		  		  		    	 		 		   		 		  
"""
class QLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  

    def author(self):
        return 'jhan446'

    def __init__(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		  	  		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		  	  		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		  	  		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		  	  		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		  	  		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		  	  		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		  	  		  		  		    	 		 		   		 		  
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        # Initialize s (state), a (action), q (Q-table)
        self.s = 0
        self.a = 0
        self.q = np.zeros((num_states, num_actions))

        # Dyna index (always pick a & s from experience, so no need to initialize T/Tc with 0.0000001)
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.Tc = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))

        # History s, a for Dyna
        self.history_s = []
        self.history_a = []
    """
    -------------------------------------------------------  		  	   		  	  		  		  		    	 		 		   		 		  
         Update the state without updating the Q-table  		  	   		  	  		  		  		    	 		 		   		 		  
    -------------------------------------------------------
    Parameter: 
        s (int): The new state  		  	   		  	  		  		  		    	 		 		   		 		  
    Return:
        action (int): The selected action
    -------------------------------------------------------		  	   		  	  		  		  		    	 		 		   		 		  
    """
    def querysetstate(self, s):
        self.s = s
        random = rand.random()
        if random < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q[self.s])
        self.a = action
        return action
    """ 
    -------------------------------------------------------	   
            Update the Q table and return an action  		  	   		  	  		  		  		    	 		 		   		 		  
    -------------------------------------------------------	
    Pparameters:
        1. s_prime (int): The new state 		  	   		  	  		  		  		    	 		 		   		 		  
        2. r (float): The immediate reward		  	   		  	  		  		  		    	 		 		   		 		  
    Return: 
        action (int): The selected action 
    -------------------------------------------------------			  	   		  	  		  		  		    	 		 		   		 		  
    """
    def query(self, s_prime, r):
        # Update q (Q-table)
        self.q[self.s,self.a]=(1.0-self.alpha)*self.q[self.s,self.a]+self.alpha*(r+self.gamma*np.max(self.q[s_prime, :]))
        # Store history data for Dyna
        self.history_s.append(self.s)
        self.history_a.append(self.a)
        """ 
        -----------------------------------------	   
                            Dyna  		  	   		  	  		  		  		    	 		 		   		 		  
        -----------------------------------------			  	   		  	  		  		  		    	 		 		   		 		  
        """
        # Implement Dyna to update q (Q-table)
        if self.dyna > 0:
            # Calculate T (Transition Matrix) and R (Reward Matrix)
            self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1
            self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :] / np.sum(self.Tc[self.s, self.a, :])
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            for i in range(self.dyna):
                # Randomly select s and a from experience (history)
                num = rand.randint(0, len(self.history_s) - 1)
                sDyna = self.history_s[num]
                aDyna = self.history_a[num]
                # Get r from R
                rDyna = self.R[sDyna, aDyna]
                # Pick s_prime from T
                s_primeDyna = np.argmax(self.T[sDyna, aDyna, :])
                # Update Q
                self.q[sDyna,aDyna]=(1.0-self.alpha)*self.q[sDyna,aDyna]+self.alpha*(rDyna+self.gamma*np.max(self.q[s_primeDyna]))

        # Determine next action
        random = rand.random()
        if random < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q[s_prime])
        # Update rar (random action rate), s (state), a (action)
        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action
        return action