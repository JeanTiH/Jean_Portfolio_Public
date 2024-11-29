""""""
"""OMSCS2023FALL-P2: Randomized Optimization	  	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import time
import matplotlib
matplotlib.use("TkAgg")

import mlrose_hiive
'''
#################################################
               NN Optimization
#################################################
'''
def NN(X_train, X_test, Y_train, filename):
    if filename == 'Data_Diabetes':
        hidden_nodes = [20]

    NN_BP = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, algorithm='gradient_descent',
                                learning_rate=0.0001,
                                activation='relu', curve=True, early_stopping=True, is_classifier=True,
                                bias=True, max_iters=2000, clip_max=5, max_attempts=200, random_state=100)

    NN_RHC = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, algorithm='random_hill_climb',
                                learning_rate=0.8, restarts=6,
                                activation='relu', curve=True, early_stopping=True, is_classifier=True,
                                bias=True, max_iters=2000, clip_max=5, max_attempts=200, random_state=230)

    # mlrose_hiive.GeomDecay(), mlrose_hiive.ExpDecay(), mlrose_hiive.ArithDecay()
    NN_SA = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, algorithm='simulated_annealing',
                                learning_rate=1, schedule=mlrose_hiive.GeomDecay(),
                                activation='relu', curve=True, early_stopping=True, is_classifier=True,
                                bias=True, max_iters=2000, clip_max=5, max_attempts=200, random_state=83)

    NN_GA = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, algorithm='genetic_alg',
                                learning_rate=0.00001, pop_size=40,
                                activation='relu', curve=True, early_stopping=True,  is_classifier=True,
                                bias=True, max_iters=2000, clip_max=5, max_attempts=200, random_state=42)

    time_train = []
    time_test = []
    Y_pred_train = []
    Y_pred_test = []
    loss_curve_train = []

    clfs = [NN_BP, NN_RHC, NN_SA, NN_GA]
    for clf in clfs:
        # Train on train(X,Y)
        start_time = time.time()
        clf.fit(X_train, Y_train)
        time_train0 = time.time() - start_time
        time_train.append(time_train0)

        # Test on train(X,Y)
        Y_pred_train0 = clf.predict(X_train)
        Y_pred_train.append(Y_pred_train0)

        # Test on test(X)
        start_time = time.time()
        Y_pred_test0 = clf.predict(X_test)
        Y_pred_test.append(Y_pred_test0)
        time_test0 = time.time() - start_time
        time_test.append(time_test0)

        # Loss curve for training
        if clf == NN_BP:
            loss_curve_train0 = clf.fitness_curve * (-1.0)
        else:
            loss_curve_train0 = clf.fitness_curve[:,0]
        loss_curve_train.append(loss_curve_train0)

    return clfs, Y_pred_train, Y_pred_test, time_train, time_test, loss_curve_train