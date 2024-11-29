""""""
"""OMSCS2023FALL-P1: Supervised Learning 		  	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

'''
5 ML algorithms
'''
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as NN
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, classification_report, confusion_matrix

'''
#################################################
            Alg1 DecisionTreeClassifier
#################################################
'''
def DecisionTree(X_train, X_test, Y_train, filename, gridsearch, scoring):
    '''
    '''
    '''
    # 2.1 Optimization (grid-search)
    '''
    if gridsearch:
        # Define the hyperparameter grid
        param_grid = {
        'max_depth': np.arange(1, 21),
        'min_samples_leaf': np.arange(1, 31)
        }
        # Create GridSearchCV with the model and hyperparameter grid
        grid_search = GridSearchCV(estimator=DT(random_state=42), param_grid=param_grid, cv=5, scoring=scoring)
        # Fit to data
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters and model
        best_params = grid_search.best_params_
        print(best_params)
        best_hyperpara = [best_params['max_depth'], best_params['min_samples_leaf']]
    else:
        if filename == 'Data_Diabetes':
            best_hyperpara = [13, 1]
        elif filename == 'Data_Sleepiness':
            best_hyperpara = [8, 17]
    '''
    # 2.2 Best fit model
    '''
    # Retrain the model with the best super-parameter
    clf = DT(max_depth=best_hyperpara[0],min_samples_leaf=best_hyperpara[1],random_state=42)
    best_estimator = clf

    # Train on train(X,Y)
    start_time = time.time()
    clf.fit(X_train, Y_train)
    time_train = time.time() - start_time

    # Test on test(X)
    start_time = time.time()
    Y_pred_test = clf.predict(X_test)
    time_test = time.time() - start_time

    return best_estimator, best_hyperpara, Y_pred_test, time_train, time_test
'''
#################################################
            Alg2 GradientBoostingClassifier
#################################################
'''
def Boosting(X_train, X_test, Y_train, filename, gridsearch, scoring):
    '''
    '''
    '''
    # 2.1 Optimization (grid-search)
    '''
    if gridsearch:
        # Define the hyperparameter grid
        param_grid = {
        'min_samples_leaf': np.arange(1, 21),
        'max_depth': np.arange(1, 31),
        'n_estimators': [10],
        'learning_rate': [0.01, 0.1, 1]
        }
        # Create GridSearchCV with the model and hyperparameter grid
        grid_search = GridSearchCV(estimator=GB(random_state=42), param_grid=param_grid, cv=5, scoring=scoring)

        # Fit to data
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters and model
        best_params = grid_search.best_params_
        print(best_params)
        best_hyperpara = [best_params['max_depth'], best_params['min_samples_leaf'], best_params['n_estimators'], best_params['learning_rate']]
    else:
        if filename == 'Data_Diabetes':
            best_hyperpara = [14, 13, 10, 1]
        elif filename == 'Data_Sleepiness':
            best_hyperpara = [11, 12, 20, 0.1]
    '''
    # 2.2 Best fit model
    '''
    # Retrain the model with the best super-parameter
    clf = GB(max_depth=best_hyperpara[0], min_samples_leaf=best_hyperpara[1], n_estimators=best_hyperpara[2], learning_rate=best_hyperpara[3], random_state=42)
    best_estimator = clf

    # Train on train(X,Y)
    start_time = time.time()
    clf.fit(X_train, Y_train)
    time_train = time.time() - start_time

    # Test on test(X)
    start_time = time.time()
    Y_pred_test = clf.predict(X_test)
    time_test = time.time() - start_time

    return best_estimator, best_hyperpara, Y_pred_test, time_train, time_test
'''
#################################################
            Alg3 KNeighborsClassifier
#################################################
'''
def KNeighbors(X_train, X_test, Y_train, filename, gridsearch, scoring):
    '''
    '''
    '''
    # 2.1 Optimization (grid-search)
    '''
    if gridsearch:
        # Define the hyperparameter grid
        param_grid = {
        'n_neighbors': np.arange(1, 100),
        'weights': ['uniform', 'distance'],
        'p': [1,2,3,4,5]
        }
        # Create GridSearchCV with the model and hyperparameter grid
        grid_search = GridSearchCV(estimator=KNN(), param_grid=param_grid, cv=5, scoring=scoring)

        # Fit to data
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters and model
        best_params = grid_search.best_params_
        print(best_params)
        best_hyperpara = [best_params['n_neighbors'], best_params['weights'], best_params['p']]
    else:
        if filename == 'Data_Diabetes':
            best_hyperpara = [1, 'uniform', 2]
        elif filename == 'Data_Sleepiness':
            best_hyperpara = [11, 'distance', 1]
    '''
    # 2.2 Best fit model
    '''
    # Retrain the model with the best super-parameter
    clf = KNN(n_neighbors=best_hyperpara[0], weights=best_hyperpara[1], p=best_hyperpara[2])
    best_estimator = clf

    # Train on train(X,Y)
    start_time = time.time()
    clf.fit(X_train, Y_train)
    time_train = time.time() - start_time

    # Test on test(X)
    start_time = time.time()
    Y_pred_test = clf.predict(X_test)
    time_test = time.time() - start_time

    return best_estimator, best_hyperpara, Y_pred_test, time_train, time_test
'''
#################################################
            Alg4 NeuralNetworks
#################################################
'''
def NeuralNetworks(X_train, X_test, Y_train, filename, gridsearch, scoring):
    '''
    '''
    '''
    # 2.1 Optimization (grid-search)
    '''
    if gridsearch:
        # Define the hyperparameter grid
        param_grid = {
#        'hidden_layer_sizes': [(5,), (10,), (15,), (20,), (5,5), (10,10), (10,20), (15,15), (20,20)],
        'hidden_layer_sizes': [(size, size) for size in np.arange(1, 101)],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
        }

        # Create GridSearchCV with the model and hyperparameter grid
        grid_search = GridSearchCV(estimator=NN(random_state=42, max_iter=2000, activation='logistic'), param_grid=param_grid, cv=5, scoring=scoring)

        # Fit to data
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters and model
        best_params = grid_search.best_params_
        print(best_params)
        best_hyperpara = [best_params['hidden_layer_sizes'], best_params['alpha'], best_params['learning_rate_init']]
    else:
        if filename == 'Data_Diabetes':
            best_hyperpara = [(20,), 0.001, 0.01]
        elif filename == 'Data_Sleepiness':
            max_iter = 300
            best_hyperpara = [(62,), 0.0001, 0.1]
    '''
    # 2.2 Best fit model
    '''
    # Retrain the model with the best super-parameter
    clf = NN(hidden_layer_sizes=best_hyperpara[0], alpha=best_hyperpara[1], learning_rate_init=best_hyperpara[2], activation='logistic', random_state=42, max_iter=max_iter)
    best_estimator = clf

    # Train on train(X,Y)
    start_time = time.time()
    clf.fit(X_train, Y_train)
    time_train = time.time() - start_time

    # Test on test(X)
    start_time = time.time()
    Y_pred_test = clf.predict(X_test)
    time_test = time.time() - start_time

    # Loss curve
    clf.fit(X_train, Y_train)
    loss_curve_train = clf.loss_curve_

    return best_estimator, best_hyperpara, Y_pred_test, time_train, time_test, loss_curve_train
'''
#################################################
        Alg5 SVM (Support Vector Machines)
#################################################
'''
def SVM(X_train, X_test, Y_train, filename, gridsearch, scoring):
    '''
    '''
    '''
    # 2.1 Optimization (grid-search)
    '''
    if gridsearch:
        # Define the hyperparameter grid
        param_grid = {
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.1, 1, 10, 100, 1000]
#        'degree': [1,2,3,4]
        }
        # Create GridSearchCV with the model and hyperparameter grid
        grid_search = GridSearchCV(estimator=SVC(random_state=42), param_grid=param_grid, cv=5, scoring=scoring)

        # Fit to data
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters and model
        best_params = grid_search.best_params_
        print(best_params)
        if best_params['kernel'] == 'poly':
            best_hyperpara = [best_params['kernel'], best_params['C'], best_params['gamma'], best_params['degree']]
        else:
            best_hyperpara = [best_params['kernel'], best_params['C'], best_params['gamma']]
    else:
        if filename == 'Data_Diabetes':
            best_hyperpara = ['rbf', 1, 1000]
        elif filename == 'Data_Sleepiness':
            best_hyperpara = ['rbf', 10, 10]
    '''
    # 2.2 Best fit model
    '''
    # Retrain the model with the best super-parameter
    if best_hyperpara[0] == 'poly':
        clf = SVC(kernel=best_hyperpara[0], C=best_hyperpara[1], gamma=best_hyperpara[2], degree=best_hyperpara[3], random_state=42)
    else:
        clf = SVC(kernel=best_hyperpara[0], C=best_hyperpara[1], gamma=best_hyperpara[2], random_state=42)
    best_estimator = clf

    # Train on train(X,Y)
    start_time = time.time()
    clf.fit(X_train, Y_train)
    time_train = time.time() - start_time

    # Test on test(X)
    start_time = time.time()
    Y_pred_test = clf.predict(X_test)
    time_test = time.time() - start_time

    return best_estimator, best_hyperpara, Y_pred_test, time_train, time_test