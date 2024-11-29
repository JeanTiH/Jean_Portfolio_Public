""""""
"""OMSCS2023FALL-P3: Dimensionality Reduction	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'legend.fontsize': 30})
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection as SRP
from sklearn.manifold import Isomap

from sklearn.neural_network import MLPClassifier as NN
from sklearn.model_selection import GridSearchCV, learning_curve

from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
'''
#################################################
                NeuralNetworks
#################################################
'''
# Learning curve
def lc(clf, X_train, Y_train, train_sizes, cv, scoring, filename, DR_name):
    train_sizes_N, train_scores, test_scores = learning_curve(clf, X_train, Y_train, train_sizes=train_sizes, cv=cv, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(train_sizes_N, train_scores_mean, 'o-', color='b', label='Training')
    plt.plot(train_sizes_N, test_scores_mean, 'o-', color='r', label='Validation')
    plt.fill_between(train_sizes_N, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="b")
    plt.fill_between(train_sizes_N, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="r")
    plt.title(DR_name + ' Learning Curve with ' + filename)
    plt.xlabel('Training Set Size', fontsize=30)
    plt.grid(True)

    if scoring == 'f1':
        plt.ylabel(scoring.capitalize() + ' Score', fontsize=30)
    else:
        plt.ylabel(scoring.capitalize(), fontsize=30)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('plots/NN_DR/' + filename + '-' + DR_name + '_LearningCurve (' + scoring + ').png')
    plt.close()


def NeuralNetworks(X_train, X_test, Y_train, Y_test, filename, DR_name, gridsearch_NN, scoring, train_sizes, cv, NNDR_metric_filename, random_state):
    '''
    '''
    '''
    # Optimization (grid-search)
    '''
    if gridsearch_NN:
        # Define the hyperparameter grid
        param_grid = {
        'hidden_layer_sizes': [(5,), (10,), (15,), (20,), (50,), (5,5), (10,10), (10,20), (15,15), (20,20)],
#        'hidden_layer_sizes': [(size, size) for size in np.arange(1, 101)],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
        }

        # Create GridSearchCV with the model and hyperparameter grid
        grid_search = GridSearchCV(estimator=NN(random_state=random_state, max_iter=2000, activation='logistic'), param_grid=param_grid, cv=5, scoring=scoring)

        # Fit to data
        grid_search.fit(X_train, Y_train)
        # Get the best hyperparameters and model
        best_params = grid_search.best_params_
        print(best_params)
    else:
        if filename == 'Data_Rice':
            if DR_name == 'PCA':
                best_params = {'alpha': 0.001, 'hidden_layer_sizes': (10, 20), 'learning_rate_init': 0.1}
            elif DR_name == 'ICA':
                best_params = {'alpha': 0.1, 'hidden_layer_sizes': (5, 5), 'learning_rate_init': 0.001}
            elif DR_name == 'SRP':
                best_params = {'alpha': 0.01, 'hidden_layer_sizes': (15,), 'learning_rate_init': 0.01}
            elif DR_name == 'IMP':
                best_params = {'alpha': 0.01, 'hidden_layer_sizes': (50,), 'learning_rate_init': 0.1}
            elif DR_name == 'Origin Data':
                best_params = {'alpha': 0.1, 'hidden_layer_sizes': (10, 20), 'learning_rate_init': 0.01}
        elif filename == 'Data_Diabetes':
            if DR_name == 'PCA':
                best_params = {'alpha': 0.0001, 'hidden_layer_sizes': (15, 15), 'learning_rate_init': 0.1}
            elif DR_name == 'ICA':
                best_params = {'alpha': 0.0001, 'hidden_layer_sizes': (20, 20), 'learning_rate_init': 0.1}
            elif DR_name == 'SRP':
                best_params = {'alpha': 0.0001, 'hidden_layer_sizes': (10, 10), 'learning_rate_init': 0.05}
            elif DR_name == 'IMP':
                best_params = {'alpha': 0.0001, 'hidden_layer_sizes': (15, 15), 'learning_rate_init': 0.1}
            elif DR_name == 'Origin Data':
                best_params = {'alpha': 0.0001, 'hidden_layer_sizes': (15, 15), 'learning_rate_init': 0.1}
    best_hyperpara = [best_params['hidden_layer_sizes'], best_params['alpha'], best_params['learning_rate_init']]
    '''
    # Best fit model
    '''
    # Retrain the model with the best superparameter
    clf = NN(hidden_layer_sizes=best_hyperpara[0], alpha=best_hyperpara[1], learning_rate_init=best_hyperpara[2], activation='logistic', random_state=random_state, max_iter=2000)
    clf.fit(X_train, Y_train)

    # Metrics
    Y_pred_train = clf.predict(X_train)
    Y_pred_test = clf.predict(X_test)
    accuray_train = accuracy_score(Y_train, Y_pred_train)
    accuray_test = accuracy_score(Y_test, Y_pred_test)
    f1_train = f1_score(Y_train, Y_pred_train)
    f1_test = f1_score(Y_test, Y_pred_test)

    # Wall clock time
    start_time = time.time()
    clf.fit(X_train, Y_train)
    time_train = time.time() - start_time
    start_time = time.time()
    Y_pred_test = clf.predict(X_test)
    time_test = time.time() - start_time
    # Output metrics + wall clock time
    output(NNDR_metric_filename, filename, DR_name, time_train, time_test, accuray_train, accuray_test, f1_train, f1_test)

    # Learning curve
    lc(clf, X_train, Y_train, train_sizes, cv, scoring, filename, DR_name)
    # Loss curve
    clf.fit(X_train, Y_train)
    loss_curve_train = clf.loss_curve_
    plt.figure(figsize=(8.5, 7))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(loss_curve_train, '-', color='b')
    plt.title(filename + '-' + DR_name + ' Loss Curve with ' + filename)
    plt.xlabel('Epochs', fontsize=30)
    plt.ylabel('Training Loss', fontsize=30)
    plt.grid(True)
    plt.tight_layout
    plt.savefig('plots/NN_DR/' + filename + '-' + DR_name + '_LossCurve.png')
    plt.close()

def output(NNDR_metric_filename, filename, DR_name, time_train, time_test, accuracy_train, accuracy_test, f1_train, f1_test):
    with open(NNDR_metric_filename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{DR_name}\n')
        fp.write(f'{filename}\n')
        fp.write(f'***********************************' + '\n')
        fp.write(f'Training time (s): {time_train:.5f}\n')
        fp.write(f'Testing time (s):  {time_test:.5f}\n')
        fp.write(f'Accuracy_train  :  {accuracy_train:.3f}\n')
        fp.write(f'F1-score_train  :  {f1_train:.3f}\n')
        fp.write(f'Accuracy_test   :  {accuracy_test:.3f}\n')
        fp.write(f'F1-score_test   :  {f1_test:.3f}\n')
'''
#################################################
            Dimensionality Reduction
               PCA, ICA, SRP, IMP
#################################################
'''
def NNEval(X_train, Y_train, X_test, Y_test, filename, n_components, gridsearch_NN, scoring, NNDR_metric_filename, random_state):
    cv = 5
    train_sizes = np.arange(0.1, 1, 0.05)
    DR_names = ['PCA', 'ICA', 'SRP', 'IMP', 'Origin Data']
    for i, DR_name in enumerate(DR_names):
        if DR_name == 'Origin Data':
            NeuralNetworks(X_train, X_test, Y_train, Y_test, filename, DR_name, gridsearch_NN, scoring, train_sizes, cv, NNDR_metric_filename, random_state)
        else:
            if DR_name == 'PCA':
                clf = PCA(n_components=n_components[i], random_state=random_state)
            elif DR_name == 'ICA':
                clf = FastICA(n_components=n_components[i], random_state=random_state)
            elif DR_name == 'SRP':
                clf = SRP(n_components=n_components[i], random_state=random_state)
            elif DR_name == 'IMP':
                clf = Isomap(n_components=n_components[i])

            X_trans_train = clf.fit_transform(X_train)
            X_trans_test = clf.transform(X_test)
            NeuralNetworks(X_trans_train, X_trans_test, Y_train, Y_test, filename, DR_name, gridsearch_NN, scoring, train_sizes, cv, NNDR_metric_filename, random_state)