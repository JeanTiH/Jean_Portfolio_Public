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
matplotlib.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'legend.fontsize': 16})
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn.neural_network import MLPClassifier as NN
from sklearn.model_selection import GridSearchCV, learning_curve

from sklearn.metrics import accuracy_score, f1_score

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
    plt.savefig('plots/NN_Cluster/' + filename + '-' + DR_name + '_LearningCurve (' + scoring + ').png')
    plt.close()


def NeuralNetworks(X_train, X_test, Y_train, Y_test, filename, clf_name, gridsearch_NN, scoring, train_sizes, cv, NNCluster_metric_filename, random_state):
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
            if clf_name == 'EM':
                best_params = {'alpha': 0.01, 'hidden_layer_sizes': (10,20), 'learning_rate_init': 0.001}
            elif clf_name == 'KMeans':
                best_params = {'alpha': 0.01, 'hidden_layer_sizes': (10,10), 'learning_rate_init': 0.001}
            elif clf_name == 'Origin Data':
                best_params = {'alpha': 0.1, 'hidden_layer_sizes': (10, 20), 'learning_rate_init': 0.001}
        elif filename == 'Data_Diabetes':
            if clf_name == 'EM':
                best_params = {'alpha': 0.0001, 'hidden_layer_sizes': (15, 15), 'learning_rate_init': 0.1}
            elif clf_name == 'KMeans':
                best_params = {'alpha': 0.0001, 'hidden_layer_sizes': (15, 15), 'learning_rate_init': 0.1}
            elif clf_name == 'Origin Data':
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
    output(NNCluster_metric_filename, filename, clf_name, time_train, time_test, accuray_train, accuray_test, f1_train, f1_test)

    # Learning curve
    lc(clf, X_train, Y_train, train_sizes, cv, scoring, filename, clf_name)
    # Loss curve
    clf.fit(X_train, Y_train)
    loss_curve_train = clf.loss_curve_
    plt.figure(figsize=(8.5, 7))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(loss_curve_train, '-', color='b')
    plt.title(filename + '-' + clf_name + ' Loss Curve with ' + filename)
    plt.xlabel('Epochs', fontsize=30)
    plt.ylabel('Training Loss', fontsize=30)
    plt.grid(True)
    plt.tight_layout
    plt.savefig('plots/NN_Cluster/' + filename + '-' + clf_name + '_LossCurve.png')
    plt.close()

def output(NNCluster_metric_filename, filename, clf_name, time_train, time_test, accuracy_train, accuracy_test, f1_train, f1_test):
    with open(NNCluster_metric_filename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{clf_name}\n')
        fp.write(f'{filename}\n')
        fp.write(f'***********************************' + '\n')
        fp.write(f'Training time (s): {time_train:.5f}\n')
        fp.write(f'Testing time (s):  {time_test:.5f}\n')
        fp.write(f'Accuracy_train  :  {accuracy_train:.4f}\n')
        fp.write(f'F1-score_train  :  {f1_train:.4f}\n')
        fp.write(f'Accuracy_test   :  {accuracy_test:.4f}\n')
        fp.write(f'F1-score_test   :  {f1_test:.4f}\n')
'''
#################################################
            Dimensionality Reduction
               PCA, ICA, SRP, IMP
#################################################
'''
def NNEval(X_train, Y_train, X_test, Y_test, filename, k, gridsearch_NN, scoring, NNCluster_metric_filename, random_state):
    cv = 5
    train_sizes = np.arange(0.1, 1, 0.05)
    clf_names = ['EM', 'KMeans', 'Origin Data']
    for i, clf_name in enumerate(clf_names):
        if clf_name == 'Origin Data':
            NeuralNetworks(X_train, X_test, Y_train, Y_test, filename, clf_name, gridsearch_NN, scoring, train_sizes, cv, NNCluster_metric_filename, random_state)
        else:
            if clf_name == 'EM':
                clf = GaussianMixture(n_components=k[0], random_state=random_state)
            elif clf_name == 'KMeans':
                clf = KMeans(n_clusters=k[1], n_init=1, random_state=random_state)

            train_labels = clf.fit_predict(X_train)
            test_labels = clf.predict(X_test)
            # Add new feature (labels) to X
            train_labels = train_labels.reshape((-1, 1))
            X_train_labeled = np.hstack((X_train, train_labels))
            test_labels = test_labels.reshape((-1, 1))
            X_test_labeled = np.hstack((X_test, test_labels))

            NeuralNetworks(X_train_labeled, X_test_labeled, Y_train, Y_test, filename, clf_name, gridsearch_NN, scoring, train_sizes, cv, NNCluster_metric_filename, random_state)