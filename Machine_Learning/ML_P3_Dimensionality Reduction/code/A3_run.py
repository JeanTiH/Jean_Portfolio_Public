""""""
"""OMSCS2023FALL-P3: Dimensionality Reduction	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import Cluster as CT
import DR
import ClusterDR as CDR
import NNDR
import NNCluster

'''
#################################################
            Step1 Data preprocessing
#################################################
'''
def preprocess(filename, test_size, undersample):
    data_raw = pd.read_csv(filename+'.csv')
    # -------------------------------------------
    # 1.1 Undersample for balanced data: X, Y, data
    # -------------------------------------------
    X = data_raw.iloc[:, :-1]
    Y = data_raw.iloc[:, -1]

    if undersample:
        undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
        X, Y = undersampler.fit_resample(X, Y)

    data = pd.concat([X, Y], axis=1)

    # Data dimension
    count_row = data.shape[0]
    count_column = data.shape[1] - 1
    # -------------------------------------------
    # 1.2 Separate features (X) and targets (Y)
    #           Split train and test
    #               Normalize X
    # -------------------------------------------
    # Split training/testing set
    X_train0, X_test0, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Normalize X, X_train, X_test, output will be numpy array, Y, Y_train, Y_test, is still pandas DataFrame
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train = scaler.fit_transform(X_train0)
    X_test = scaler.transform(X_test0)

    return X, Y, X_train, X_test, Y_train, Y_test, count_row, count_column

def neg_log_likelihood_score(estimator, X):
    return -np.sum(estimator.score_samples(X))

def GridSearchGMM(X):
    param_grid = {
        'n_components': [2, 3, 4, 5],
        'covariance_type': ['full', 'tied', 'diag', 'spherical'],
        'n_init': [1, 5, 10],
        'max_iter': [100, 200, 300],
        'init_params': ['kmeans', 'k-means++', 'random', 'random_from_data']
    }
    clf = GaussianMixture()
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=make_scorer(neg_log_likelihood_score))
    grid_search.fit(X)

    best_params = grid_search.best_params_
    print(best_params)

def GridSearchKmeans(X):
    param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'init': ['k-means++', 'random'],
        'n_init': [1, 5, 10],
        'max_iter': [100, 200, 300]
    }
    clf = KMeans()
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=make_scorer(neg_log_likelihood_score))
    grid_search.fit(X)

    best_params = grid_search.best_params_
    print(best_params)
'''
#################################################
            Read Data and Run Experiments
#################################################
'''
if __name__ == "__main__":
    undersample = False  # True: balance data;     False: turn off
    test_size = 0.2
    K = np.arange(1, 15, 1)
    GridSearch = False
    Cluster = True                  # True: do Clustering; False: turn off
    DimReduction = True             # True: do Dim Reduction; False: turn off
    Cluster_DimReduction = True     # True: do Dim Reduction then Clustering; False: turn off
    NN_DR = True                    # True: do NN with Dim Reduction; False: turn off
    NN_Cluster = True               # True: do NN with Clustering; False: turn off

    if Cluster:
        CT_metric_filename = 'Cluster_metric.txt'
        if os.path.exists(CT_metric_filename):
            os.remove(CT_metric_filename)

    if DimReduction:
        DR_metric_filename = 'DimRedcution_metric.txt'
        if os.path.exists(DR_metric_filename):
            os.remove(DR_metric_filename)

    if Cluster_DimReduction:
        CDR_metric_filename = 'Cluster_DimRedcution_metric.txt'
        if os.path.exists(CDR_metric_filename):
            os.remove(CDR_metric_filename)

    if NN_DR:
        NNDR_metric_filename = 'NN_DR_metric.txt'
        if os.path.exists(NNDR_metric_filename):
            os.remove(NNDR_metric_filename)

    if NN_Cluster:
        NNCluster_metric_filename = 'NN_Cluster_metric.txt'
        if os.path.exists(NNCluster_metric_filename):
            os.remove(NNCluster_metric_filename)

    for filename in ['Data_Diabetes', 'Data_Rice']:
        if filename == 'Data_Diabetes':
            random_state = 42
        elif filename == 'Data_Rice':
            random_state = 0
        # Data preprocessing
        X, Y, X_train, X_test, Y_train, Y_test, count_row, count_column = preprocess(filename, test_size, undersample)
        feature_num = X_train.shape[1]
        '''
        data_rank = np.linalg.matrix_rank(X)
        print("Rank of the dataset:", data_rank)
        '''
        if GridSearch:
            GridSearchGMM(X_train)
            GridSearchKmeans(X_train)

        # Step1: Clustering: EM & Kmeans
        if Cluster:
            if filename == 'Data_Diabetes':
                K_optimal = [7, 2]   # Optimal from step1 unsupervised EM, Kmeans
            elif filename == 'Data_Rice':
                K_optimal = [12, 2]  # Optimal from step1 unsupervised EM, Kmeans
            CT.Cluster_Eval(K, K_optimal, X, Y, X_train, Y_train, X_test, Y_test, filename, CT_metric_filename, random_state)

        # Step2: Dimensionality Reduction
        if DimReduction:

            DR.NNEval_clf(X_train, Y_train, X_test, Y_test, filename, feature_num, random_state)
            DR.PCA_Eval(X_train, Y_train, X_test, Y_test, filename, feature_num, random_state)
            DR.ICA_Eval(X_train, Y_train, X_test, Y_test, filename, feature_num, random_state)
            DR.SRP_Eval(X_train, Y_train, X_test, Y_test, filename, feature_num, random_state)
            DR.IMP_Eval(X_train, Y_train, X_test, Y_test, filename, feature_num)

            if filename == 'Data_Diabetes':
                n_components = [6, 6, 4, 4]  # Optimal from step2 unsupervised PCA, ICA, SRP, IMP
            elif filename == 'Data_Rice':
                n_components = [3, 6, 4, 3]  # Optimal from step2 unsupervised PCA, ICA, SRP, IMP
            DR.pairplots_corrmatrix(X_train, Y_train, X_test, Y_test, filename, n_components, random_state)

        # Step3: Dimensionality Reduction + Clustering Step3 Unsuperived-Determined component numbers
        if Cluster_DimReduction:
            if filename == 'Data_Diabetes':
                n_components = [6, 6, 4, 4]  # Optimal from step2 unsupervised PCA, ICA, SRP, IMP
            elif filename == 'Data_Rice':
                n_components = [3, 6, 4, 3]  # Optimal from step2 unsupervised PCA, ICA, SRP, IMP
            CDR.DR_cluster(X_train, Y_train, X_test, Y_test, filename, n_components, K, CDR_metric_filename, random_state)
        '''
        Step 4-5
        '''
        # Step4: NN + Dimensionality Reduction
        if NN_DR:
            if filename == 'Data_Diabetes':
                n_components = [6, 6, 4, 4]  # Optimal from step2 unsupervised PCA, ICA, SRP, IMP
                pass
            elif filename == 'Data_Rice':
                n_components = [3, 6, 4, 3]  # Optimal from step2 unsupervised PCA, ICA, SRP, IMP
                gridsearch_NN = False       # Trun on to gridsearch the best hyperparameters
                scoring = 'f1'
                NNDR.NNEval(X_train, Y_train, X_test, Y_test, filename, n_components, gridsearch_NN, scoring, NNDR_metric_filename, random_state)
        # Step5: NN + Clustering
        if NN_Cluster:
            if filename == 'Data_Diabetes':
                k = [7, 2]  # Optimal from step1 unsupervised EM & Kmeans
                pass
            elif filename == 'Data_Rice':
                k = [12, 2]  # Optimal from step1 unsupervised EM & Kmeans
                gridsearch_NN = False  # Trun on to gridsearch the best hyperparameters
                scoring = 'f1'
                NNCluster.NNEval(X_train, Y_train, X_test, Y_test, filename, k, gridsearch_NN, scoring, NNCluster_metric_filename, random_state)



