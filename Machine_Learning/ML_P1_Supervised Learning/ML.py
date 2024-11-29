""""""
"""OMSCS2023FALL-P1: Supervised Learning 		  	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

'''
Pre-processing and post-processing code for 5 ML algorithms
'''
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as NN
from sklearn.svm import SVC

import ML_Algorithms as mla
'''
#################################################
            Step1 Data preprocessing
#################################################
'''
def preprocess(filename, undersample):
    data_raw = pd.read_csv(filename+'.csv')
    # -------------------------------------------
    # 1.1 Undersample for balanced data: X, Y, data
    # -------------------------------------------
    X = data_raw.iloc[:, :-1]
    Y = data_raw.iloc[:, -1]

    if undersample:
        undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
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
    X_train0, X_test0, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Normalize X, X_train, X_test
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train = scaler.fit_transform(X_train0)
    X_test = scaler.transform(X_test0)

    return X, Y, X_train, X_test, Y_train, Y_test, count_row, count_column
'''
#################################################
            Step2 General Metrics
-------------------------------------------------
  Validation/Learning Curves + Confusion Matrix
                Table Output
#################################################
'''
'''
# 2.1 Validation curve
'''
def vc(estimator, X_train, Y_train, para_name, para_range, cv, scoring):
    train_scores, test_scores = validation_curve(estimator, X_train, Y_train, param_name=para_name, param_range=para_range, cv=cv, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    return train_scores_mean, test_scores_mean, train_scores_std, test_scores_std
'''
# 2.2 Learning curve
'''
def lc(estimator, X_train, Y_train, train_sizes, cv, scoring):
    train_sizes_N, train_scores, test_scores = learning_curve(estimator, X_train, Y_train, train_sizes=train_sizes, cv=cv, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    return train_sizes_N, train_scores_mean, test_scores_mean, train_scores_std, test_scores_std
'''
# 2.3 Confusion matrix
'''
def cmatrix(Y_test, Y_pred_test):
    cm = confusion_matrix(Y_test, Y_pred_test)
    return cm
'''
# 2.4 Overall metrics
'''
def metrics_output(Y_test, Y_pred_test):
    accuracy = accuracy_score(Y_test, Y_pred_test)
    precision = precision_score(Y_test, Y_pred_test)
    f1 = f1_score(Y_test, Y_pred_test)
    recall = recall_score(Y_test, Y_pred_test)
    auc = roc_auc_score(Y_test, Y_pred_test)
    return accuracy, precision, f1, recall, auc
'''
#################################################
                Step3 Plot and Table
#################################################
'''
'''
# 3.1 Validation curve (accuracy/loss) for 2.1
'''
def plot_vc(para_name, para_range, train_scores_mean, test_scores_mean, train_scores_std, test_scores_std, scoring):
    plt.figure(figsize=(10, 5))
    if para_name == 'p' or para_name == 'weights' or para_name == 'kernel':
        # Define the width of the bars
        bar_width = 0.35
        # Create x-coordinates for the bars
        x_train = np.arange(len(para_range))
        x_test = x_train + bar_width
        # Create bar plots
        plt.bar(x_train, train_scores_mean, label='Training', alpha=0.7, color='b', width=bar_width)
        plt.bar(x_test, test_scores_mean, label='Validation', alpha=0.7, color='r', width=bar_width)
        plt.xticks(x_train + bar_width / 2, para_range)

        if para_name == 'p':
            plt.ylim(0, 1.2)

    else:
        if para_name == 'alpha' or para_name == 'C':
            plt.xscale('log')
        plt.plot(para_range, train_scores_mean, 'o-', color='b', label='Training')
        plt.plot(para_range, test_scores_mean, 'o-', color='r', label='Validation')
        plt.fill_between(para_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='b')
        plt.fill_between(para_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color='r')

    plt.title(model + ': Validation Curve on ' + para_name.capitalize() + ' with ' + filename)
    plt.grid(True)

    if para_name == 'hidden_layer_sizes':
        plt.xlabel('Number of Neurons (One Hidden Layer)')
    else:
        plt.xlabel(para_name.capitalize())

    if scoring == 'f1':
        plt.ylabel(scoring.capitalize() + ' Score')
    else:
        plt.ylabel(scoring.capitalize())

    if filename == 'Data_Diabetes' and model == 'Boosting' or para_name == 'hidden_layer_sizes':
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='best')

    plt.savefig(model + '_' + filename + '_' + 'ValidationCurve on ' + para_name + ' (' + scoring + ').png')
    plt.close()
'''
# 3.2 Learning curve for 2.2
'''
def plot_lc(train_sizes_N, train_scores_mean, test_scores_mean, train_scores_std, test_scores_std, scoring):
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes_N, train_scores_mean, 'o-', color='b', label='Training')
    plt.plot(train_sizes_N, test_scores_mean, 'o-', color='r', label='Validation')
    plt.fill_between(train_sizes_N, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="b")
    plt.fill_between(train_sizes_N, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="r")
    plt.title(model + ': Learning Curve with ' + filename)
    plt.xlabel('Training Set Size')
    plt.grid(True)

    if scoring == 'f1':
        plt.ylabel(scoring.capitalize() + ' Score')
    else:
        plt.ylabel(scoring.capitalize())
    if (model == 'DecisionTree' and filename == 'Data_Diabetes') or (model == 'Boosting') or (model == 'KNeighbors'):
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='best')
    plt.savefig(model + '_' + filename + '_' + 'LearningCurve (' + scoring + ').png')
    plt.close()
'''
# 3.3 Confusion matrix for 2.3
'''
def plot_cm(cm):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
    plt.title(model + ': Confusion Matrix with ' + filename)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.savefig(model + '_' + filename + '_' + 'ConfusionMatrix.png')
    plt.close()
'''
# 3.4 Overall metrics for 3.4
'''
def output(txtfilename, model, filename, count_row, count_column, best_hyperpara, time_train, time_test, accuracy, precision, f1, recall, auc):
    with open(txtfilename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{model}\n')
        fp.write(f'{filename}\n')
        fp.write(f'***********************************' + '\n')

        fp.write('The data has ' + str(count_row) + ' rows and ' + str(count_column) + ' features' + '\n')
        if model == 'DecisionTree':
            best_max_depth = best_hyperpara[0]
            best_min_samples_leaf = best_hyperpara[1]
            fp.write('Best max_depth is: ' + str(best_max_depth) + ', Best min_samples_leaf is: ' + str(best_min_samples_leaf) + '\n')
        elif model == 'Boosting':
            best_max_depth = best_hyperpara[0]
            best_min_samples_leaf = best_hyperpara[1]
            best_n_estimators = best_hyperpara[2]
            best_learning_rate = best_hyperpara[3]
            fp.write('Best max_depth is: ' + str(best_max_depth) + ', Best min_samples_leaf is: ' + str(best_min_samples_leaf) + ', Best n_estimators is: ' + str(best_n_estimators) + ', Best learning_rate is: ' + str(best_learning_rate) + '\n')
        elif model == 'KNeighbors':
            best_n_neighbors = best_hyperpara[0]
            best_weights = best_hyperpara[1]
            best_p = best_hyperpara[2]
            fp.write('Best n_neighbors is: ' + str(best_n_neighbors) + ', Best weights is: ' + str(best_weights) + ', Best p is: ' + str(best_p) + '\n')
        elif model == 'NeuralNetworks':
            best_hidden_layer_sizes = best_hyperpara[0]
            best_alpha = best_hyperpara[1]
            best_learning_rate_init = best_hyperpara[2]
            fp.write('Best hidden_layer_sizes is ' + str(best_hidden_layer_sizes) + ', Best alpha is ' + str(best_alpha) + ', Best learning_rate_init is ' + str(best_learning_rate_init) + '\n')
        elif model == 'SVM':
            best_kernel = best_hyperpara[0]
            best_C = best_hyperpara[1]
            best_gamma = best_hyperpara[2]
            if best_kernel == 'poly':
                best_degree = best_hyperpara[3]
                fp.write('Best kernel is: ' + str(best_kernel) + ', Best degree is: ' + str(best_degree) + ', Best C is: ' + str(best_C) + ', Best gamma is: ' + str(best_gamma) + '\n')
            else:
                fp.write('Best kernel is: ' + str(best_kernel) + ', Best C is: ' + str(best_C) + ', Best gamma is: ' + str(best_gamma) + '\n')

        fp.write(f'Training time (s): {time_train:.5f}\n')
        fp.write(f'Testing time (s):  {time_test:.5f}\n')
        fp.write(f'Accuracy:  {accuracy:.2f}\n')
        fp.write(f'Precision: {precision:.2f}\n')
        fp.write(f'F1-score:  {f1:.2f}\n')
        fp.write(f'Auc_score: {auc:.2f}\n')
        fp.write(f'Recall:    {recall:.2f}\n')
'''
# 3.5 Loss curve for NN
'''
def plot_loss(loss_curve):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve, '-', color='b')
    plt.title(model + ': Loss Curve with ' + filename)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.savefig(model + '_' + filename + '_' + 'LossCurve.png')
    plt.close()
'''
# 3.6 Compare for models
'''
def plot_compare(var_list, models, filename, str):
    yticks = np.arange(len(models))
    plt.figure(figsize=(10, 5))
    plt.barh(yticks, var_list)
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels(models)
    plt.title('Comparison of ' + str + ' with ' + filename)
    plt.xlabel(str)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('Comparison of ' + str + ' with ' + filename + '.png')
    plt.close()
'''
#################################################
            Read Data and Run Experiments
#################################################
'''
if __name__ == "__main__":
    cv=5
    undersample = True    #True: balance data; False: turn off
    gridsearch = False     #True: trun on the grid_search function; False: turn off
    train_sizes = np.arange(0.1, 1, 0.05)
    scoring = 'accuracy'
    txtfilename = 'model_metrics-test-nn.txt'
    # Final model compare variables
    mt_train1 = []
    mt_train2 = []
    mt_test1 = []
    mt_test2 = []
    accuracy1 = []
    accuracy2 = []

    with open(txtfilename, 'a') as fp:
        pass
#    models = ['DecisionTree', 'Boosting', 'KNeighbors', 'NeuralNetworks', 'SVM']
    models = ['NeuralNetworks']
    models_compare = ['DT', 'GB', 'KNN', 'NN', 'SVM']
    for model in models:

#        for filename in ['Data_Diabetes', 'Data_Sleepiness']:
        for filename in ['Data_Diabetes']:
            # Data preprocessing
            X, Y, X_train, X_test, Y_train, Y_test, count_row, count_column = preprocess(filename, undersample)
#            print(count_row, count_column)
            para_name = []
            para_range = []
            estimator = []

            if model == 'DecisionTree':
                best_estimator, best_hyperpara, Y_pred_test, time_train, time_test = mla.DecisionTree(X_train, X_test, Y_train, filename, gridsearch, scoring)

                para_name.append('max_depth')
                para_range.append(np.arange(1, 30))
                estimator.append(DT(min_samples_leaf=best_hyperpara[1], random_state=42))

                para_name.append('min_samples_leaf')
                para_range.append(np.arange(1, 30))
                estimator.append(DT(max_depth=best_hyperpara[0], random_state=42))

            elif model == 'Boosting':
                best_estimator, best_hyperpara, Y_pred_test, time_train, time_test = mla.Boosting(X_train, X_test, Y_train, filename, gridsearch, scoring)

                para_name.append('n_estimators')
                para_range.append(np.arange(1, 350, 10))
                estimator.append(GB(max_depth=best_hyperpara[0], min_samples_leaf=best_hyperpara[1], learning_rate=best_hyperpara[3], random_state=42))

                para_name.append('max_depth')
                para_range.append(np.arange(1, 30))
                estimator.append(GB(min_samples_leaf=best_hyperpara[1], n_estimators=best_hyperpara[2], learning_rate=best_hyperpara[3], random_state=42))

            elif model == 'KNeighbors':
                best_estimator, best_hyperpara, Y_pred_test, time_train, time_test = mla.KNeighbors(X_train, X_test, Y_train, filename, gridsearch, scoring)

                para_name.append('n_neighbors')
                para_range.append(np.arange(1, 250, 10))
                estimator.append(KNN(weights=best_hyperpara[1], p=best_hyperpara[2]))

#                para_name.append('weights')
#                para_range.append(['uniform', 'distance'])

                para_name.append('p')
                para_range.append([1,2,3,4,5])
                estimator.append(KNN(n_neighbors=best_hyperpara[0], weights=best_hyperpara[1]))

            elif model == 'NeuralNetworks':
                best_estimator, best_hyperpara, Y_pred_test, time_train, time_test, loss_curve_train = mla.NeuralNetworks(X_train, X_test, Y_train, filename, gridsearch, scoring)

                para_name.append('hidden_layer_sizes')
                para_range.append(np.arange(1, 101, 5))
                estimator.append(NN(hidden_layer_sizes=(para_range,), alpha=best_hyperpara[1], learning_rate_init=best_hyperpara[2], activation='logistic', random_state=42, max_iter=2000))

                para_name.append('alpha')
                para_range.append(np.logspace(-3, 3, num=7))
                estimator.append(NN(hidden_layer_sizes=best_hyperpara[0], learning_rate_init=best_hyperpara[2], activation='logistic', random_state=42, max_iter=2000))

                plot_loss(loss_curve_train)

            elif model == 'SVM':
                best_estimator, best_hyperpara, Y_pred_test, time_train, time_test = mla.SVM(X_train, X_test, Y_train, filename, gridsearch, scoring)

                para_name.append('kernel')
                para_range.append(['linear', 'poly', 'rbf', 'sigmoid'])
                estimator.append(SVC(C=best_hyperpara[1], gamma=best_hyperpara[2], degree=2, random_state=42))

                para_name.append('C')
                para_range.append(np.logspace(-2, 2, num=5))
                estimator.append(SVC(kernel=best_hyperpara[0], C=best_hyperpara[1], gamma=best_hyperpara[2], degree=2, random_state=42))

            # Output table
            accuracy, precision, f1, recall, auc = metrics_output(Y_test, Y_pred_test)
            output(txtfilename, model, filename, count_row, count_column, best_hyperpara, time_train, time_test, accuracy, precision, f1, recall, auc)
            print(accuracy)
            input('pause')
            # Fianl model compare variables
            if filename == 'Data_Diabetes':
                mt_train1.append(time_train)
                mt_test1.append(time_test)
                accuracy1.append(accuracy)
            elif filename == 'Data_Sleepiness':
                mt_train2.append(time_train)
                mt_test2.append(time_test)
                accuracy2.append(accuracy)

            # Validation curve
            for i in [0, 1]:
                train_scores_mean, test_scores_mean, train_scores_std, test_scores_std = vc(estimator[i], X_train, Y_train, para_name[i], para_range[i], cv, scoring)
                plot_vc(para_name[i], para_range[i], train_scores_mean, test_scores_mean, train_scores_std, test_scores_std, scoring)

            # Learning curve
            train_sizes_N, train_scores_mean, test_scores_mean, train_scores_std, test_scores_std = lc(best_estimator, X_train, Y_train, train_sizes, cv, scoring)
            plot_lc(train_sizes_N, train_scores_mean, test_scores_mean, train_scores_std, test_scores_std, scoring)

            # Confusion matrix
            cm = cmatrix(Y_test, Y_pred_test)
            plot_cm(cm)
'''
    # Model time & accuracy
    plot_compare(mt_train1, models_compare, 'Data_Diabetes', 'Training Time (s)')
    plot_compare(mt_train2, models_compare, 'Data_Sleepiness', 'Training Time (s)')
    plot_compare(mt_test1, models_compare, 'Data_Diabetes', 'Testing Time (s)')
    plot_compare(mt_test2, models_compare, 'Data_Sleepiness', 'Testing Time (s)')
    plot_compare(accuracy1, models_compare, 'Data_Diabetes', 'Accuracy')
    plot_compare(accuracy2, models_compare, 'Data_Sleepiness', 'Accuracy')
'''