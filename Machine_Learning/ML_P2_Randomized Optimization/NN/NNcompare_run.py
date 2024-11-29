""""""
"""OMSCS2023FALL-P2: Randomized Optimization	  	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 12, 'axes.labelsize': 14})

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, confusion_matrix

import NNCompare
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
    X_train0, X_test0, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    # Normalize X, X_train, X_test, output will be numpy array, Y, Y_train, Y_test, is still pandas DataFrame
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
# 2.1 Learning curve
'''
# test_size and count_row are for the entire data, train_sizes is for the subsets
def lc(clf, test_size, count_row, X, Y, train_sizes, cv):
    train_sizes_N = []
    train_scores = []
    test_scores = []
    train_std =[]
    test_std = []
    for train_size in train_sizes:
        train_size_N = count_row * (1 - test_size) * train_size
        train_sizes_N.append(train_size_N)
        X_train0, X_test0, Y_train0, Y_test0 = train_test_split(X, Y, test_size=1-train_size, random_state=42)

        # Cross-validation
        kf = KFold(n_splits = cv, shuffle = True, random_state = 42)
        # Lists to store cross-validation results for this iteration
        train_score_cv = []
        test_score_cv = []

        for train_index, test_index in kf.split(X_train0):
            X_train, X_test = X_train0[train_index], X_train0[test_index]
            Y_train, Y_test = Y_train0.iloc[train_index], Y_train0.iloc[test_index]

            clf.fit(X_train, Y_train)
            Y_train_pred = clf.predict(X_train)
            Y_test_pred = clf.predict(X_test)

            train_score = accuracy_score(Y_train, Y_train_pred)
            test_score = accuracy_score(Y_test, Y_test_pred)

            train_score_cv.append(train_score)
            test_score_cv.append(test_score)

        # Calculate the mean score
        train_scores.append(np.mean(train_score_cv))
        test_scores.append(np.mean(test_score_cv))
        train_std.append(np.std(train_score_cv))
        test_std.append(np.std(test_score_cv))

    return train_sizes_N, train_scores, test_scores, train_std, test_std
'''
# 2.2 Confusion matrix
'''
def cmatrix(Y_test, Y_pred_test):
    cm = confusion_matrix(Y_test, Y_pred_test)
    return cm
'''
# 2.3 Overall metrics
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
# 3.1 Learning curve for 2.1
'''
def plot_lc(train_sizes_N, train_scores, test_scores, train_std, test_std, scoring):
    # Convert lists to NumPy arrays
    train_scores = np.array(train_scores)
    train_std = np.array(train_std)
    test_scores = np.array(test_scores)
    test_std = np.array(test_std)

    #plt.figure(figsize=(10, 5))
    plt.plot(train_sizes_N, train_scores, 'o-', color='b', label='Training')
    plt.plot(train_sizes_N, test_scores, 'o-', color='r', label='Validation')
    plt.fill_between(train_sizes_N, train_scores - train_std, train_scores + train_std, alpha=0.2, color="b")
    plt.fill_between(train_sizes_N, test_scores - test_std, test_scores + test_std, alpha=0.2, color="r")
    plt.title(model + ': Learning Curve with ' + filename)
    plt.xlabel('Training Set Size')
    if scoring == 'f1':
        plt.ylabel(scoring.capitalize() + ' Score')
    else:
        plt.ylabel(scoring.capitalize())
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('plots/' + model + '_' + filename + '_' + 'LearningCurve (' + scoring + ').png')
    plt.close()
'''
# 3.2 Confusion matrix for 2.2
'''
def plot_cm(cm):
    #plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
    plt.title(model + ': Confusion Matrix with ' + filename)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.tight_layout()
    plt.savefig('plots/' + model + '_' + filename + '_' + 'ConfusionMatrix.png')
    plt.close()
'''
# 3.4 Overall metrics for 3.4
'''
def output(txtfilename, model, filename, count_row, count_column, time_train, time_test, accuracy, precision, f1, recall, auc):
    with open(txtfilename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{model}\n')
        fp.write(f'{filename}\n')
        fp.write(f'***********************************' + '\n')

        fp.write('The data has ' + str(count_row) + ' rows and ' + str(count_column) + ' features' + '\n')
        fp.write(f'Training time (s): {time_train:.5f}\n')
        fp.write(f'Testing time (s):  {time_test:.5f}\n')
        fp.write(f'Accuracy:  {accuracy:.3f}\n')
        fp.write(f'Precision: {precision:.2f}\n')
        fp.write(f'F1-score:  {f1:.2f}\n')
        fp.write(f'Auc_score: {auc:.2f}\n')
        fp.write(f'Recall:    {recall:.2f}\n')
'''
# 3.3 Loss curve for NN
'''
def plot_loss(loss_curve):
    #plt.figure(figsize=(10, 5))
    plt.plot(loss_curve, '-', color='b')
    plt.title(model + ': Loss Curve with ' + filename)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/' + model + '_' + filename + '_' + 'LossCurve.png')
    plt.close()
'''
# 3.4 Compare for models
'''
def plot_compare(var_list, models, filename, str):
    yticks = np.arange(len(models))
    if str == 'Accuracy' or str == 'F1 Score':
        plt.figure(figsize=(10, 5))
    plt.barh(yticks, var_list)
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels(models)
    plt.title('NN Comparison of ' + str + ' with ' + filename)
    plt.xlabel(str)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plots/Comparison of ' + str + ' with ' + filename + '.png')
    plt.close()
'''
#################################################
            Read Data and Run Experiments
#################################################
'''
if __name__ == "__main__":
    # General setting
    cv=5
    test_size = 0.2
    undersample = True    #True: balance data; False: turn off
    filename = 'Data_Diabetes'
    train_sizes = np.arange(0.1, 1, 0.05)
    scoring = 'accuracy'
    # Output metrics file name
    txtfilename = 'model_metrics.txt'
    with open(txtfilename, 'a') as fp:
        pass
    # Data preprocessing
    X, Y, X_train, X_test, Y_train, Y_test, count_row, count_column = preprocess(filename, test_size, undersample)
    # Final model compare variables
    mt_train = []
    mt_test = []
    accuracy_value = []
    f1_value = []
    models_compare = ['BP', 'RHC', 'SA', 'GA']

    # Run optimization
    clf, Y_pred_train, Y_pred_test, time_train, time_test, loss_curve_train = NNCompare.NN(X_train, X_test, Y_train, filename)
    models = ['NN_BP', 'NN_RHC', 'NN_SA', 'NN_GA']
    for ir in np.arange(4):
        model = models[ir]
        # 1 Loss curve
        matplotlib.rcParams.update({'font.size': 12, 'axes.labelsize': 14})
        #plot_loss(loss_curve_train[ir])
        accuracy, precision, f1, recall, auc = metrics_output(Y_test, Y_pred_test[ir])
        # 2 Output metrics
        output(txtfilename, model, filename, count_row, count_column, time_train[ir], time_test[ir], accuracy, precision, f1, recall, auc)
        # 3 Learning curve
        #train_sizes_N, train_scores, test_scores, train_std, test_std = lc(clf[ir], test_size, count_row, X, Y, train_sizes, cv)
        #plot_lc(train_sizes_N, train_scores, test_scores, train_std, test_std, scoring)
        # 4 Confusion matrix
        matplotlib.rcParams.update({'font.size': 11, 'axes.labelsize': 12})
        cm = cmatrix(Y_test, Y_pred_test[ir])
        #plot_cm(cm)
        # 5.1 Comparison data
        mt_train.append(time_train[ir])
        mt_test.append(time_test[ir])
        accuracy_value.append(accuracy)
        f1_value.append(f1)

    # 5.2 Model time & accuracy comparison
    matplotlib.rcParams.update({'font.size': 12, 'axes.labelsize': 14})
    plot_compare(mt_train, models_compare, filename, 'Training Time (s)')
    plot_compare(mt_test, models_compare, filename, 'Testing Time (s)')
    plot_compare(accuracy_value, models_compare, filename, 'Accuracy')
    plot_compare(f1_value, models_compare, filename, 'F1 Score')