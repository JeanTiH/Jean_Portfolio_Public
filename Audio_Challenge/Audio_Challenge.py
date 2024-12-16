# 2024 Renesas Reality AI project "Audio Data Challenge" - Part2: models
# Proprietary and Confidential Information of Renesas Electronics Americans
# Author: jjhan201707@gmail.com

import os
import time
import librosa
import numpy as np
import pandas as pd

from scipy.signal import butter, lfilter

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
Part1: Pre-processing
'''
# Read Meta File
def read_meta_file(data_path, meta_filename):
    data = pd.read_csv(os.path.join(data_path, meta_filename))
    return data

# Nosie Reduction + MFCC
def extract_mfcc(row, use_bandpass):
    audio_filename = row['file']
    class_name = row['label']
    wav_file_path = os.path.join('data', class_name, audio_filename)

    try:
        y, sr = librosa.load(wav_file_path, sr=None)

        if use_bandpass:
            lowcut = 300
            highcut = 8000
            y_filtered = bandpass_filter(y, lowcut=lowcut, highcut=highcut, fs=sr, order=5)
        else:
            y_filtered = y

        mfcc_features = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc_features, axis=1)
    except Exception as e:
        print(f"Error loading {wav_file_path}: {e}")
        mfcc_mean = np.zeros(13)

    return mfcc_mean

# Bandpass filter helper function
def bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

# Feature label, data split, k-folder CV
def preprocess_data(data):
    X = np.vstack(data['mfcc'].values)
    y = data['label'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return X, y, kf

'''
Part2: Model evaluation helper functions
'''
def plot_learning_curve(model, X, y, cv, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)

    plt.plot(train_sizes, val_mean, label='Validation Accuracy', color='red', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='red', alpha=0.2)

    plt.title(title, fontsize=14)
    plt.xlabel('Training Set Size', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{title}.png')

def plot_cofusion_matrix(cm, y, title, filename):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16, labelpad=15, rotation=90)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png')

'''
Part3: Models
'''
def SVM_model(X, y, kf):
    print('****************************')
    print('            SVM             ')
    print('****************************')
    param_grid = {'C': [0.5, 1, 1.5, 2, 5]}
    svm_model = SVC(kernel='linear')

    total_start_time = time.time()  # Track total time
    plot_learning_curve(svm_model, X, y, cv=kf, title='SVM_learning_curve')

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    confusion_matrices_sum = np.zeros((len(np.unique(y)), len(np.unique(y))))

    # Perform K-Fold Cross Validation with Grid Search
    fold = 1
    for train_index, test_index in kf.split(X):
        print(f'Training on fold {fold}...')

        # Split data into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Perform Grid Search
        grid_search = GridSearchCV(svm_model, param_grid, cv=5)
        grid_search.fit(X_train_scaled, y_train)
        print(f"Best C value for fold {fold}: {grid_search.best_params_['C']}")

        # Use the best model to make predictions on the test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Evaluate the model
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Print the evaluation metrics for each fold
        print(f'Fold {fold} Accuracy: {accuracy}')
        print(f'Fold {fold} Precision: {precision}')
        print(f'Fold {fold} Recall: {recall}')
        print(f'Fold {fold} F1 Score: {f1}')
        print('----------------------------')

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices_sum += cm
        # Plot confusion matrix for each fold
        plot_cofusion_matrix(confusion_matrices_sum, y, f'Confusion Matrix - Fold {fold}', f'SVM_Confusion_Matrix_{fold}')
        fold += 1

    # Plot confusion matrix across the folds
    plot_cofusion_matrix(confusion_matrices_sum, y, 'Cumulative Confusion Matrix', 'SVM_Confusion_Matrix')

    # Print overall cross-validation results
    print(f"\nMean Accuracy across all folds: {np.mean(accuracies)}")
    print(f"Standard Deviation of Accuracy: {np.std(accuracies)}")
    print(f"Mean Precision across all folds: {np.mean(precisions)}")
    print(f"Mean Recall across all folds: {np.mean(recalls)}")
    print(f"Mean F1 Score across all folds: {np.mean(f1_scores)}")

    total_end_time = time.time()
    print(f"\nTotal time for SVM model: {total_end_time - total_start_time:.5f} seconds")

def KNN_model(X, y, kf):
    print('****************************')
    print('            KNN             ')
    print('****************************')
    param_grid = {'n_neighbors': [1, 2, 3, 4, 5], 'weights': ['uniform', 'distance']}
    knn_model = KNeighborsClassifier()

    total_start_time = time.time()  # Track total time
    plot_learning_curve(knn_model, X, y, cv=kf, title='KNN_learning_curve')
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    confusion_matrices_sum = np.zeros((len(np.unique(y)), len(np.unique(y))))

    # Perform K-Fold Cross Validation with Grid Search
    fold = 1
    for train_index, test_index in kf.split(X):
        print(f'Training on fold {fold}...')

        # Split data into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Perform Grid Search
        grid_search = GridSearchCV(knn_model, param_grid, cv=5)
        grid_search.fit(X_train_scaled, y_train)
        print(f"Best parameters for fold {fold}: {grid_search.best_params_}")

        # Use the best model to make predictions on the test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Print the evaluation metrics for each fold
        print(f'Fold {fold} Accuracy: {accuracy}')
        print(f'Fold {fold} Precision: {precision}')
        print(f'Fold {fold} Recall: {recall}')
        print(f'Fold {fold} F1 Score: {f1}')
        print('----------------------------')

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices_sum += cm
        # Plot confusion matrix for each fold
        plot_cofusion_matrix(confusion_matrices_sum, y, f'Confusion Matrix - Fold {fold}', f'KNN_Confusion_Matrix_{fold}')
        fold += 1

    # Plot confusion matrix across the folds
    plot_cofusion_matrix(confusion_matrices_sum, y, 'Cumulative Confusion Matrix', 'KNN_Confusion_Matrix')

    # Print overall cross-validation results
    print(f"\nMean Accuracy across all folds: {np.mean(accuracies)}")
    print(f"Standard Deviation of Accuracy: {np.std(accuracies)}")
    print(f"Mean Precision across all folds: {np.mean(precisions)}")
    print(f"Mean Recall across all folds: {np.mean(recalls)}")
    print(f"Mean F1 Score across all folds: {np.mean(f1_scores)}")
    total_end_time = time.time()
    print(f"\nTotal time for KNN model: {total_end_time - total_start_time:.5f} seconds")

if __name__ == "__main__":
    data = read_meta_file('data', 'meta.csv')
    data['mfcc'] = data.apply(extract_mfcc, axis=1, use_bandpass = True)
    X, y, kf = preprocess_data(data)
    SVM_model(X, y, kf)
    KNN_model(X, y, kf)