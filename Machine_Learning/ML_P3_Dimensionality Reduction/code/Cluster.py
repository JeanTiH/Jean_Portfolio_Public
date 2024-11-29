""""""
"""OMSCS2023FALL-P3: Dimensionality Reduction	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import pandas as pd
import numpy as np
import math
from scipy import stats

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 30, 'axes.labelsize': 30, 'legend.fontsize': 30})
import seaborn as sns

from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import zscore
from sklearn.neural_network import MLPClassifier
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
'''
#################################################
                    Clustering
                   EM & Kmeans
#################################################
'''
# Step1: Two Pairplots for original data to examine the data nature (linear)
def Data_Nature(X, Y, filename):
    Y = Y.reset_index(drop=True).rename('Target')
    df_X = pd.DataFrame(data=X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
    df_data = pd.concat([df_X, Y], axis=1)
    sns.set(style="ticks")
    sns.pairplot(df_data, hue='Target')
    plt.tight_layout
    plt.savefig('plots/CT/' + filename + '-ori_Features.png')
    plt.close()

    correlation_matrix = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(10, 8))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    matplotlib.rcParams.update({'font.size': 18, 'axes.labelsize': 30, 'legend.fontsize': 18})
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.xlabel('Features', fontsize=30)
    plt.ylabel('Features', fontsize=30)
    plt.title('Correlation Matrix Heatmap with ' + filename)
    plt.tight_layout
    plt.savefig('plots/CT/' + filename + '-ori_CorreMatrix.png')
    plt.close()

    # Gaussian distribution (<0.05 non Gaussian)
    p_value = stats.shapiro(X)[1]
    # Outlier
    z_scores = zscore(X)
    num_outliers_per_feature = np.sum(np.abs(z_scores) > 3, axis=0)
    outlier = np.sum(num_outliers_per_feature)

    return p_value, outlier

# Step2: Silhouette_score & Distortion for Kmeans
def Kmeans_metric(K, X_train, filename, random_state):
    distortions = []
    silhouette_scores = []

    for k in K:
        # Distortion
        clf = KMeans(n_clusters=k, random_state=random_state, n_init=1).fit(X_train)
        distortions.append(clf.inertia_)

        # Silhouette_score
        if k == 1:
            silhouette_scores.append(0)  # Silhouette score undefined for a single cluster
        else:
            labels = clf.predict(X_train)
            silhouette = silhouette_score(X_train, labels)
            silhouette_scores.append(silhouette)

    plt.figure(figsize=(8, 6))
    plt.xlim(0, max(K+1))
    x_ticks = np.arange(1, max(K+2), 2)
    plt.xticks(x_ticks, fontsize=18)
    plt.yticks(fontsize=18)

    plt.plot(K, silhouette_scores, marker='o', color='b', label='Silhouette Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score & Distortion with ' + filename)
    plt.grid(True)

    lines1, labels1 = plt.gca().get_legend_handles_labels()

    ax2 = plt.twinx()
    ax2.tick_params(axis='y', labelsize=18)
    ax2.plot(K, distortions, marker='s', color='g', label='Distortion')
    ax2.set_ylabel('Distortion')
    ax2.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/CT/' + filename + '-Kmeans.png')
    plt.close()

# Step3: AIC & BIC for EM
def EM_metric(K, X_train, filename, random_state):
    # 3.1 Try different covariance_types
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    results_bic = []

    for k in range(1,8):
        for covariance_type in covariance_types:
            clf = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=random_state).fit(X_train)
            bic_score = clf.bic(X_train)
            results_bic.append({'Number of components': k, 'BIC score': bic_score, 'Type of covariance': covariance_type})
            df_bic = pd.DataFrame(results_bic)

    sns.catplot(
        data=df_bic,
        kind="bar",
        x="Number of components",
        y="BIC score",
        hue="Type of covariance",
    )
    plt.savefig('plots/CT/' + filename + '-EM_ParaSelection.png')
    plt.close()

    # 3.2 AIC & BIC based on 'full' (default), which is the best from 3.1
    aic_scores = []
    bic_scores = []

    for k in K:
        clf = GaussianMixture(n_components=k, random_state=random_state).fit(X_train)
        bic_score = clf.bic(X_train)
        bic_scores.append(bic_score)
        aic_score = clf.aic(X_train)
        aic_scores.append(aic_score)

    plt.figure(figsize=(8, 6))
    plt.xlim(0, max(K + 1))
    x_ticks = np.arange(1, max(K + 2), 2)
    plt.xticks(x_ticks, fontsize=18)
    plt.yticks(fontsize=18)

    plt.plot(K, bic_scores, marker='o', color='b', label='BIC Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('BIC Score')
    plt.title('AIC/BIC Score with ' + filename)
    plt.grid(True)

    lines1, labels1 = plt.gca().get_legend_handles_labels()

    ax2 = plt.twinx()
    ax2.tick_params(axis='y', labelsize=18)
    ax2.plot(K, aic_scores, marker='s', color='g', label='AIC Score')
    ax2.set_ylabel('AIC Score')
    ax2.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/CT/' + filename + '-EM.png')
    plt.close()
# Step4: Elbow method, curve, center for Kmeans
def elbow_method(X_train, filename, random_state):
    clf = KMeans(n_init=1, random_state=random_state)
    visualizer = KElbowVisualizer(clf, k=(1, 15))
    visualizer.fit(X_train)
    visualizer.finalize()

    plt.tight_layout()
    plt.savefig('plots/CT/' + filename + '-KMeans_ElbowMethod.png')
    plt.close()

def elbow_curve(K_elbow, X_train, filename, random_state):
    fig_col = 2
    fig_num = len(K_elbow)
    fig_row = math.ceil(fig_num/fig_col)
    fig, ax = plt.subplots(fig_row, fig_col, figsize=(12, 7))

    if fig_row == 1:
        ax = ax.reshape(1, -1)

    position = 0
    for k in K_elbow:
        clf = KMeans(n_clusters=k, n_init=1, random_state=random_state)
        plot_row, plot_col = divmod(position,fig_col)
        visualizer = SilhouetteVisualizer(clf, colors='yellowbrick', ax=ax[plot_row][plot_col])
        visualizer.fit(X_train)
        visualizer.finalize()
        position += 1
    plt.tight_layout()
    plt.savefig('plots/CT/' + filename + '-Kmeans_Elbow.png')
    plt.close()

def center(K_elbow, X_train, filename, random_state):
    fig_col = 2
    fig_num = len(K_elbow)
    fig_row = math.ceil(fig_num / fig_col)
    fig, ax = plt.subplots(fig_row, fig_col, figsize=(12, 7))

    if fig_row == 1:
        ax = ax.reshape(1, -1)

    position = 0
    for k in K_elbow:
        clf = KMeans(n_clusters=k, n_init=1, random_state=random_state)
        plot_row, plot_col = divmod(position, fig_col)
        visualizer = InterclusterDistance(clf, colors='yellowbrick', ax=ax[plot_row][plot_col])
        visualizer.fit(X_train)
        visualizer.finalize()

        position += 1
    plt.tight_layout()
    plt.savefig('plots/CT/' + filename + '-Kmeans_Center.png')
    plt.close()
# Step5: Heatmap for EM & Kmeans
def cm(true_labels, cluster_labels, filename, clf_name):
    cm = confusion_matrix(true_labels, cluster_labels)
    plt.figure(figsize=(8, 6))
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j + 0.5, i + 0.5, str(cm[i, j]), ha='center', va='center', color='gray')

    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Clustering Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix with ' + clf_name + ' on ' + filename)
    plt.tight_layout
    plt.savefig('plots/CT/' + filename + '-' + clf_name + '_Confusion.png')
    plt.close()
# Step6: Output metrics
def Cluster_Metrics(K_optimal, X_train, Y_train, X_test, Y_test, clf_name, filename, CT_metric_filename, p_value, outlier, random_state):
    if clf_name == 'EM':
        clf = GaussianMixture(n_components=K_optimal, random_state=random_state)
    elif clf_name == 'KMeans':
        clf = KMeans(n_clusters=K_optimal, random_state=random_state, n_init=1)

    train_labels = clf.fit_predict(X_train)
    test_labels = clf.predict(X_test)
    # Check train_labels and the true ground labels
    if K_optimal== 2:
        cm(Y_train, train_labels, filename, clf_name)
    ari = adjusted_rand_score(Y_train, train_labels)
    nmi = normalized_mutual_info_score(Y_train, train_labels)
    hom, com, v = homogeneity_completeness_v_measure(Y_train, train_labels)

    # Add new feature (labels) to X and run NN
    train_labels = train_labels.reshape((-1, 1))
    X_train_labeled = np.hstack((X_train, train_labels))

    test_labels = test_labels.reshape((-1, 1))
    X_test_labeled = np.hstack((X_test, test_labels))

    accuracy_origin, f1_origin = NNEval(X_train, Y_train, X_test, Y_test, random_state)
    accuracy_labels, f1_labels = NNEval(X_train_labeled, Y_train, X_test_labeled, Y_test, random_state)

    with open(CT_metric_filename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{filename} with p_value = ' + str(p_value) + ' and outlier = ' + str(outlier) + '\n')
        fp.write(f'***********************************' + '\n')
        fp.write('Algorithm: ' +  clf_name + '\n')
        fp.write('Number of Clusters: ' + str(K_optimal) + '\n')
        fp.write('Adjusted Rand Index (ARI)   : ' + str(ari) + '\n')
        fp.write('norm_mutual_info_score (NMI): ' + str(nmi) + '\n')
        fp.write('homogeneity                 : ' + str(hom) + '\n')
        fp.write('completeness                : ' + str(com) + '\n')
        fp.write('v_measure                   : ' + str(v) + '\n')
        fp.write('Original Accuracy + f1_score: ' + str(accuracy_origin) + ', ' + str(f1_origin) + '\n')
        fp.write('Labeled  Accuracy + f1_score: ' + str(accuracy_origin) + ', ' + str(f1_labels) + '\n')
# Step7: NN learner for evaluation
def NNEval(X_train, Y_train, X_test, Y_test, random_state):
    model = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=random_state)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuray = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    return accuray, f1

def heatmap(k, X_train, filename, clf_name, random_state):
    if clf_name == 'EM':
        clf = GaussianMixture(n_components=k, random_state=random_state)
    elif clf_name == 'KMeans':
        clf = KMeans(n_clusters=k, random_state=random_state)
    clf.fit(X_train)
    cluster_labels = clf.fit_predict(X_train)

    HeatMap = pd.DataFrame({
        'Feature1': X_train[:, 0],
        'Feature2': X_train[:, 1],
        'Cluster': cluster_labels
    })
    corr = HeatMap['Feature1'].corr(HeatMap['Feature2'], method='spearman')
    sns.set(style='ticks', font_scale = 0.9)
    pairplot = sns.pairplot(HeatMap, hue='Cluster', palette='Dark2', x_vars=['Feature1'], y_vars=['Feature2'], plot_kws={'alpha': 0.7})
    pairplot.fig.suptitle(clf_name + ' on ' + filename, y=1, fontsize=8)
    plt.tight_layout
    plt.savefig('plots/CT/' + filename + '-Heatmap-' + clf_name + '.png', dpi=300)
    plt.close()
'''
Run functions
'''
def Cluster_Eval(K, K_optimal, X, Y, X_train, Y_train, X_test, Y_test, filename, CT_metric_filename, random_state):
    # Data nature: correlation matrix, Gaussian distribution (p), Outliers
    p_value, outlier = Data_Nature(X, Y, filename)

    Kmeans_metric(K, X_train, filename, random_state)
    EM_metric(K, X_train, filename, random_state)

    # Metric based on Kmeans_metric & EM_metric
    clf_names = ['EM', 'KMeans']
    for i, clf_name in enumerate(clf_names):
        Cluster_Metrics(K_optimal[i], X_train, Y_train, X_test, Y_test, clf_name, filename, CT_metric_filename, p_value, outlier, random_state)
        heatmap(K_optimal[i], X_train, filename, clf_name, random_state)
    # Only work for KMeans
    elbow_method(X_train, filename, random_state)
    # Metric based on Kmeans_metric
    K_elbow = [2, 3]
    elbow_curve(K_elbow, X_train, filename, random_state)
    center(K_elbow, X_train, filename, random_state)
