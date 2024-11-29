""""""
"""OMSCS2023FALL-P3: Dimensionality Reduction	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import math
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'legend.fontsize': 16})
import seaborn as sns

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection as SRP
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure, confusion_matrix, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
'''
#################################################
      Dimensionality Reduction + Clustering
         PCA, ICA, SRP, IMP + EM, KMeans
#################################################
'''
# Step1: Silhouette_score & Distortion for Kmeans
def Kmeans_metric(K, X_train, filename, DR_name, random_state):
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

    matplotlib.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'legend.fontsize': 16})
    plt.figure(figsize=(8, 6))
    plt.xlim(0, max(K+1))
    x_ticks = np.arange(1, max(K+2), 2)
    plt.xticks(x_ticks)

    plt.plot(K, silhouette_scores, marker='o', color='b', label='Silhouette Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title(DR_name + ' Silhouette Score & Distortion with ' + filename)
    plt.grid(True)

    lines1, labels1 = plt.gca().get_legend_handles_labels()

    ax2 = plt.twinx()
    ax2.plot(K, distortions, marker='s', color='g', label='Distortion')
    ax2.set_ylabel('Distortion')
    ax2.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.tight_layout()
    plt.savefig('plots/CDR/' + filename + '-' + DR_name + '_Kmeans.png')
    plt.close()

# Step2: AIC & BIC for EM
def EM_metric(K, X_train, filename, DR_name, random_state):
    # 2.1 Try different covariance_types
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
    plt.savefig('plots/CDR/' + filename + '-' + DR_name + '_EM_ParaSelection.png')
    plt.close()

    # 2.2 AIC & BIC based on 'full' (default), which is the best from 3.1
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
    plt.xticks(x_ticks)

    plt.plot(K, bic_scores, marker='o', color='b', label='BIC Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('BIC Score')
    plt.title('AIC/BIC Score with ' + filename)
    plt.grid(True)

    lines1, labels1 = plt.gca().get_legend_handles_labels()

    ax2 = plt.twinx()
    ax2.plot(K, aic_scores, marker='s', color='g', label='AIC Score')
    ax2.set_ylabel('AIC Score')
    ax2.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
    plt.tight_layout()
    plt.savefig('plots/CDR/' + filename + '-' + DR_name + '_EM.png')
    plt.close()
# Step3: Elbow method, curve, center for Kmeans
def elbow_method(X, filename, DR_name, random_state):
    clf = KMeans(n_init=1, random_state=random_state)
    visualizer = KElbowVisualizer(clf, k=(1, 15))
    visualizer.fit(X)
    visualizer.finalize()

    plt.tight_layout()
    plt.savefig('plots/CDR/' + filename + '-' + DR_name + '_KMeans_ElbowMethod.png')
    plt.close()

def elbow_curve(K_elbow, X, filename, DR_name, random_state):
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
        visualizer.fit(X)
        visualizer.finalize()
        position += 1
    plt.tight_layout()
    plt.savefig('plots/CDR/' + filename + '-' + DR_name + '_KmeansElbow.png')
    plt.close()

def center(K_elbow, X, filename, DR_name, random_state):
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
        visualizer.fit(X)
        visualizer.finalize()

        position += 1
    plt.tight_layout()
    plt.savefig('plots/CDR/' + filename + '-' + DR_name + '_KmeansCenter.png')
    plt.close()
# Step4: Heatmap for EM & Kmeans
def cm(true_labels, cluster_labels, filename, clf_name, DR_name):
    cm = confusion_matrix(true_labels, cluster_labels)
    matplotlib.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'legend.fontsize': 16})
    plt.figure(figsize=(8, 6))
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j + 0.5, i + 0.5, str(cm[i, j]), ha='center', va='center', color='gray')

    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Clustering Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix with ' + clf_name + ' on ' + filename)
    plt.savefig('plots/CDR/' + filename + '-Confusion-' + clf_name + '_' + DR_name + '.png')
    plt.close()
# Step5: Output metrics
def Cluster_Metrics(K_optimal, X_train, Y_train, X_test, Y_test, clf_name, filename, DR_name, CDR_metric_filename, corr, random_state):
    if clf_name == 'EM':
        clf = GaussianMixture(n_components=K_optimal, random_state=random_state)
    elif clf_name == 'KMeans':
        clf = KMeans(n_clusters=K_optimal, random_state=random_state, n_init=1)
    train_labels = clf.fit_predict(X_train)
    test_labels = clf.predict(X_test)
    '''
    if K_optimal== 2:
        cm(Y_train, train_labels, filename, clf_name, DR_name)
    '''
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

    with open(CDR_metric_filename, 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{filename}\n')
        fp.write(f'***********************************' + '\n')
        fp.write('Algorithm: ' +  clf_name + '_' + DR_name + '  First two components corr = ' + str(corr) + '\n')
        fp.write('Number of Clusters: ' + str(K_optimal) + '\n')
        fp.write('Adjusted Rand Index (ARI)   : ' + str(ari) + '\n')
        fp.write('norm_mutual_info_score (NMI): ' + str(nmi) + '\n')
        fp.write('homogeneity                 : ' + str(hom) + '\n')
        fp.write('completeness                : ' + str(com) + '\n')
        fp.write('v_measure                   : ' + str(v) + '\n')
        fp.write('Original Accuracy + f1_score: ' + str(accuracy_origin) + ', ' + str(f1_origin) + '\n')
        fp.write('Labeled  Accuracy + f1_score: ' + str(accuracy_labels) + ', ' + str(f1_labels) + '\n')
# Step6: Heatmap for first 2 components
def heatmap(X_trans, k, filename, DR_name, clf_name, random_state):
    if clf_name == 'EM':
        clf = GaussianMixture(n_components=k, random_state=random_state)
    elif clf_name == 'KMeans':
        clf = KMeans(n_clusters=k, random_state=random_state)
    clf.fit(X_trans)
    cluster_labels = clf.fit_predict(X_trans)

    if DR_name == 'SRP' and filename == 'Data_Rice':
        HeatMap = pd.DataFrame({
            'Component1': X_trans[:, 1],
            'Component2': X_trans[:, 2],
            'Cluster': cluster_labels
        })
    else:
        HeatMap = pd.DataFrame({
            'Component1': X_trans[:, 0],
            'Component2': X_trans[:, 1],
            'Cluster': cluster_labels
        })
    corr = HeatMap['Component1'].corr(HeatMap['Component2'], method='spearman')
    sns.set(style='ticks', font_scale = 0.9)
    pairplot = sns.pairplot(HeatMap, hue='Cluster', palette='Dark2', x_vars=['Component1'], y_vars=['Component2'], plot_kws={'alpha': 0.7})
    #pairplot.fig.suptitle(DR_name + ' with ' + clf_name + ' on ' + filename, y=1, fontsize=8)
    plt.tight_layout
    plt.savefig('plots/CDR/' + filename + '-Heatmap-' + clf_name + '_' + DR_name + '.png', dpi=300)
    plt.close()

    '''
    pairplot = sns.pairplot(HeatMap, hue='Cluster', palette='Dark2', plot_kws={'alpha': 0.7})
    pairplot.fig.suptitle(DR_name + ' with ' + clf_name + ' on ' + filename, y=1, fontsize=8)
    plt.tight_layout
    plt.savefig('plots/CDR/' + filename + '-HeatmapWhole-' + clf_name + '_' + DR_name + '.png', dpi=300)
    plt.close()
    '''
    return corr
# Step7: NN learner for evaluation
def NNEval(X_train, Y_train, X_test, Y_test, random_state):
    model = MLPClassifier(hidden_layer_sizes=(20, 20), random_state=random_state)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuray = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    return accuray, f1
'''
Run functions
'''
def DR_cluster(X_train, Y_train, X_test, Y_test, filename, n_components, K, CDR_metric_filename, random_state):
    for DR_name in ['PCA', 'ICA', 'SRP', 'IMP']:
        if DR_name == 'PCA':
            clf = PCA(n_components=n_components[0], random_state=random_state)
        elif DR_name == 'ICA':
            clf = FastICA(n_components=n_components[1], random_state=random_state)
        if DR_name == 'SRP':
            clf = SRP(n_components=n_components[2], random_state=random_state)
        if DR_name == 'IMP':
            clf = Isomap(n_components=n_components[3])

        X_train_trans = clf.fit_transform(X_train)
        X_test_trans = clf.transform(X_test)

        for clf_name in ['EM', 'KMeans']:
            #scaler = MinMaxScaler()
            #X_trans = scaler.fit_transform(X_trans)
            Kmeans_metric(K, X_train_trans, filename, DR_name, random_state)
            EM_metric(K, X_train_trans, filename, DR_name, random_state)
            # Only work for KMeans
            elbow_method(X_train_trans, filename, DR_name, random_state)

            # Metric based on Kmeans_metric & EM_metric
            K_optimal = 2
            if filename == 'Data_Diabetes':
                if clf_name == 'EM':
                    if DR_name == 'PCA':
                        K_optimal = 8
                    elif DR_name == 'ICA':
                        K_optimal = 6
                    elif DR_name == 'SRP':
                        K_optimal = 5
                    elif DR_name == 'IMP':
                        K_optimal = 11
            elif filename == 'Data_Rice':
                if DR_name == 'PCA' and clf_name == 'EM':
                    K_optimal = 5
                elif DR_name == 'ICA' and clf_name == 'KMeans':
                    K_optimal = 6
                elif DR_name == 'ICA' and clf_name == 'EM':
                    K_optimal = 13
                elif DR_name == 'SRP' and clf_name == 'EM':
                    K_optimal = 9
                elif DR_name == 'IMP' and clf_name == 'EM':
                    K_optimal = 8
            corr = heatmap(X_train_trans, K_optimal, filename, DR_name, clf_name, random_state)  # Heatmap for first 2 components
            Cluster_Metrics(K_optimal, X_train_trans, Y_train, X_test_trans, Y_test, clf_name, filename, DR_name, CDR_metric_filename, corr, random_state)

            '''
            # Only work for KMeans & Metric based on Kmeans_metric
            if clf_name == 'KMeans':
                K_elbow = [2, 3, 6, 9]
                elbow_curve(K_elbow, X_train, filename, DR_name, random_state)
                center(K_elbow, X_train, filename, DR_name, random_state)
            '''