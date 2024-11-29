""""""
"""OMSCS2023FALL-P3: Dimensionality Reduction	   		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 30, 'axes.labelsize': 30, 'legend.fontsize': 30})
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection as SRP
from sklearn.manifold import Isomap
from scipy.stats import kurtosis
import scipy.stats
import scipy.sparse as sps
from scipy.linalg import pinv

from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

'''
#################################################
            Dimensionality Reduction
               PCA, ICA, SRP, IMP
#################################################
'''
# NN learner for evaluation
def NNEval(clf, X_train, Y_train, X_test, Y_test, random_state, ica_p):
    model = MLPClassifier(hidden_layer_sizes=(15, 15), random_state=random_state)
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)
    f1_ori_test = f1_score(Y_test, Y_pred_test)

    X_train_trans = clf.fit_transform(X_train)
    X_test_trans = clf.transform(X_test)
    if ica_p == None:
        model.fit(X_train_trans, Y_train)
        Y_pred_train = model.predict(X_train_trans)
        Y_pred_test = model.predict(X_test_trans)
    else:
        model.fit(X_train_trans[:, ica_p], Y_train)
        Y_pred_train = model.predict(X_train_trans[:, ica_p])
        Y_pred_test = model.predict(X_test_trans[:, ica_p])

    f1_dr_train = f1_score(Y_train, Y_pred_train)
    f1_dr_test = f1_score(Y_test, Y_pred_test)

    return f1_ori_test, f1_dr_train, f1_dr_test

def NNEval_clf(X_train, Y_train, X_test, Y_test, filename, feature_num, random_state):
    f1_pca_train = []
    f1_pca_test = []
    f1_ica_train = []
    f1_ica_test = []
    f1_srp_train = []
    f1_srp_test = []
    f1_imp_train = []
    f1_imp_test = []
    for i in np.arange(1, feature_num+1):

        clf = PCA(n_components=i, random_state=random_state)
        f1_ori_test, f1_dr_train, f1_dr_test = NNEval(clf, X_train, Y_train, X_test, Y_test, random_state, None)
        f1_pca_train.append(f1_dr_train)
        f1_pca_test.append(f1_dr_test)

        clf = FastICA(n_components=i, random_state=random_state)
        f1_ori_test, f1_dr_train, f1_dr_test = NNEval(clf, X_train, Y_train, X_test, Y_test, random_state, None)
        f1_ica_train.append(f1_dr_train)
        f1_ica_test.append(f1_dr_test)

        clf = SRP(n_components=i, random_state=random_state)
        f1_ori_test, f1_dr_train, f1_dr_test = NNEval(clf, X_train, Y_train, X_test, Y_test, random_state, None)
        f1_srp_train.append(f1_dr_train)
        f1_srp_test.append(f1_dr_test)

        clf = Isomap(n_components=i)
        f1_ori_test, f1_dr_train, f1_dr_test = NNEval(clf, X_train, Y_train, X_test, Y_test, random_state, None)
        f1_imp_train.append(f1_dr_train)
        f1_imp_test.append(f1_dr_test)

    with open('DimRedcution_metric.txt', 'a') as fp:
        fp.write(f'***********************************' + '\n')
        fp.write(f'{filename}\n')
        fp.write(f'***********************************' + '\n')
        fp.write('ORI F1 score test : ' + str(f1_ori_test) + '\n')
        #fp.write('PCA F1 score train: ' + str(f1_pca_train) + '\n')
        fp.write('PCA F1 score test : ' + str(f1_pca_test) + '\n')
        #fp.write('ICA F1 score train: ' + str(f1_ica_train) + '\n')
        fp.write('ICA F1 score test : ' + str(f1_ica_test) + '\n')
        #fp.write('SRP F1 score train: ' + str(f1_srp_train) + '\n')
        fp.write('SRP F1 score test : ' + str(f1_srp_test) + '\n')
        #fp.write('IMP F1 score train: ' + str(f1_imp_train) + '\n')
        fp.write('IMP F1 score test : ' + str(f1_imp_test) + '\n')

def ReconstructionError(Xp, X):
    if sps.issparse(Xp):
        Xp = Xp.todense()
    p = pinv(Xp)
    reconstructed_data = ((p@Xp)@(X.T)).T
    r_errors = np.square(X-reconstructed_data)
    r_error = np.nanmean(r_errors)
    return r_error
'''
-----------------------------
            PCA
-----------------------------
'''
def PCA_Eval(X_train, Y_train, X_test, Y_test, filename, feature_num, random_state):
    pca = PCA(random_state=random_state).fit(X_train)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    eigenvalues = pca.explained_variance_
    explained_variance = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    d = np.argmax(cumulative_variance >= 0.9) + 1

    # EV & SV
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, color='b', label='Explained Variance')
    plt.xlabel('Number of Components', fontsize=30)
    plt.ylabel('Explained Variance', fontsize=30)
    plt.title('PCA Explained Variance and Singular Values on ' + filename)
    plt.grid(True)

    lines1, labels1 = plt.gca().get_legend_handles_labels()

    ax2 = plt.twinx()
    ax2.tick_params(axis='y', labelsize=18)
    ax2.plot(range(1, len(singular_values) + 1), singular_values, color='g', label='Singular Values')
    ax2.set_ylabel('Singular Values')
    ax2.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=18)
    plt.tight_layout()
    plt.savefig('plots/DR/' + filename + '-PCA_EVariance.png')
    plt.close()

    # CEV & Eigenvalues
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color='b', label='Cumulative Explained Variance')
    plt.xlabel('Number of Components', fontsize=30)
    plt.ylabel('Cumulative EV', fontsize=30)
    plt.title('PCA Cumulative Explained Variance and Eigenvalues on ' + filename)
    plt.grid(True)

    lines1, labels1 = plt.gca().get_legend_handles_labels()

    ax2 = plt.twinx()
    ax2.tick_params(axis='y', labelsize=18)
    ax2.plot(range(1, len(eigenvalues) + 1), eigenvalues, color='g', label='Eigenvalues')
    ax2.set_ylabel('Eigenvalues', fontsize=30)
    ax2.grid(False)

    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/DR/' + filename + '-PCA_Eigenvalues.png')
    plt.close()

    # Reconstructed Error
    r_error = []
    for i in np.arange(1, feature_num):
        pca = PCA(n_components=i, random_state=random_state)
        pca.fit_transform(X_train)
        Xp = pca.components_
        r_error.append(ReconstructionError(Xp, X_train))

    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.bar(range(1, len(r_error) + 1), r_error, tick_label=range(1, len(r_error) + 1))
    plt.plot(range(1, len(r_error) + 1), r_error)
    plt.xlabel('Component', fontsize=30)
    plt.ylabel('Reconstructed Error', fontsize=30)
    plt.title('PCA Reconstructed Error of Components on ' + filename)
    plt.tight_layout()
    plt.savefig('plots/DR/' + filename + '-PCA_RError.png')
    plt.close()

'''
-----------------------------
            ICA
-----------------------------
'''
def ICA_Eval(X_train, Y_train, X_test, Y_test, filename, feature_num, random_state):

    if filename == 'Data_Diabetes':
        optimal_component = 2
    elif filename == 'Data_Rice':
        optimal_component = 3

    mean_absolute_kurtosis = []
    for i in range(1, feature_num + 1):
        ica_components = FastICA(n_components=i, max_iter=2000, random_state=random_state).fit_transform(X_train)
        # Calculate kurtosis for each component
        component_kurtosis = [scipy.stats.kurtosis(ica_components[:, j]) for j in range(i)]
        # Calculate the mean of absolute kurtosis values
        mean_abs_kurtosis = np.mean(np.abs(component_kurtosis))
        mean_absolute_kurtosis.append(mean_abs_kurtosis)

    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.bar(range(1, len(mean_absolute_kurtosis) + 1), mean_absolute_kurtosis, tick_label=range(1, len(mean_absolute_kurtosis) + 1))
    plt.plot(range(1, len(mean_absolute_kurtosis) + 1), mean_absolute_kurtosis)
    plt.xlabel('Independent Component', fontsize=30)
    plt.ylabel('Mean Abs Kurtosis', fontsize=30)
    plt.title('ICA Kurtosis of Independent Components on ' + filename)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('plots/DR/' + filename + '-ICA_Kurtosis_mean.png')
    plt.close()
'''
-----------------------------
            SRP
-----------------------------
'''
def SRP_Eval(X_train, Y_train, X_test, Y_test, filename, feature_num, random_state):

    if filename == 'Data_Diabetes':
        optimal_component = 5
    elif filename == 'Data_Rice':
        optimal_component = 3

    r_error = []
    for i in np.arange(1, feature_num):
        srp = SRP(n_components=i, random_state=random_state)
        srp.fit_transform(X_train)
        Xp = srp.components_
        r_error.append(ReconstructionError(Xp, X_train))

    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(range(1, len(r_error) + 1), r_error)
    plt.xlabel('Component', fontsize=30)
    plt.ylabel('Reconstructed Error', fontsize=30)
    plt.title('SRP Reconstructed Error of Components on ' + filename)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('plots/DR/' + filename + '-SRP_RError.png')
    plt.close()
'''
-----------------------------
            IsoMap
-----------------------------
'''
def IMP_Eval(X_train, Y_train, X_test, Y_test, filename, feature_num):

    r_error = []
    for i in np.arange(1, feature_num):
        imp = Isomap(n_components=i)
        imp.fit_transform(X_train)
        r_error.append(imp.reconstruction_error())

    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(range(1, len(r_error) + 1), r_error)
    plt.xlabel('Component', fontsize=30)
    plt.ylabel('Reconstructed Error', fontsize=30)
    plt.title('IMP Reconstructed Error of Components on ' + filename)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('plots/DR/' + filename + '-IMP_RError.png')
    plt.close()

def pairplots_corrmatrix(X_train, Y_train, X_test, Y_test, filename, n_components, random_state):
    DR_names = ['PCA', 'ICA', 'SRP', 'IMP']
    for i, DR_name in enumerate(DR_names):
        if DR_name == 'PCA':
            clf = PCA(n_components=n_components[i], random_state=random_state)
        elif DR_name == 'ICA':
            clf = FastICA(n_components=n_components[i], random_state=random_state)
        elif DR_name == 'SRP':
            clf = SRP(n_components=n_components[i], random_state=random_state)
        elif DR_name == 'IMP':
            clf = Isomap(n_components=n_components[i])

        X_dr = clf.fit_transform(X_train)

        Y = Y_train.reset_index(drop=True).rename('Target')
        df_X = pd.DataFrame(data=X_dr, columns=[f'Component{i+1}' for i in range(X_dr.shape[1])])
        df_data = pd.concat([df_X, Y], axis=1)

        df_data_subset = df_data[['Component1', 'Component2', 'Target']]
        sns.set(style="ticks", font_scale = 0.9)
        #sns.pairplot(df_data, hue='Target')
        sns.pairplot(df_data_subset, hue='Target', palette='Dark2', x_vars=['Component1'], y_vars=['Component2'], plot_kws={'alpha': 0.7})
        plt.tight_layout
        plt.savefig('plots/DR/Features-' + filename + '_' + DR_name + '.png')
        plt.close()

        '''
        correlation_matrix = np.corrcoef(X_dr, rowvar=False)
        plt.figure(figsize=(10, 8))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        matplotlib.rcParams.update({'font.size': 18, 'axes.labelsize': 30, 'legend.fontsize': 18})
        ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        plt.xlabel('Features', fontsize=30)
        plt.ylabel('Features', fontsize=30)
        plt.title(DR_name + ' Correlation Matrix Heatmap with ' + filename)
        plt.tight_layout
        plt.savefig('plots/DR/CorreMatrix-' + filename + '_' + DR_name + '.png')
        plt.close()
        '''