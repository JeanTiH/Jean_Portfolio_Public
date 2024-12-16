# Project
# ISyE 6420: Fall 2024
# Juejing Han, jhan446@gatech.edu

# some functions/code are inspired from online resources.
# https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-robust.html
# https://www.pymc.io/projects/bart/en/latest/examples/bart_introduction.html

import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import pymc_bart as pmb
import statsmodels.api as sm
import matplotlib.lines as mlines
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

import warnings
# Suppress RuntimeWarnings & FutureWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from scipy.stats import pearsonr

'''
***********************************************************
                    Part 1. Generate Data
***********************************************************
'''
def generate_data(outlier_count, seed):
    np.random.seed(seed)
    datasize = 100
    true_alpha = 2
    true_beta = 3

    # Generate x and y, add noise
    x = np.linspace(0, 1, datasize)
    true_regression_line = true_alpha + true_beta * x
    noise = np.random.normal(scale=0.5, size=datasize)
    y = true_regression_line + noise

    # Add random outliers
    outlier_size = outlier_count
    x_outliers = np.random.uniform(0, 1, size=outlier_size)
    y_outliers = np.random.uniform(6, 9.5, size=outlier_size)
    x_data = np.append(x, x_outliers)
    y_data = np.append(y, y_outliers)

    total_datasize = datasize + outlier_size
    correlation, p_value = pearsonr(x_data.flatten(), y_data)
    print('******************************************************************************')
    print(f'                      Data Info (total datasize = {total_datasize})')
    print(f'                             Outlier count: {outlier_count}')
    print(f'          Pearson Correlation Coefficient: {correlation:.4f}, p-value: {p_value:.4f}')
    print('******************************************************************************')

    # Plot data
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.scatter(x_data, y_data, c='g', alpha=0.6, label='Data Points')
    plt.plot(x, true_regression_line, c='r', lw=3, label='True Regression Line')
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig('Data.png', bbox_inches='tight')
    plt.close()

    return x, true_regression_line, x_data, y_data
'''
***********************************************************
                Part 2. Stats & Plots Check
***********************************************************
'''
def model_stats(trace, ppc, y_data, model_name):
    print('--------------------------------------------------------')
    print(f'                {model_name}                           ')
    print('--------------------------------------------------------')
    # Summary stats
    if 'BART' not in model_name:
        res = az.summary(trace, var_names=['alpha', 'beta'], hdi_prob=0.95)
        print(res)

    # Bayesian R²
    y_pred = ppc.posterior_predictive.stack(sample=('chain', 'draw'))['likelihood'].values
    score = az.r2_score(y_data, y_pred.T)
    print(score.round(3))

    # Errors
    y_pred_mean = y_pred.mean(axis=1)
    mae = round(mean_absolute_error(y_data, y_pred_mean), 3)
    medae = round(median_absolute_error(y_data, y_pred_mean), 3)
    rmse = round(np.sqrt(mean_squared_error(y_data, y_pred_mean)), 3)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Median Absolute Error (MedAE): {medae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # LOO_CV and WAIC
    print('#######################################')
    loo_res = az.loo(trace)
    waic_res = az.waic(trace)
    print(f'LOO_CV: {loo_res}')
    print(f'WAIC: {waic_res}')
    print('#######################################')
    print(f'-------------------------END OF {model_name}----------------------------')

def check_regression_line(trace, x_data, y_data, x, true_regression_line, model_name, seed):
    extrat_sample = az.extract(trace, num_samples=25, rng=seed)
    x_values = np.linspace(x_data.min(), x_data.max(), 100)

    beta_samples = extrat_sample["beta"].values.reshape(-1, 1)
    regression_lines = extrat_sample["alpha"].values[:, None] + beta_samples * x_values

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.scatter(x_data, y_data, c='g', alpha=0.6, label='Data Points')

    for line in regression_lines:
        plt.plot(x_values, line.T, c='royalblue', alpha=0.4)
    plt.plot([],[], c='b', label=f'{model_name} Line')

    plt.plot(x, true_regression_line, c='r', lw=3, label='True Regression Line')
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(f'Check_{model_name}.png', bbox_inches='tight')
    plt.close()

def check_BART(ppc, x_data, y_data, x, true_regression_line, model_name):
    y_pred = ppc.posterior_predictive.stack(sample=('chain', 'draw'))['likelihood'].values
    y_pre = y_pred.mean(axis=1)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.plot(x_data.flatten(), y_pre, 'x', c='royalblue', lw=2, markersize=6, label=f'{model_name} Estimates')
    # Plot the 95% HDI
    az.plot_hdi(x_data.flatten(), y_pred.T, hdi_prob=0.95, smooth=True, color='lightblue')
    plt.scatter(x_data, y_data, c='g', alpha=0.6, label="Data Points")
    plt.plot(x, true_regression_line, c='r', lw=3, label="True Regression Line")
    hdi_legend = mlines.Line2D([], [], color='lightblue', alpha=0.5, lw=4, label='95% HDI')
    plt.legend(handles=[hdi_legend, *plt.gca().get_legend_handles_labels()[0]], loc='lower right', fontsize=14)
    plt.savefig(f'Check_{model_name}.png', bbox_inches='tight')
    plt.close()
'''
***********************************************************
                        Part 3. Models
***********************************************************
'''
'''
Model 1 - Frequentist Linear Regression
'''
def frequentist_linear_regression(x, true_regression_line, x_data, y_data, model_name):
    # Add intercept
    x_with_const = sm.add_constant(x_data)
    model = sm.OLS(y_data, x_with_const)
    res = model.fit()

    # Summary stats
    print(res.summary())

    # Metrics
    y_pred = res.predict(x_with_const)
    mae = round(mean_absolute_error(y_data, y_pred), 3)
    medae = round(median_absolute_error(y_data, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_data, y_pred)), 3)
    aic = round(res.aic, 3)
    bic = round(res.bic, 3)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Median Absolute Error (MedAE): {medae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Akaike Information Criterion (AIC): {aic}")
    print(f"Bayesian Information Criterion (BIC): {bic}")

    alpha, beta = res.params
    regression_line = alpha + beta * x

    # Check regression line
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.scatter(x_data, y_data, c='g', alpha=0.6, label='Data Points')
    plt.plot(x, regression_line, c='royalblue', lw=2, alpha=0.8, label=f'{model_name} Line')
    plt.plot(x, true_regression_line, c='r', lw=3, label='True Regression Line')
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(f'Check_{model_name}.png', bbox_inches='tight')
    plt.close()

'''
Model 2 - Bayesian Linear Regression
'''
def Bayesian_linear_regression(x, true_regression_line, x_data, y_data, draws, model_name, seed):
    with pm.Model() as standard_bayesian:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=1)
        mu = alpha + beta * x_data

        sigma = pm.HalfNormal('sigma', sigma=1)

        # Gaussian likelihood
        pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_data)

        # Sampling
        trace = pm.sample(draws=draws, target_accept=0.9, idata_kwargs={"log_likelihood": True}, random_seed=seed)
        ppc = pm.sample_posterior_predictive(trace, random_seed=seed)

    model_stats(trace, ppc, y_data, model_name)
    check_regression_line(trace, x_data, y_data, x, true_regression_line, model_name, seed)
    graph = pm.model_to_graphviz(standard_bayesian)
    graph.render(filename=f'model_graph_{model_name}', format='png', view=False)
'''
Model 3 - Robust Bayesian Linear Regression
'''
def Robust_Bayesian_linear_regression(x, true_regression_line, x_data, y_data, draws, model_name, seed):
    with pm.Model() as robust_bayesian:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=1)
        mu = alpha + beta * x_data

        sigma = pm.HalfNormal('sigma', sigma=1)

        # Student-t likelihood
        nu = pm.Exponential('nu', lam=1 / 30)  # Degrees of freedom, promoting heavier tails
        pm.StudentT('likelihood', mu=mu, sigma=sigma, nu=nu, observed=y_data)

        # Sampling
        trace = pm.sample(draws=draws, target_accept=0.9, idata_kwargs={"log_likelihood": True}, random_seed=seed)
        ppc = pm.sample_posterior_predictive(trace, random_seed=seed)

    model_stats(trace, ppc, y_data, model_name)
    check_regression_line(trace, x_data, y_data, x, true_regression_line, model_name, seed)
    graph = pm.model_to_graphviz(robust_bayesian)
    graph.render(filename=f'model_graph_{model_name}', format='png', view=False)
'''
Model 4 - Mixture Bayesian Linear Regression
'''
def Mixture_Bayesian_linear_regression(x, true_regression_line, x_data, y_data, draws, model_name, seed):
    with pm.Model() as mixture_bayesian:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=1)
        mu = alpha + beta * x_data

        # Outlier detection: Mixture model (Low prior probability of being an outlier)
        is_outlier = pm.Bernoulli('is_outlier', p=0.09, shape=y_data.shape[0])
        sigma_inlier = pm.HalfNormal('sigma_inlier', sigma=1)
        sigma_outlier = pm.HalfNormal('sigma_outlier', sigma=10)

        # Mixture likelihood
        pm.Mixture(
            'likelihood',
            w=pm.math.stack([1 - is_outlier, is_outlier]).T,
            comp_dists=[
                pm.Normal.dist(mu=mu, sigma=sigma_inlier),
                pm.Normal.dist(mu=mu, sigma=sigma_outlier)
            ],
            observed=y_data
        )

        # Sampling
        trace = pm.sample(draws=draws, target_accept=0.9, idata_kwargs={"log_likelihood": True}, random_seed=seed)
        ppc = pm.sample_posterior_predictive(trace, random_seed=seed)

    model_stats(trace, ppc, y_data, model_name)
    check_regression_line(trace, x_data, y_data, x, true_regression_line, model_name, seed)
    graph = pm.model_to_graphviz(mixture_bayesian)
    graph.render(filename=f'model_graph_{model_name}', format='png', view=False)

'''
Model 5 - Standard BART Regression
'''
def BART_regression(x, true_regression_line, x_data, y_data, draws, model_name, seed):
    x_data = x_data.reshape(-1, 1)
    with pm.Model() as standard_bart:
        mu = pmb.BART('mu', X=x_data, Y=y_data)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Gaussian likelihood
        pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_data)

        # Sampling
        trace = pm.sample(draws=draws, target_accept=0.9, idata_kwargs={"log_likelihood": True}, random_seed=seed)
        ppc = pm.sample_posterior_predictive(trace, random_seed=seed)

    model_stats(trace, ppc, y_data, model_name)
    check_BART(ppc, x_data, y_data, x, true_regression_line, model_name)
    graph = pm.model_to_graphviz(standard_bart)
    graph.render(filename=f'model_graph_{model_name}', format='png', view=False)

'''
Model 6 - Robust BART Regression
'''
def Robust_BART_regression(x, true_regression_line, x_data, y_data, draws, model_name, seed):
    x_data = x_data.reshape(-1, 1)
    with pm.Model() as robust_bart:
        mu = pmb.BART('mu', X=x_data, Y=y_data, m=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Student-t likelihood
        nu = pm.Exponential('nu', lam=1 / 30)  # Degrees of freedom, promoting heavier tails
        pm.StudentT('likelihood', mu=mu, sigma=sigma, nu=nu, observed=y_data)

        # Sampling
        trace = pm.sample(draws=draws, target_accept=0.9, idata_kwargs={"log_likelihood": True}, random_seed=seed)
        ppc = pm.sample_posterior_predictive(trace, random_seed=seed)

    model_stats(trace, ppc, y_data, model_name)
    check_BART(ppc, x_data, y_data, x, true_regression_line, model_name)
    graph = pm.model_to_graphviz(robust_bart)
    graph.render(filename=f'model_graph_{model_name}', format='png', view=False)

'''
Model 7 - Mixture BART Regression
'''
def Mixture_BART_regression(x, true_regression_line, x_data, y_data, draws, model_name, seed):
    x_data = x_data.reshape(-1, 1)
    with pm.Model() as mixture_bart:
        mu = pmb.BART('mu', X=x_data, Y=y_data, m=10)

        # Outlier detection: Mixture model (Low prior probability of being an outlier)
        is_outlier = pm.Bernoulli('is_outlier', p=0.09, shape=y_data.shape[0])
        sigma_inlier = pm.HalfNormal('sigma_inlier', sigma=1)
        sigma_outlier = pm.HalfNormal('sigma_outlier', sigma=10)

        # Mixture likelihood
        pm.Mixture(
            'likelihood',
            w=pm.math.stack([1 - is_outlier, is_outlier]).T,
            comp_dists=[
                pm.Normal.dist(mu=mu, sigma=sigma_inlier),
                pm.Normal.dist(mu=mu, sigma=sigma_outlier)
            ],
            observed=y_data
        )

        # Sampling
        trace = pm.sample(draws=draws, target_accept=0.9, idata_kwargs={"log_likelihood": True}, random_seed=seed)
        ppc = pm.sample_posterior_predictive(trace, random_seed=seed)

    model_stats(trace, ppc, y_data, model_name)
    check_BART(ppc, x_data, y_data, x, true_regression_line, model_name)
    graph = pm.model_to_graphviz(mixture_bart)
    graph.render(filename=f'model_graph_{model_name}', format='png', view=False)

if __name__ == "__main__":
    seed = 10
    draws = 2000
    outlier_count = 10
    x, true_regression_line, x_data, y_data = generate_data(outlier_count, seed)
    frequentist_linear_regression(x, true_regression_line, x_data, y_data, 'Frequentist Regression')
    Bayesian_linear_regression(x, true_regression_line, x_data, y_data, draws, 'Standard Bayesian', seed)
    Robust_Bayesian_linear_regression(x, true_regression_line, x_data, y_data, draws, 'Robust Bayesian', seed)
    Mixture_Bayesian_linear_regression(x, true_regression_line, x_data, y_data, draws, 'Mixture Bayesian', seed)
    BART_regression(x, true_regression_line, x_data, y_data, draws, 'Standard BART', seed)
    Robust_BART_regression(x, true_regression_line, x_data, y_data, draws, 'Robust BART', seed)
    Mixture_BART_regression(x, true_regression_line, x_data, y_data, draws, 'Mixture BART', seed)