import numpy as np
import pandas as pd

from mca import MCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM

import seaborn as sns
import matplotlib.pyplot as plt


### Model utils ###

def plot_cluster_metrics(df: pd.DataFrame, variables: list, n_rows: int, n_cols: int, target: str):
    fig = plt.figure(figsize=(13, 5))
    for i, var in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        sns.lineplot(data=df, x=target, y=var, err_style="bars", ci=68, ax=ax)
        ax.set_title(i)
    fig.tight_layout()
    plt.show()


def plot_scatter(x, y, x_label, y_label, title):
    plt.scatter(x, y, alpha=0.3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


def plot_histograms(df, variables, n_rows, n_cols):
    fig = plt.figure(figsize=(15, 45))
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        sns.histplot(data=df, x=var_name, alpha=0.3, bins=40, ax=ax)
        ax.set_title(var_name + " Distribution")
    fig.tight_layout()
    plt.show()


def plot_tsne_visualization(data, n_clusters):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="cluster",
        palette=sns.color_palette("hls", n_clusters),
        data=data,
        legend="full",
        alpha=0.3
    )

def repeated_measures_anova(data, measurement, subject_col='key', within_col='year', exclude_cols=None):
    """
    Conducts Repeated Measures ANOVA for each measurement column in the data.

    Parameters:
    - data (pd.DataFrame): The dataframe containing the measurements.
    - subject_col (str): The column name for the subject/individual identifier.
    - within_col (str): The column name for the repeated measure (e.g., time, year).
    - exclude_cols (list): List of column names to exclude from the analysis.

    Returns:
    - dict: A dictionary where keys are measurement column names and values are the ANOVA results for that column.
    """

    subset_data = data[[subject_col, within_col, measurement]]
        
    # Conduct Repeated Measures ANOVA
    anova_results = AnovaRM(data=subset_data, depvar=measurement, subject=subject_col, within=[within_col]).fit()

    return anova_results

def plot_dimensionality_reduction_stats(first_components: list, eigenvalues: pd.DataFrame, x_title: str, y_title: str,
                                        title: str):
    """
    Visualize distribution of data with the first two components and the explained variance per additional eigenvector
    estimated by the SVD method.
    :param first_components: list
    :param eigenvalues: pd.DataFrame
    :param x_title: str
    :param y_title: str
    :param title: str
    :return:
    """
    # Visualize data distribution with first two components
    fig = plt.figure(figsize=(13, 5))
    fig.add_subplot(1, 2, 1)
    plt.scatter(first_components[0], first_components[1], alpha=0.3)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)

    # Visualize explained variance per eigenvector
    fig.add_subplot(1, 2, 2)
    sns.lineplot(data=eigenvalues, x='eigenvectors', y='perc_expl_var')
    plt.xlabel('Eigenvectors')
    plt.ylabel('% Explained')
    plt.title('% Explained Variance')
    fig.tight_layout()
    plt.show()


def mca_factor_visualizations(mca: MCA, perc_inertia: float):
    """
    Project the first two factors estimated from the categorical features transformed.
    :param mca: MCA object
    :param perc_inertia: float
    :return: int
    """
    # Calculate first two factors for visualization
    res_mca_2 = pd.DataFrame(mca.fs_r(N=2))

    # Calculate the number of eigenvectors that explain the desired % of correspondence
    n_eigenvectors = len(mca.L)
    expl_var = mca.expl_var(greenacre=False)[:n_eigenvectors]
    total_expl_var = np.sum(expl_var)
    perc_expl_var = 100 - expl_var * 100 / total_expl_var
    mca_eigenvalues = pd.DataFrame({'eigenvectors': range(1, n_eigenvectors + 1),
                                    'perc_expl_var': perc_expl_var})
    n_eigenvectors = np.max(mca_eigenvalues[mca_eigenvalues['perc_expl_var'] < perc_inertia]['eigenvectors'])

    # Visualize MCA relevant plots
    plot_dimensionality_reduction_stats(res_mca_2, mca_eigenvalues, 'Factor #1', 'Factor #2', 'MCA Decomposition')

    return n_eigenvectors


def pca_component_visualizations(data: pd.DataFrame, perc_inertia: float):
    """
    Project the first two components estimated from the continuous features transformed.
    :param data: pd.DataFrame
    :param perc_inertia: float
    :return: int
    """
    # Calculate first two components for visualization
    pca = PCA(n_components=2)
    res_pca = pca.fit_transform(data)
    res_pca_2 = [res_pca[:, 0], res_pca[:, 1]]

    # calculate optimal number of components to explain the desired percent of variance
    pca = PCA()
    pca.fit(data)

    n_eigenvectors = len(pca.explained_variance_ratio_)
    expl_var = pca.explained_variance_ratio_
    total_expl_var = np.sum(expl_var)
    perc_expl_var = 100 - expl_var * 100 / total_expl_var
    pca_eigenvalues = pd.DataFrame({'eigenvectors': range(1, n_eigenvectors + 1),
                                    'perc_expl_var': perc_expl_var})
    n_eigenvectors = np.max(pca_eigenvalues[pca_eigenvalues['perc_expl_var'] < perc_inertia]['eigenvectors'])

    # Visualize PCA relevant plots
    plot_dimensionality_reduction_stats(res_pca_2, pca_eigenvalues, 'Principal Component #1',
                                        'Principal Component #2', 'PCA results')

    return n_eigenvectors

def screen_gmm_optimal_n_clusters(X: pd.DataFrame, n_clusters: int, n_iterations: int):
    """
    Explore silhouette_score and BIC score of for n_clusters from GMM.
    :param X: pd.DataFrame
    :param n_clusters: int
    :param n_iterations: int
    :return: pd.DataFrame
    """
    data = []
    for c in range(2, n_clusters):
        for i in range(n_iterations):
            d = {}
            gm = GaussianMixture(n_components=c, random_state=0).fit(X)
            labels = gm.predict(X)
            s_score = silhouette_score(X, labels, metric='euclidean')
            bic = gm.bic(X)
            d['n_clusters'] = c
            d['iteration'] = i
            d['silhouette_score'] = s_score
            d['bic'] = bic
            data.append(d)

    return pd.DataFrame.from_records(data)


def screen_kmeans_optimal_n_clusters(X: pd.DataFrame, n_clusters: int, n_iterations: int):
    """
    Explore silhouette_score and BIC score of for n_clusters from K-mean.
    :param X: pd.DataFrame
    :param n_clusters: int
    :param n_iterations: int
    :return: pd.DataFrame
    """
    data = []
    for c in n_clusters:
        for i in range(n_iterations):
            d = {}
            km = KMeans(n_clusters=c, random_state=0).fit(X)
            labels = km.predict(X)
            s_score = silhouette_score(X, labels, metric='euclidean')
            d['n_clusters'] = c
            d['iteration'] = i
            d['silhouette_score'] = s_score
            d['inertia'] = km.inertia_
            data.append(d)

    return pd.DataFrame.from_records(data)
