# %% [markdown]
# ## Exploratory data analysis: PCA and Mahalanobis distance analysis
#
# - **Principal Component Analysis (PCA)**: use PCA to reduce the dimensionality of the feature space. It will allow us to visualize the new data alongside the literature data within the same reduced feature space. PCA does this by transforming the original variables into a new set of uncorrelated variables (i.e., principal components), which are ordered by the amount of variance they can explain from the original data. This allows us to capture most of the information in the original data with fewer dimensions.
#
# - **Uniform Manifold Approximation and Projection (UMAP)**: To better visualise the data in 2D space and perseve the global structure, we also employ UMAP, which is a nonlinear dimensionality reduction method. It excels in preserving the structure of high-dimensional data in low-dimensional space, making it suitable for visualizing clusters or groups within data. Unlike PCA, UMAP can capture nonlinear relationships within the data, making it more capable of separating different classes or clusters. Our UMAP application involves a Mahalanobis distance metric, which respects the covariance of the data, and the implementation scales the data and calculates the inverse covariance matrix as part of its workflow. This technique will further aid in visualizing and understanding the structure of our compositional feature space.
#
# - **Mahalanobis Distance Calculation**: Following the visualization, we'll take a more in-depth look into the data by calculating the Mahalanobis distance for all the new data points from the centroid of the literature data set. The Mahalanobis distance is a measure of the distance between a point and a distribution, not between two distinct points. It's effectively a multivariate equivalent of the Euclidean distance. However, unlike Euclidean distance, the Mahalanobis distance is scale-invariant and takes into account the correlations of the data set. By calculating the Mahalanobis distance, we can quantify how much the new data deviates from the distribution of the literature data.
#

# %%
from matplotlib.colors import LogNorm
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import chi2
import scipy.stats as stats
from numba import set_num_threads
import random
from scipy.linalg import inv
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import math
import matplotlib.cm as cm
import matplotlib as mpl
import os

try:
    from IPython.display import display
except ImportError:
    def display(*objs):
        for obj in objs:
            print(obj)
# mpl.rcParams['font.family'] = 'FreeSans'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

DATASET_COL = 'dataset'
UMAP_COLS = ['UMAP Component 1', 'UMAP Component 2']
NON_FEATURE_COLS = [DATASET_COL] + UMAP_COLS

LABEL_C = 'corrosion dataset'
LABEL_H = 'hardness dataset'
LABEL_NEW_MOTI = 'new NiCrMoTiFe data'
LABEL_NEW_COV = 'new NiCrCoVFe data'


def output_path(filename):
    return os.path.join(OUTPUT_DIR, filename)

# display the current working directory
display("Current working directory: {0}".format(os.getcwd()))

data_path = os.path.abspath(os.path.join(
    SCRIPT_DIR, '..', '01_Dataset_Cleaned'))
display(os.path.isfile(os.path.join(
    data_path, 'LiteratureDataset_Hardness_YW_v3_processed.xlsx')))

# %% [markdown]
# ### Import hardness and corrosion LITERATURE datasets
#

# %%
# Declare column names for the chemical composition dataframe, specific testing conditions, selected features, and output for Hardness and Corrosion datasets.
compo_column = ['Fe', 'Cr', 'Ni', 'Mo', 'W', 'N', 'Nb', 'C', 'Si',
                'Mn', 'Cu', 'Al', 'V', 'Ta', 'Ti', 'Co', 'Mg', 'Y', 'Zr', 'Hf']
C_specific_testing_column = ['TestTemperature_C',
                             'ChlorideIonConcentration', 'pH', 'ScanRate_mVs']
specific_features_sel_column = ['delta_a', 'Tm', 'sigma_Tm',
                                'Hmix', 'sigma_Hmix', 'sigma_elec_nega', 'VEC', 'sigma_VEC']
H_output_column = ['converted HV']
C_output_column = ['AvgPittingPotential_mV']

# Load the Hardness and Corrosion datasets
df_H = pd.read_excel(
    os.path.join(data_path, 'LiteratureDataset_Hardness_YW_v4_processed.xlsx'))
df_C = pd.read_excel(
    os.path.join(data_path, 'LiteratureDataset_Corrosion_YW_v4_processed.xlsx'))

# Partition the datasets into component composition, specific features, and output data
df_H_compo, df_H_specific_features, df_H_output = df_H[compo_column], df_H[
    specific_features_sel_column], df_H[H_output_column]
(df_C_compo, df_C_specific_testing,
 df_C_specific_features, df_C_output) = df_C[compo_column], df_C[C_specific_testing_column], df_C[specific_features_sel_column], df_C[C_output_column]

df_H_compo_specific_features = pd.concat(
    [df_H_compo, df_H_specific_features], axis=1)
df_C_compo_specific_features = pd.concat(
    [df_C_compo, df_C_specific_features], axis=1)

# %% [markdown]
# ### Import NEW dataset
#

# %%
# Specify columns for NiCrCoVFe and NiCrMoTiFe composition dataframes
NiCrCoVFe_compo_column = ['Ni', 'Cr', 'Co', 'V', 'Fe']
NiCrMoTiFe_compo_column = ['Ni', 'Cr', 'Mo', 'Ti', 'Fe']

# Load NiCrCoVFe and NiCrMoTiFe datasets
df_NiCrCoVFe = pd.read_excel(
    os.path.join(data_path, 'MultiTaskModel_NiCrCoVFe_KW99_wt_pct_processed.xlsx'))
df_NiCrMoTiFe = pd.read_excel(
    os.path.join(data_path, 'MultiTaskModel_NiCrMoTiFe_KW131_wt_pct_processed.xlsx'))

# Extract composition and specific feature data from each dataset
df_NiCrCoVFe_compo, df_NiCrCoVFe_specific_features = df_NiCrCoVFe[
    NiCrCoVFe_compo_column], df_NiCrCoVFe[specific_features_sel_column]
df_NiCrMoTiFe_compo, df_NiCrMoTiFe_specific_features = df_NiCrMoTiFe[
    NiCrMoTiFe_compo_column], df_NiCrMoTiFe[specific_features_sel_column]

# Create a base dataframe for composition data with required columns
df_compo = pd.DataFrame(columns=compo_column)

# Merge base composition dataframe with each dataset's composition, filling missing values with 0
df_NiCrCoVFe_compo = pd.concat(
    [df_compo, df_NiCrCoVFe_compo], axis=0).fillna(0)
df_NiCrMoTiFe_compo = pd.concat(
    [df_compo, df_NiCrMoTiFe_compo], axis=0).fillna(0)

# Combine composition and specific feature data for each dataset
df_NiCrCoVFe_compo_specific_features = pd.concat(
    [df_NiCrCoVFe_compo, df_NiCrCoVFe_specific_features], axis=1)
df_NiCrMoTiFe_compo_specific_features = pd.concat(
    [df_NiCrMoTiFe_compo, df_NiCrMoTiFe_specific_features], axis=1)

# Display the first row of each combined dataframe for verification
display(df_NiCrCoVFe_compo_specific_features.head(1))
display(df_NiCrMoTiFe_compo_specific_features.head(1))

# %% [markdown]
# ### Datasets labelling for both datasets and concatenation
#

# %%
# Add 'dataset' column to Corrosion, Hardness, and the two new dataframes
for df_compo, df_compo_specific_features, label in zip(
    [df_C_compo, df_H_compo, df_NiCrMoTiFe_compo, df_NiCrCoVFe_compo],
    [df_C_compo_specific_features, df_H_compo_specific_features,
        df_NiCrMoTiFe_compo_specific_features, df_NiCrCoVFe_compo_specific_features],
        [LABEL_C, LABEL_H, LABEL_NEW_MOTI, LABEL_NEW_COV]):
    df_compo[DATASET_COL] = label
    df_compo_specific_features[DATASET_COL] = label

# Combine Corrosion and Hardness composition data into a single dataframe
df_compo_conc = pd.concat([df_C_compo, df_H_compo], ignore_index=True)
# df_compo_conc.to_excel('pairplot_corrosion_hardness_datasets.xlsx', index=False)
display(df_compo_conc.iloc[[0, -1]], df_compo_conc.shape)

# Add the new datasets to the combined composition dataframe
df_compo_conc_new = pd.concat(
    [df_compo_conc, df_NiCrMoTiFe_compo, df_NiCrCoVFe_compo], ignore_index=True)
display(df_compo_conc_new.iloc[[0, 712, -70, -1]], df_compo_conc_new.shape)

# Combine Corrosion and Hardness composition with specific features into a single dataframe
df_compo_specific_features_conc = pd.concat(
    [df_C_compo_specific_features, df_H_compo_specific_features], ignore_index=True)
display(df_compo_specific_features_conc.iloc[[
        0, -1]], df_compo_specific_features_conc.shape)

# Add the new datasets to the combined composition with specific features dataframe
df_compo_specific_features_conc_new = pd.concat(
    [df_compo_specific_features_conc, df_NiCrMoTiFe_compo_specific_features, df_NiCrCoVFe_compo_specific_features], ignore_index=True)
display(df_compo_specific_features_conc_new.iloc[[
        0, 712, -70, -1]], df_compo_specific_features_conc_new.shape)

# %% [markdown]
# ### Pairplots: the new datasets in relation to hardness and corrosion literatrue datasets
#
# composition space
#

# %%
sns.set_context("notebook", font_scale=2)

# Define color palette for the pairplot
palette = ["steelblue", "firebrick", "green", "darkorange"]
# Create pairplot with KDE for all data
grid_kde = sns.pairplot(df_compo_conc_new, vars=['Fe', 'Cr', 'Ni', 'Mo', 'Ti', 'Co', 'V'],
                        hue="dataset", kind="kde", corner=True, palette=palette)

# Adjust x and y limits using list comprehension
_ = [[ax.set_xlim(left=0),
      ax.set_ylim(bottom=0),
      ax.xaxis.label.set_size(25),
      ax.yaxis.label.set_size(25)]
     for ax_row in grid_kde.axes for ax in ax_row if ax is not None]

plt.savefig(output_path('pairplot_compo_section_literature+new.pdf'), bbox_inches='tight')

# Show the plots
plt.show()

# %% [markdown]
# composition + engineered feature space
#

# %%
sns.set_context("notebook", font_scale=2)

# Define color palette for the pairplot
palette = ["steelblue", "firebrick", "green", "darkorange"]

# Create pairplot with KDE for all data
grid_kde = sns.pairplot(df_compo_specific_features_conc_new, vars=specific_features_sel_column,
                        hue="dataset", kind="kde", corner=True, palette=palette)

# Adjust x and y limits using list comprehension
_ = [[ax.set_xlim(left=0),
      ax.set_ylim(bottom=0),
      ax.xaxis.label.set_size(25),
      ax.yaxis.label.set_size(25)] for ax_row in grid_kde.axes for ax in ax_row if ax is not None]

plt.savefig(output_path('pairplot_feature_literature+new.pdf'), bbox_inches='tight')

# Show the plots
plt.show()

# %% [markdown]
# ## 1. Let's try PCA (Principal Component Analysis)
#

# %%


def perform_pca(df, dim_name1='Principle Component 1', dim_name2='Principle Component 2'):
    df_pca = df.copy()

    # Scale the data
    # X_conc = StandardScaler().fit_transform(df_pca.drop(columns='dataset').values)
    X_conc = MinMaxScaler().fit_transform(df_pca.drop(columns='dataset').values)
    y_conc = df_pca['dataset'].values

    # Perform PCA
    pca = PCA()
    X_conc_r = pca.fit_transform(X_conc)
    X_conc_r = X_conc_r[:, :2]

    # Add the PCA components to the dataframe
    df_pca[dim_name1] = X_conc_r[:, 0]
    df_pca[dim_name2] = X_conc_r[:, 1]

    # Print explained variance ratio
    print(
        f"Explained variance ratio (first two components): {pca.explained_variance_ratio_[0:2]}")

    return df_pca


def plot_data(df_pca, title='PCA',
              dim_name1='Principle Component 1', dim_name2='Principle Component 2',
              x_label='Reduced Dimension 1 (UMAP)', y_label='Reduced Dimension 2 (UMAP)',
              axis_lim=[-1.3, 1.3, -1.3, 1.3], threshold=0.1, levels=5):

    # Set up the plot
    colors = ["steelblue", "firebrick", "green", "darkorange"]
    # colors = ["steelblue", "firebrick"]

    # Create a joint plot with KDE contour
    g = sns.jointplot(data=df_pca, x=dim_name1, y=dim_name2,
                      hue='dataset', palette=colors, alpha=0.4)
    g.plot_joint(sns.kdeplot, zorder=0, palette=colors,
                 levels=levels, linewidths=1, alpha=0.75, thresh=threshold)

    # Customize the plot
    g.ax_joint.grid(linewidth=0.1, alpha=.25)
    g.ax_joint.set_xlabel(x_label, fontsize=20)
    g.ax_joint.set_ylabel(y_label, fontsize=20)
    g.ax_joint.tick_params(labelsize=20)
    # g.ax_joint.set_aspect('equal', 'box')
    g.ax_joint.set_xlim(axis_lim[0], axis_lim[1])
    g.ax_joint.set_ylim(axis_lim[2], axis_lim[3])
    # g.ax_joint.set_title(title, fontsize=20, y=1.5)
    # g.ax_joint.legend(loc='upper right', fontsize=22, bbox_to_anchor=(2.5, 1))
    g.ax_joint.legend(loc='upper right', fontsize=22,
                      bbox_to_anchor=(2.5, 1), labelspacing=0.05)

    # Set the same limits for the marginal plots
    g.ax_marg_x.set_xlim(axis_lim[0], axis_lim[1])
    g.ax_marg_y.set_ylim(axis_lim[2], axis_lim[3])

    # Save and show the plot
    plt.savefig(output_path(title + '.pdf'), bbox_inches='tight')
    plt.show()


# %%
# Perform PCA on df_compo_conc_new
df_compo_conc_new_pca = perform_pca(
    df_compo_conc_new, dim_name1='Principle Component 1', dim_name2='Principle Component 2')

# Perform PCA on df_compo_specific_features_conc_new
df_compo_specific_features_conc_new_pca = perform_pca(df_compo_specific_features_conc_new,
                                                      dim_name1='Principle Component 1',
                                                      dim_name2='Principle Component 2')

# Plot data for df_compo_conc_new_pca
plot_data(df_compo_conc_new_pca, title='PCA 2D_Compositional Space',
          dim_name1='Principle Component 1', dim_name2='Principle Component 2',
          x_label='Principle Component 1', y_label='Principle Component 2',
          axis_lim=[-1, 1.3, -1, 1.05],
          #   axis_lim=[-5, 10, -5, 5],
          threshold=0.02, levels=8)

# Plot data for df_compo_specific_features_conc_new_pca
plot_data(df_compo_specific_features_conc_new_pca, title='PCA 2D_Compositional and Engineered Feature Space',
          dim_name1='Principle Component 1', dim_name2='Principle Component 2',
          x_label='Principle Component 1', y_label='Principle Component 2',
          axis_lim=[-1, 1.3, -1, 1.05],
          #   axis_lim=[-5, 10, -5, 5],
          threshold=0.02, levels=8)

# %% [markdown]
# ## 2. Let's try UMAP (Uniform Manifold Approximation and Projection)
#

# %%

# np.random.seed(0)
# random.seed(0)


def perform_umap(df, n_neighbors=30, min_dist=0.6, n_components=2, metric='mahalanobis',
                 dim_name1='UMAP Component 1', dim_name2='UMAP Component 2', random_state=42):
    df_umap = df.copy()

    np.random.seed(random_state)
    random.seed(random_state)
    set_num_threads(1)

    # # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # X = df_umap.drop(columns='dataset').values

    # # Scale the data
    # X_conc = scaler.fit_transform(X)
    # y_conc = df_umap['dataset'].values

    # Scale the data
    X_conc = MinMaxScaler().fit_transform(df_umap.drop(columns='dataset').values)
    y_conc = df_umap['dataset'].values

    # Calculate inverse covariance matrix if using 'mahalanobis' metric
    if metric == 'mahalanobis':
        cov = EmpiricalCovariance().fit(X_conc)
        inv_cov_matrix = inv(cov.covariance_)
        metric_kwds = {'VI': inv_cov_matrix}
    else:
        metric_kwds = {}

    # Perform UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        metric=metric,
                        metric_kwds=metric_kwds,
                        random_state=random_state)

    X_conc_r = reducer.fit_transform(X_conc)

    # Add the UMAP components to the dataframe
    df_umap[dim_name1] = X_conc_r[:, 0]
    df_umap[dim_name2] = X_conc_r[:, 1]

    return df_umap

# %%
# # Perform UMAP on df_compo_conc_new (composition only)
# df_compo_conc_new_umap = perform_umap(df_compo_conc_new,
#                                       n_neighbors=20, min_dist=0.5, n_components=2, metric='mahalanobis',
#                                       dim_name1='UMAP Component 1', dim_name2='UMAP Component 2',
#                                       random_state=42)

# # Plot data for df_compo_conc_new_umap
# plot_data(df_compo_conc_new_umap, title='UMAP 2D_Compositional Space',
#           dim_name1='UMAP Component 1', dim_name2='UMAP Component 2',
#           axis_lim=[-100, 100, -50, 50], threshold=0.001, levels=30)


# %%
np.random.seed(42)

# display(df_compo_specific_features_conc_new.head(1))

# Perform UMAP on df_compo_specific_features_conc_new
df_compo_specific_features_conc_new_umap = perform_umap(df_compo_specific_features_conc_new,
                                                        n_neighbors=20, min_dist=0.5, n_components=2, metric='mahalanobis',
                                                        dim_name1='UMAP Component 1', dim_name2='UMAP Component 2',
                                                        random_state=42)

df_HC_umap = df_compo_specific_features_conc_new_umap[
    (df_compo_specific_features_conc_new_umap[DATASET_COL] == LABEL_C) |
    (df_compo_specific_features_conc_new_umap[DATASET_COL] == LABEL_H)]

display(df_HC_umap.head(1))

# Plot data for df_compo_specific_features_conc_new_umap
plot_data(df_HC_umap, title='UMAP 2D_Compositional and Engineered Feature Space',
          dim_name1='UMAP Component 1', dim_name2='UMAP Component 2',
          x_label='Reduced dimension 1 (UMAP)', y_label='Reduced dimension 2 (UMAP)',
          axis_lim=[-30, 40, -20, 50], threshold=0.001, levels=20)

# %% [markdown]
# ## 3. Let's try to use the local Mahalanobis distance
#
# ### Multivariate analysis to examine the local (Mahalanobis) Distance from new data to existing datasets
#

# %%

# %% [markdown]
# here decide to use composition + engineered feature space for piror Mahalanobis distance analysis
#

# %%
df_C_umap = df_compo_specific_features_conc_new_umap[
    df_compo_specific_features_conc_new_umap[DATASET_COL] == LABEL_C]
df_H_umap = df_compo_specific_features_conc_new_umap[
    df_compo_specific_features_conc_new_umap[DATASET_COL] == LABEL_H]
df_new_FeCrNiMoTi_umap = df_compo_specific_features_conc_new_umap[
    df_compo_specific_features_conc_new_umap[DATASET_COL] == LABEL_NEW_MOTI]
df_new_FeCrNiCoV_umap = df_compo_specific_features_conc_new_umap[
    df_compo_specific_features_conc_new_umap[DATASET_COL] == LABEL_NEW_COV]


# prepare the df for calculating Mahalanobis distance (cannot have string in df)
display(df_C_umap.iloc[[0, -1]], df_C_umap.shape,
        df_H_umap.iloc[[0, -1]], df_H_umap.shape,
        df_new_FeCrNiMoTi_umap.iloc[[0, -1]], df_new_FeCrNiMoTi_umap.shape,
        df_new_FeCrNiCoV_umap.iloc[[0, -1]], df_new_FeCrNiCoV_umap.shape)

# %% [markdown]
# fit the literature-only reference models and score each dataset against them
#

# %%


def get_numeric_features(df):
    return df.drop(columns=[col for col in NON_FEATURE_COLS if col in df.columns]).astype(float)


def fit_reference_model(df_ref):
    X_ref = get_numeric_features(df_ref).copy()

    keep_cols = X_ref.var(axis=0) > 0
    dropped_cols = X_ref.columns[~keep_cols].tolist()
    print("Columns dropped from reference model because variance is zero:", dropped_cols)

    X_ref = X_ref.loc[:, keep_cols]
    scaler = StandardScaler().fit(X_ref)
    X_ref_scaled = scaler.transform(X_ref)
    cov_model = LedoitWolf().fit(X_ref_scaled)

    return {
        'feature_cols': X_ref.columns.tolist(),
        'scaler': scaler,
        'mean_': cov_model.location_,
        'precision_': cov_model.precision_,
        'df': X_ref.shape[1],
    }


def score_against_reference(df_target, model):
    X_target = get_numeric_features(df_target)[model['feature_cols']]
    X_target_scaled = model['scaler'].transform(X_target)

    delta = X_target_scaled - model['mean_']
    md2 = np.einsum('ij,jk,ik->i', delta, model['precision_'], delta)

    scored = df_target.copy()
    scored['Mahalanobis_sq'] = md2
    scored['p value'] = chi2.sf(md2, df=model['df'])
    return scored


model_C = fit_reference_model(df_C_umap)
model_H = fit_reference_model(df_H_umap)

df_C_ref_scored = score_against_reference(df_C_umap, model_C)
df_H_ref_scored = score_against_reference(df_H_umap, model_H)
df_new_MoTi_vs_C = score_against_reference(df_new_FeCrNiMoTi_umap, model_C)
df_new_CoV_vs_C = score_against_reference(df_new_FeCrNiCoV_umap, model_C)
df_new_MoTi_vs_H = score_against_reference(df_new_FeCrNiMoTi_umap, model_H)
df_new_CoV_vs_H = score_against_reference(df_new_FeCrNiCoV_umap, model_H)

df_C_new_FeCrNiMoTi_Mahl_label = pd.concat(
    [df_C_ref_scored, df_new_MoTi_vs_C], ignore_index=True)
df_C_new_FeCrNiCoV_Mahl_label = pd.concat(
    [df_C_ref_scored, df_new_CoV_vs_C], ignore_index=True)
df_H_new_FeCrNiMoTi_Mahl_label = pd.concat(
    [df_H_ref_scored, df_new_MoTi_vs_H], ignore_index=True)
df_H_new_FeCrNiCoV_Mahl_label = pd.concat(
    [df_H_ref_scored, df_new_CoV_vs_H], ignore_index=True)

display(df_C_new_FeCrNiMoTi_Mahl_label.iloc[[0, -1]])
display(df_C_new_FeCrNiCoV_Mahl_label.head(1))
display(df_H_new_FeCrNiMoTi_Mahl_label.head(1))
display(df_H_new_FeCrNiCoV_Mahl_label.head(1))

# %%
print(model_C['df'])


# %%
# Define dataframes in a list
dfs = [df_C_new_FeCrNiMoTi_Mahl_label, df_C_new_FeCrNiCoV_Mahl_label,
       df_H_new_FeCrNiMoTi_Mahl_label, df_H_new_FeCrNiCoV_Mahl_label]
titles = ['df_C_new_FeCrNiMoTi_Mahl', 'df_C_new_FeCrNiCoV_Mahl',
          'df_H_new_FeCrNiMoTi_Mahl', 'df_H_new_FeCrNiCoV_Mahl']

# Set up subplots
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Mahalanobis distance', fontsize=20)

# Iterate through dataframes
for i, df in enumerate(dfs):
    # Plot histogram
    axs[0, i].hist(df['Mahalanobis_sq'], bins=100,
                   edgecolor='black', alpha=0.5, density=True)
    axs[0, i].set_xlabel('Squared Mahalanobis distance')
    axs[0, i].set_ylabel('Frequency')
    axs[0, i].set_xlim(0, 200)
    axs[0, i].set_title(titles[i])

    # Fit a chi-squared distribution to the data
    df_param, loc, scale = stats.chi2.fit(df['Mahalanobis_sq'])
    # Plot the fitted distribution over the histogram
    x_chi2 = np.linspace(0, np.amax(df['Mahalanobis_sq']), 500)
    pdf_chi2 = stats.chi2.pdf(x_chi2, df=df_param, loc=loc, scale=scale)
    axs[0, i].plot(x_chi2, pdf_chi2, 'maroon')
    # Add the fitted values to the plot
    axs[0, i].text(0.7, 0.5, f"df = {df_param:.2f}\nloc = {loc:.2f}\nscale = {scale:.2f}",
                   transform=axs[0, i].transAxes, ha='left', va='center', fontsize=12)

    # Plot CDF
    counts, bins, patches = axs[1, i].hist(
        df['Mahalanobis_sq'], bins=100, edgecolor='black', alpha=0.5, cumulative=True, density=True)
    axs[1, i].plot(bins[:-1], counts, 'maroon', lw=2)
    axs[1, i].set_xlim(0, 200)
    axs[1, i].set_xlabel('Squared Mahalanobis distance')
    axs[1, i].set_ylabel('Cumulative Frequency')

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlapping
plt.show()

# %% [markdown]
# Use chi2 statistics to get a more presentable number: p-value based on the retained
# reference-space degrees of freedom
#
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4381501/
#

# %%
display(df_C_new_FeCrNiMoTi_Mahl_label.iloc[[0, -1]])

# plot the histogram of p-values for all the datasets
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
fig.suptitle('p-value', fontsize=20)
axs[0, 0].hist(df_C_new_FeCrNiMoTi_Mahl_label['p value'], bins=100,
               edgecolor='black', alpha=0.5, density=True)
axs[0, 0].set_xlabel('p-value')
axs[0, 0].set_ylabel('Frequency')

axs[0, 1].hist(df_C_new_FeCrNiCoV_Mahl_label['p value'], bins=100,
               edgecolor='black', alpha=0.5, density=True)
axs[0, 1].set_xlabel('p-value')
axs[0, 1].set_ylabel('Frequency')

axs[1, 0].hist(df_H_new_FeCrNiMoTi_Mahl_label['p value'], bins=100,
               edgecolor='black', alpha=0.5, density=True)
axs[1, 0].set_xlabel('p-value')
axs[1, 0].set_ylabel('Frequency')

axs[1, 1].hist(df_H_new_FeCrNiCoV_Mahl_label['p value'], bins=100,
               edgecolor='black', alpha=0.5, density=True)
axs[1, 1].set_xlabel('p-value')
axs[1, 1].set_ylabel('Frequency')

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# %% [markdown]
# If we believe the p value from chi2 statistics can be a measure of "novelty" (smaller ones are more likely outliers), we plot it back to PCA 2D project and also PVD representation
#

# %% [markdown]
# map the chi2 pvalues to the PCA 2D projection: it seems the variation of p value is NOT monotonic on this 2D projection
#
# I mainly highlighted the "new" dataset (you can still see the translucent "training" data points)
#

# %%


def create_scatter(df, dataset_values, ax, cmap="RdBu", titlename="name?"):
    # Create separate dataframes for each dataset
    df_literature = df[df[DATASET_COL] == dataset_values[0]]
    df_new = df[df[DATASET_COL] == dataset_values[1]]

    # Create the scatter plots
    scatters = []
    for df, alpha in zip([df_literature, df_new], [0.2, 0.9]):
        scatter = ax.scatter(df[UMAP_COLS[0]], df[UMAP_COLS[1]], c=df["p value"], cmap=cmap, edgecolor="grey",
                             s=500, marker='.', alpha=alpha, vmin=0, vmax=0.1)
        scatters.append(scatter)

        ax.set_xlabel(UMAP_COLS[0], fontsize=20)
        ax.set_ylabel(UMAP_COLS[1], fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # set the subplot title to the dataset name
        ax.set_title(titlename, fontsize=20, y=1.05)
        # make equal aspect ratio
        ax.set_aspect('equal', 'box')

    return scatters


# create the subplots of 2x2
fig, ax = plt.subplots(2, 2, figsize=(18, 15), dpi=150)

datasets = [(df_C_new_FeCrNiMoTi_Mahl_label, [LABEL_C, LABEL_NEW_MOTI], "RdBu",
             '"distance" from new FeCrNiMoTi data\nto corrosion dataset'),
            (df_C_new_FeCrNiCoV_Mahl_label, [LABEL_C, LABEL_NEW_COV], "RdBu",
             '"distance" from new FeCrNiCoV data\nto corrosion dataset'),
            (df_H_new_FeCrNiMoTi_Mahl_label, [LABEL_H, LABEL_NEW_MOTI], "RdYlBu",
             '"distance" from new FeCrNiMoTi data\nto hardness dataset'),
            (df_H_new_FeCrNiCoV_Mahl_label, [LABEL_H, LABEL_NEW_COV], "RdYlBu",
             '"distance" from new FeCrNiCoV data\nto hardness dataset')]

# Get global min and max for UMAP components
xmin, xmax = min(df[UMAP_COLS[0]].min() for df, _, _, _ in datasets), max(
    df[UMAP_COLS[0]].max() for df, _, _, _ in datasets)
ymin, ymax = min(df[UMAP_COLS[1]].min() for df, _, _, _ in datasets), max(
    df[UMAP_COLS[1]].max() for df, _, _, _ in datasets)


for i, (df, dataset_values, cmap, title) in enumerate(datasets):
    axi = ax[i//2, i % 2]  # Get the current axis
    scatters = create_scatter(df, dataset_values, axi,
                              cmap=cmap, titlename=title)

    # Set the limits
    axi.set_xlim(xmin, xmax)
    axi.set_ylim(ymin, ymax)

    # Create a divider for the existing axes instance
    divider = make_axes_locatable(axi)

    # Append axes for colorbar to the right of axi, with 5% width of axi
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(scatters[-1], cax=cax)
    cbar.set_label("p-value", size=20, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig(output_path('UMAP 2D_Mahalanobis.png'))
plt.show()

# %% [markdown]
# Now I will plot the p value on the representation of PVD wafer
#

# %%


# %%
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import pandas as pd

# PVD_x_y = pd.read_excel(data_path + 'PVD_x_y.xlsx')


# def create_PVD_scatter(df, PVD_x_y, dataset_values, ax, cmap="RdBu", title=""):
#     df_new = df[df["dataset"] == dataset_values[1]]
#     scatter = ax.scatter(PVD_x_y["x"], PVD_x_y["y"], c=df_new["p value"], cmap=cmap, edgecolor="grey",
#                          s=2000, marker='.', alpha=1, vmin=0, vmax=0.05)

#     for i, txt in enumerate(PVD_x_y.index+1):
#         ax.annotate(txt, (PVD_x_y["x"].iloc[i]-3, PVD_x_y["y"].iloc[i]-1),
#                     color="white", alpha=0.7, fontsize=14)

#     ax.set_xlabel("position x", fontsize=16)
#     ax.set_ylabel("position y", fontsize=16)
#     ax.tick_params(axis='both', labelsize=14)
#     ax.set_title(title, fontsize=18, y=1.05)
#     ax.set_aspect(1)
#     ax.set_xticks(range(0, 101, 10))
#     ax.set_xticklabels(ax.get_xticks(), rotation=45)
#     ax.set_yticks(range(0, 101, 10))

#     df_new.to_excel(title + ".xlsx")

#     return scatter


# fig, ax = plt.subplots(1, 4, figsize=(20, 4.7), dpi=150)

# for i, (df, dataset_values, cmap, title) in enumerate(datasets):
#     axi = ax[i]
#     scatter = create_PVD_scatter(
#         df, PVD_x_y, dataset_values, axi, cmap=cmap, title=title)

#     # Only add colorbar to the last subplot
#     if i == len(datasets) - 1 or i == len(datasets)/2 - 1:
#         divider = make_axes_locatable(axi)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         cbar = fig.colorbar(scatter, cax=cax)
#         cbar.set_label('p-values for Mahalanobis distances',
#                        size=16)
#         cbar.ax.tick_params(labelsize=16)

# plt.tight_layout()
# plt.savefig('PVD 2D_Mahalanobis.pdf', bbox_inches='tight')
# plt.show()


PVD_x_y = pd.read_excel(os.path.join(data_path, 'PVD_x_y.xlsx'))


def make_safe_filename(text):
    text = str(text).replace("\n", " ")
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def create_PVD_scatter(df, PVD_x_y, dataset_values, ax, cmap="RdBu", title="", export_name=None):
    df_new = df[df[DATASET_COL] == dataset_values[1]].copy()

    scatter = ax.scatter(
        PVD_x_y["x"],
        PVD_x_y["y"],
        c=df_new["p value"],
        cmap=cmap,
        edgecolor="grey",
        s=2000,
        marker='.',
        alpha=1,
        vmin=0,
        vmax=1
    )

    for i, txt in enumerate(PVD_x_y.index + 1):
        ax.annotate(
            txt,
            (PVD_x_y["x"].iloc[i] - 3, PVD_x_y["y"].iloc[i] - 1),
            color="white",
            alpha=0.7,
            fontsize=14
        )

    ax.set_xlabel("position x", fontsize=16)
    ax.set_ylabel("position y", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(title, fontsize=18, y=1.05)
    ax.set_aspect(1)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.set_yticks(range(0, 101, 10))

    if export_name is None:
        export_name = make_safe_filename(title)
    df_new.to_excel(output_path(f"{export_name}.xlsx"), index=False)

    return scatter


fig, ax = plt.subplots(1, 4, figsize=(20, 4.7), dpi=150)

for i, (df, dataset_values, cmap, title) in enumerate(datasets):
    axi = ax[i]
    export_name = f"{i+1:02d}_{make_safe_filename(title)}"

    scatter = create_PVD_scatter(
        df,
        PVD_x_y,
        dataset_values,
        axi,
        cmap=cmap,
        title=title,
        export_name=export_name
    )

    if i == len(datasets) - 1 or i == len(datasets) // 2 - 1:
        divider = make_axes_locatable(axi)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label('p-values for Mahalanobis distances', size=16)
        cbar.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig(output_path('PVD_2D_Mahalanobis.pdf'), bbox_inches='tight')
plt.show()



# %%

PVD_x_y = pd.read_excel(os.path.join(data_path, 'PVD_x_y.xlsx'))


def make_safe_filename(text):
    text = str(text).replace("\n", " ")
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def create_PVD_scatter(df, PVD_x_y, dataset_values, ax, cmap="RdBu", title="", export_name=None):
    df_new = df[df[DATASET_COL] == dataset_values[1]].copy()

    scatter = ax.scatter(
        PVD_x_y["x"],
        PVD_x_y["y"],
        c=df_new["p value"],
        cmap=cmap,
        edgecolor="grey",
        s=2000,
        marker='.',
        alpha=1,
        norm=LogNorm(vmin=1e-7, vmax=5e-2)
    )

    for i, txt in enumerate(PVD_x_y.index + 1):
        ax.annotate(
            txt,
            (PVD_x_y["x"].iloc[i] - 3, PVD_x_y["y"].iloc[i] - 1),
            color="white",
            alpha=0.7,
            fontsize=14
        )

    ax.set_xlabel("position x", fontsize=16)
    ax.set_ylabel("position y", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(title, fontsize=18, y=1.05)
    ax.set_aspect(1)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.set_yticks(range(0, 101, 10))

    if export_name is None:
        export_name = make_safe_filename(title)
    df_new.to_excel(output_path(f"{export_name}.xlsx"), index=False)

    return scatter


# Leave room on the right of subplot 2 and subplot 4 for separate colorbar axes
fig, ax = plt.subplots(1, 4, figsize=(25, 4.7), dpi=150)
plt.subplots_adjust(left=0.06, right=0.94, bottom=0.14, top=0.85, wspace=0.50)

scatters = []

for i, (df, dataset_values, cmap, title) in enumerate(datasets):
    export_name = f"{i+1:02d}_{make_safe_filename(title)}"
    scatter = create_PVD_scatter(
        df, PVD_x_y, dataset_values, ax[i], cmap=cmap, title=title, export_name=export_name
    )
    scatters.append(scatter)

# Add colorbars in dedicated figure axes so subplot sizes stay identical
# [left, bottom, width, height] in figure coordinates
cax1 = fig.add_axes([0.468, 0.17, 0.010, 0.62])   # after subplot 2
cax2 = fig.add_axes([0.959, 0.17, 0.010, 0.62])   # after subplot 4

for cax, scatter in zip(
    [cax1, cax2],
    [scatters[len(datasets)//2 - 1], scatters[len(datasets) - 1]]
):
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('p-values for Mahalanobis distances', size=16)

    # major ticks
    major_ticks = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    cbar.set_ticks(major_ticks)
    cbar.set_ticklabels([
        r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$',
        r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'
    ])

    # minor ticks aligned with vmin=1e-7
    minor_ticks = []
    for exp in range(-7, -1):
        base = 10.0 ** exp
        for m in range(2, 10):
            val = m * base
            if 1e-7 <= val <= 5e-2:
                minor_ticks.append(val)

    cbar.ax.yaxis.set_ticks(minor_ticks, minor=True)

    cbar.ax.tick_params(axis='y', which='major',
                        labelsize=14, length=8, width=1.0)
    cbar.ax.tick_params(axis='y', which='minor', length=5, width=1.0)

plt.savefig(output_path('PVD_2D_Mahalanobis.pdf'), bbox_inches='tight')
plt.show()
# %%
