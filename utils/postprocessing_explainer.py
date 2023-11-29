import seaborn as sns
import shap
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tabulate import tabulate
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow import keras
import warnings
from utils.path_explainer_tf import PathExplainerTF
from tqdm import tqdm


def compute_X_Z_W_shap(model,
                       X2_base_data, Z2_base_data, W2_base_data,
                       X2_shap_data, Z2_shap_data, W2_shap_data,
                       scaler_compo, scaler_testing, scaler_specific):

    if Z2_base_data.size == 0 and W2_base_data.size != 0:
        # Normalize the base data using the respective scalers
        X2_base_normalised = scaler_compo.transform(
            X2_base_data).astype(np.float32)
        W2_base_normalised = scaler_specific.transform(
            W2_base_data).astype(np.float32)

        X2_shap_normalised = scaler_compo.transform(
            X2_shap_data).astype(np.float32)
        W2_shap_normalised = scaler_specific.transform(
            W2_shap_data).astype(np.float32)

        # Concatenate normalized base data for input to the model
        X2_W2_base_normalised = np.concatenate(
            [X2_base_normalised, W2_base_normalised], axis=1)

        # Concatenate normalized SHAP data for calculation of SHAP values
        X2_W2_shap_normalised = np.concatenate(
            [X2_shap_normalised, W2_shap_normalised], axis=1)

        # Initialize SHAP GradientExplainer with the model and base data
        explainer = shap.GradientExplainer(model, [X2_W2_base_normalised])

        # Calculate SHAP values for the SHAP data
        shap_values_all = explainer.shap_values(X2_W2_shap_normalised)

        # Return the SHAP values for the first output of the model
        return shap_values_all[0]

    elif Z2_base_data.size == 0 and W2_base_data.size == 0:
        # Normalize the base data using the respective scalers
        X2_base_normalised = scaler_compo.transform(
            X2_base_data).astype(np.float32)

        # Normalize the SHAP data using the respective scalers
        X2_shap_normalised = scaler_compo.transform(
            X2_shap_data).astype(np.float32)

        # Initialize SHAP GradientExplainer with the model and base data
        explainer = shap.GradientExplainer(model, [X2_base_normalised])

        # Calculate SHAP values for the SHAP data
        shap_values_all = explainer.shap_values(X2_shap_normalised)

        # Return the SHAP values for the first output of the model
        return shap_values_all[0]

    elif Z2_base_data.size != 0 and W2_base_data.size != 0:

        X2_base_normalised = scaler_compo.transform(
            X2_base_data).astype(np.float32)
        Z2_base_normalised = scaler_testing.transform(
            Z2_base_data).astype(np.float32)
        W2_base_normalised = scaler_specific.transform(
            W2_base_data).astype(np.float32)

        X2_shap_normalised = scaler_compo.transform(
            X2_shap_data).astype(np.float32)
        Z2_shap_normalised = scaler_testing.transform(
            Z2_shap_data).astype(np.float32)
        W2_shap_normalised = scaler_specific.transform(
            W2_shap_data).astype(np.float32)

        # Concatenate X2 and W2 data for base and SHAP-normalised data
        X2_W2_base_normalised = np.concatenate(
            [X2_base_normalised, W2_base_normalised], axis=1)
        X2_W2_shap_normalised = np.concatenate(
            [X2_shap_normalised, W2_shap_normalised], axis=1)

        # Initialize SHAP GradientExplainer with model and background data
        explainer = shap.GradientExplainer(
            model, [X2_W2_base_normalised, Z2_base_normalised])

        # Compute SHAP values for the normalised data
        shap_values_all = explainer.shap_values(
            [X2_W2_shap_normalised, Z2_shap_normalised])

        # Combine SHAP values for X2_W2 and Z2 and return
        return np.concatenate([shap_values_all[0][0], shap_values_all[0][1]], axis=1)

    elif Z2_base_data.size != 0 and W2_base_data.size == 0:

        X2_base_normalised = scaler_compo.transform(
            X2_base_data).astype(np.float32)
        Z2_base_normalised = scaler_testing.transform(
            Z2_base_data).astype(np.float32)

        X2_shap_normalised = scaler_compo.transform(
            X2_shap_data).astype(np.float32)
        Z2_shap_normalised = scaler_testing.transform(
            Z2_shap_data).astype(np.float32)

        # Initialize SHAP GradientExplainer with model and background data
        explainer = shap.GradientExplainer(
            model, [X2_base_normalised, Z2_base_normalised])

        # Compute SHAP values for the normalised data
        shap_values_all = explainer.shap_values(
            [X2_shap_normalised, Z2_shap_normalised])

        # Combine SHAP values for X2_W2 and Z2 and return
        return np.concatenate([shap_values_all[0][0], shap_values_all[0][1]], axis=1)


# def compute_X_Z_W_shap(model, X2_base_data, Z2_base_data, W2_base_data,
#                        X2_shap_data, Z2_shap_data, W2_shap_data,
#                        scaler_compo, scaler_testing, scaler_specific):

#     def normalize_data(*datasets, scalers):
#         return [scaler.transform(data).astype(np.float32) for data, scaler in zip(datasets, scalers)]

#     def concatenate_data(*datasets):
#         return np.concatenate(datasets, axis=1)

#     def compute_shap_values(normalized_data, explainer):
#         shap_values_all = explainer.shap_values(normalized_data)
#         return shap_values_all[0]

#     normalized_base_data = normalize_data(X2_base_data, Z2_base_data, W2_base_data,
#                                           scalers=[scaler_compo, scaler_testing, scaler_specific])
#     normalized_shap_data = normalize_data(X2_shap_data, Z2_shap_data, W2_shap_data,
#                                           scalers=[scaler_compo, scaler_testing, scaler_specific])

#     if Z2_base_data.size == 0 and W2_base_data.size != 0:
#         X2_W2_base_normalized = concatenate_data(normalized_base_data[0], normalized_base_data[2])
#         X2_W2_shap_normalized = concatenate_data(normalized_shap_data[0], normalized_shap_data[2])

#         explainer = shap.GradientExplainer(model, [X2_W2_base_normalized])
#         return compute_shap_values(X2_W2_shap_normalized, explainer)

#     elif Z2_base_data.size == 0 and W2_base_data.size == 0:
#         explainer = shap.GradientExplainer(model, [normalized_base_data[0]])
#         return compute_shap_values(normalized_shap_data[0], explainer)

#     else:
#         # Handle multi-input scenarios
#         X2_Z2_W2_base_normalized = concatenate_data(*normalized_base_data)
#         X2_Z2_W2_shap_normalized = concatenate_data(*normalized_shap_data)

#         explainer = shap.GradientExplainer(model, [X2_Z2_W2_base_normalized])
#         return compute_shap_values(X2_Z2_W2_shap_normalized, explainer)


def plot_shap(shap_values, i_sample,
              X1_shap_data, Y1_shap_data, V1_shap_data,
              compo_column, specific_testing_column, specific_features_sel_column):

    print(
        f"Sum of Shapley values for sample {i_sample}: {shap_values[i_sample, :].sum()}")

    # Determine columns based on input data
    if Y1_shap_data.size == 0 and V1_shap_data.size == 0:
        columns = compo_column
    if Y1_shap_data.size != 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column + specific_testing_column
    if Y1_shap_data.size == 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column
    if Y1_shap_data.size != 0 and V1_shap_data.size == 0:
        columns = compo_column + specific_testing_column

    fig, axs = plt.subplots(1, 1, figsize=(
        4, 4), sharex=True, constrained_layout=True)
    axs.barh(columns, shap_values[i_sample, :], color='steelblue')
    axs.set_title("SHAP")
    axs.invert_yaxis()
    axs.set_box_aspect(1)

    plt.show()


# def compute_shap_attributions_interactions(model,
#                                            X2_base_data, Z2_base_data, W2_base_data,
#                                            X2_shap_data, Z2_shap_data, W2_shap_data,
#                                            scaler_compo, scaler_testing, scaler_specific):

#     if Z2_base_data.size == 0 and W2_base_data.size != 0:

#         X2_base_normalised = scaler_compo.transform(
#             X2_base_data).astype(np.float32)
#         W2_base_normalised = scaler_specific.transform(
#             W2_base_data).astype(np.float32)

#         X2_shap_normalised = scaler_compo.transform(
#             X2_shap_data).astype(np.float32)
#         W2_shap_normalised = scaler_specific.transform(
#             W2_shap_data).astype(np.float32)

#         # Concatenate normalized base data for input to the model
#         X2_W2_base_normalised = np.concatenate(
#             [X2_base_normalised, W2_base_normalised], axis=1)

#         # Concatenate normalized SHAP data for calculation of SHAP values
#         X2_W2_shap_normalised = np.concatenate(
#             [X2_shap_normalised, W2_shap_normalised], axis=1)

#         # Use SHAP explainer
#         np.random.seed(42)  # Ensure reproducibility
#         background_X = X2_W2_base_normalised[np.random.permutation(
#             X2_W2_base_normalised.shape[0])]
#         shap_explainer = shap.GradientExplainer(model, [background_X])
#         shap_values_all = shap_explainer.shap_values(X2_W2_shap_normalised)

#         # Use PathExplainerTF for attributions and interactions
#         path_explainer = PathExplainerTF(model)

#         attributions = path_explainer.attributions(
#             inputs=X2_W2_shap_normalised,
#             baseline=background_X,
#             batch_size=500,
#             num_samples=1000,
#             use_expectation=True,
#             output_indices=0,
#             verbose=True)

#         interactions = path_explainer.interactions(
#             inputs=X2_W2_shap_normalised,
#             baseline=background_X,
#             batch_size=500,
#             num_samples=1000,
#             use_expectation=True,
#             output_indices=0,
#             verbose=True)

#         return shap_values_all[0], attributions, interactions

#     elif Z2_base_data.size == 0 and W2_base_data.size == 0:
#         X2_base_normalised = scaler_compo.transform(
#             X2_base_data).astype(np.float32)

#         X2_shap_normalised = scaler_compo.transform(
#             X2_shap_data).astype(np.float32)

#         np.random.seed(42)  # Ensure reproducibility
#         background_X = X2_base_normalised[np.random.permutation(
#             X2_base_normalised.shape[0])]

#          # Use SHAP explainer
#         shap_explainer = shap.GradientExplainer(model, [background_X])
#         shap_values_all = shap_explainer.shap_values(X2_shap_normalised)

#         # Use PathExplainerTF for attributions and interactions
#         path_explainer = PathExplainerTF(model)

#         attributions = path_explainer.attributions(
#             inputs=X2_shap_normalised,
#             baseline=background_X,
#             batch_size=500,
#             num_samples=1000,
#             use_expectation=True,
#             output_indices=0,
#             verbose=True)

#         interactions = path_explainer.interactions(
#             inputs=X2_shap_normalised,
#             baseline=background_X,
#             batch_size=500,
#             num_samples=1000,
#             use_expectation=True,
#             output_indices=0,
#             verbose=True)

#         return shap_values_all[0], attributions, interactions

#     elif Z2_base_data.size != 0 and W2_base_data.size != 0:
#         warnings('this explainer can not handle multi-input model yet')


def compute_shap_attributions_interactions(model, X2_base_data, Z2_base_data, W2_base_data,
                                           X2_shap_data, Z2_shap_data, W2_shap_data,
                                           scaler_compo, scaler_testing, scaler_specific):
    def normalize_data(data, scaler):
        return scaler.transform(data).astype(np.float32)

    def compute_explainer_values(data, background):
        shap_explainer = shap.GradientExplainer(model, [background])
        shap_values_all = shap_explainer.shap_values(data)

        path_explainer = PathExplainerTF(model)
        attributions = path_explainer.attributions(data,
                                                   background,
                                                   batch_size=500,
                                                   num_samples=1000,
                                                   use_expectation=True,
                                                   output_indices=0,
                                                   verbose=True)
        interactions = path_explainer.interactions(data,
                                                   background,
                                                   batch_size=500,
                                                   num_samples=1000,
                                                   use_expectation=True,
                                                   output_indices=0,
                                                   verbose=True)

        return shap_values_all[0], attributions, interactions

    np.random.seed(42)  # Ensure reproducibility

    if Z2_base_data.size == 0:
        X2_base_normalised = normalize_data(X2_base_data, scaler_compo)
        X2_shap_normalised = normalize_data(X2_shap_data, scaler_compo)
        background_X = X2_base_normalised[np.random.permutation(
            X2_base_normalised.shape[0])]

        if W2_base_data.size != 0:
            W2_base_normalised = normalize_data(W2_base_data, scaler_specific)
            W2_shap_normalised = normalize_data(W2_shap_data, scaler_specific)

            X2_W2_base_normalised = np.concatenate(
                [X2_base_normalised, W2_base_normalised], axis=1)
            X2_W2_shap_normalised = np.concatenate(
                [X2_shap_normalised, W2_shap_normalised], axis=1)

            background_X = X2_W2_base_normalised[np.random.permutation(
                X2_W2_base_normalised.shape[0])]

            return compute_explainer_values(X2_W2_shap_normalised, background_X)

        return compute_explainer_values(X2_shap_normalised, background_X)

    elif Z2_base_data.size != 0 and W2_base_data.size != 0:
        warnings.warn('This explainer cannot handle multi-input model yet')


def plot_shap_attributions_interactions(shap_values, attributions, interactions, i_sample,
                                        X1_shap_data, Y1_shap_data, V1_shap_data,
                                        compo_column, specific_testing_column, specific_features_sel_column):

    # Determine columns based on input data
    if Y1_shap_data.size == 0 and V1_shap_data.size == 0:
        columns = compo_column
    if Y1_shap_data.size != 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column + specific_testing_column
    if Y1_shap_data.size == 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column
    if Y1_shap_data.size != 0 and V1_shap_data.size == 0:
        columns = compo_column + specific_testing_column

    # Print aggregated values
    print(
        f"Sum of Shapley values for sample {i_sample}: {shap_values[i_sample, :].sum()}")
    print(
        f"Sum of attributions for sample {i_sample}: {attributions[i_sample, :].sum()}")
    print(
        f"Sum of interactions for sample {i_sample}: {interactions[i_sample, :].sum(axis=-1).sum()}")

    fig, axs = plt.subplots(1, 4, figsize=(
        12, 3), sharex=True, constrained_layout=True)

    # Plot attributions, Shapley values, and diagonal interactions
    axs[0].barh(columns, shap_values[i_sample, :], color='steelblue')
    axs[1].barh(columns, attributions[i_sample, :], color='steelblue')
    axs[2].barh(columns, np.diag(
        interactions[i_sample, :, :]), color='steelblue')

    axs[0].set_title("SHAP")
    axs[1].set_title("Attributions (Janizek model)")
    axs[2].set_title("Interactions (diagonal)")

    # Scatter plot of attributions against summed interactions
    # reshaped_attributions = np.reshape(attributions, -1)
    # summed_interactions = np.reshape(np.sum(interactions, axis=-1), -1)
    reshaped_attributions = np.reshape(attributions[i_sample, :], -1)
    summed_interactions = np.reshape(
        np.sum(interactions[i_sample, :, :], axis=-1), -1)

    # print(reshaped_attributions)
    # print(summed_interactions)
    axs[3].scatter(reshaped_attributions, summed_interactions)
    axs[3].set(
        # xlim=[reshaped_attributions.min(), reshaped_attributions.max()],
        # ylim=[reshaped_attributions.min(), reshaped_attributions.max()],
        aspect='equal', box_aspect=1
    )
    axs[3].plot(
        [reshaped_attributions.min(), reshaped_attributions.max()],
        [reshaped_attributions.min(), reshaped_attributions.max()],
        color='grey'
    )
    axs[3].set_title('Completeness check: Attributions vs. Interactions')

    # Adjust plots for better visualization
    for ax in axs[:-1]:  # Scatter plot excluded
        ax.invert_yaxis()
        ax.set_box_aspect(1)

    plt.show()


def plot_interactions_heatmap(model_path_bo, interactions_values, sample_indices,
                              X1_shap_data, Y1_shap_data, V1_shap_data,
                              compo_column, specific_testing_column, specific_features_sel_column,
                              cmap, vmin, vmax, figsize,
                              save_flag, figname):

    # Determine columns based on input data
    if Y1_shap_data.size == 0 and V1_shap_data.size == 0:
        columns = compo_column
    if Y1_shap_data.size != 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column + specific_testing_column
    if Y1_shap_data.size == 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column
    if Y1_shap_data.size != 0 and V1_shap_data.size == 0:
        columns = compo_column + specific_testing_column

    def get_upper_triangle_mask(matrix):
        """Return a mask for the upper triangle of a matrix, excluding the diagonal."""
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        np.fill_diagonal(mask, True)
        return mask

    fig, axs = plt.subplots(1, len(sample_indices), figsize=(
        figsize * len(sample_indices), figsize), constrained_layout=True)

    for idx, sample_index in enumerate(sample_indices):
        interactions_sample = interactions_values[sample_index, :, :]

        # Filter non-zero rows and columns
        # non_zero_rows = np.any(interactions_sample != 0, axis=1)
        # non_zero_cols = np.any(interactions_sample != 0, axis=0)

        threshold = 1e-5
        non_zero_rows = np.any(np.abs(interactions_sample) > threshold, axis=1)
        non_zero_cols = np.any(np.abs(interactions_sample) > threshold, axis=0)

        filtered_interactions = interactions_sample[non_zero_rows][:, non_zero_cols]
        filtered_labels = np.array(columns)[non_zero_rows]

        # Get mask and annotations for the heatmap
        mask = get_upper_triangle_mask(filtered_interactions)
        annotations = np.array([[f'({filtered_labels[i]},{filtered_labels[j]})'
                                 for j in range(filtered_interactions.shape[1])]
                                for i in range(filtered_interactions.shape[0])], dtype=object)

        ax = axs[idx] if len(sample_indices) > 1 else axs
        # Display colorbar only for the last subplot
        cbar = idx == len(sample_indices) - 1

        # print(filtered_interactions)
        # print(annotations)

        # Display heatmap
        heatmap = sns.heatmap(filtered_interactions, ax=ax, mask=mask, annot=annotations, fmt="",
                              cmap=cmap, vmin=vmin, vmax=vmax,
                              xticklabels=filtered_labels, yticklabels=filtered_labels,
                              cbar=cbar, annot_kws={"size": 14, "weight": "bold"})

        if cbar:
            # Access the colorbar object and set its label
            cbar = heatmap.collections[0].colorbar
            cbar.set_label('Interaction Values', rotation=270,
                           labelpad=20, fontsize=14)
            cbar.ax.tick_params(labelsize=12)

        ax.set_box_aspect(1)
        ax.set_title(f"Correlation matrix - Sample {sample_index+1}",  pad=20)

        # Fine-tune visualization
        for t in ax.texts:
            t.set_size(10)
        ax.tick_params(axis='x', labelrotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    # plt.tight_layout()
    if save_flag:
        plt.savefig(model_path_bo + figname + '.pdf', bbox_inches='tight')
        plt.show()
    else:
        # plt.close(fig)
        plt.show()

# def plot_interactions_heatmap(model_path_bo, interactions_values, sample_indices, col_labels,
#                               cmap, vmin, vmax, save_flag, figname):
#     """
#     Plot heatmaps for interaction values, highlighting non-zero interactions.
#     """
#     def get_upper_triangle_mask(matrix):
#         mask = np.triu(np.ones_like(matrix, dtype=bool))
#         np.fill_diagonal(mask, True)
#         return mask

#     num_plots = len(sample_indices)
#     fig, axs = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
#     axs = axs if num_plots > 1 else [axs]

#     for idx, (ax, sample_index) in enumerate(zip(axs, sample_indices)):
#         interactions_sample = interactions_values[sample_index, :, :]
#         threshold = 1e-5
#         non_zero_rows = np.any(np.abs(interactions_sample) > threshold, axis=1)
#         non_zero_cols = np.any(np.abs(interactions_sample) > threshold, axis=0)
#         filtered_interactions = interactions_sample[non_zero_rows][:, non_zero_cols]
#         filtered_labels = np.array(col_labels)[non_zero_rows]

#         mask = get_upper_triangle_mask(filtered_interactions)
#         sns.heatmap(filtered_interactions, ax=ax, mask=mask, annot=False,
#                     cmap=cmap, vmin=vmin, vmax=vmax,
#                     xticklabels=filtered_labels, yticklabels=filtered_labels,
#                     cbar=idx == num_plots - 1)

#         if idx == num_plots - 1:
#             ax.figure.colorbar(
#                 ax.collections[0], ax=ax, label='Interaction Values', orientation='vertical')

#         ax.set_title(f"Correlation matrix - Sample {sample_index + 1}")
#         ax.tick_params(axis='x', labelrotation=45, labelsize=10)
#         ax.tick_params(axis='y', labelsize=10)

#     plt.tight_layout()
#     if save_flag:
#         plt.savefig(model_path_bo + figname + '.pdf', bbox_inches='tight')
#     plt.show()


class ModelExplainer:
    def __init__(self, model_path_bo, model_name, k_folds, n_CVrepeats, mc_repeat,
                 scaler_compo, scaler_testing, scaler_specific):
        """
        Initialize ModelExplainer with model configurations and scalers.

        Parameters:
        - model_path_bo: Path to the model.
        - model_name: Name of the model.
        - k_folds: Number of K-Folds for cross-validation.
        - n_CVrepeats: Number of cross-validation repeats.
        - mc_repeat: Number of Monte Carlo repetitions.
        - scaler_compo: Scaler for components.
        - scaler_testing: Scaler for testing.
        - scaler_specific: Scaler for specific features.
        """
        self.model_path_bo = model_path_bo
        self.model_name = model_name
        self.k_folds = k_folds
        self.n_CVrepeats = n_CVrepeats
        self.mc_repeat = mc_repeat
        self.scaler_compo = scaler_compo
        self.scaler_testing = scaler_testing
        self.scaler_specific = scaler_specific

    def _load_model(self, i):
        """Load model from specified path."""
        return keras.models.load_model(os.path.join(self.model_path_bo, self.model_name.format(i+1)))

    def _prepare_input_data(self, X_list, V_list, i):
        """Prepare and normalise input data for prediction."""
        if V_list[i].size != 0:
            X_normalised = self.scaler_compo.transform(X_list[i])
            V_normalised = self.scaler_specific.transform(V_list[i])
            return np.concatenate([X_normalised, V_normalised], axis=1)
        else:
            return self.scaler_compo.transform(X_list[i])

    def _define_predict_norm_function(self, model, input_data, Y_list, i):
        """Define the normalised prediction function for the model."""
        if Y_list[i].size != 0:
            Y_normalised = self.scaler_testing.transform(Y_list[i])
            return lambda: model.predict([input_data, Y_normalised], verbose=0)
        else:
            return lambda: model.predict(input_data, verbose=0)

    @tf.autograph.experimental.do_not_convert
    def _predict_monte_carlo_sampling(self, predict_func):
        """Predict using Monte Carlo sampling."""
        predictions = tf.map_fn(lambda _: predict_func(),
                                tf.range(self.mc_repeat),
                                dtype=tf.float32,
                                parallel_iterations=self.mc_repeat)
        return predictions.numpy(), predictions.numpy().mean(axis=0).reshape((-1,)), predictions.numpy().std(axis=0).reshape((-1,))

    def _predict_for_one_fold(self, X_list, Y_list, V_list, i):
        """Predict for a specific fold."""
        model = self._load_model(i)
        input_data = self._prepare_input_data(
            X_list, V_list, i)
        predict_func = self._define_predict_norm_function(
            model, input_data, Y_list, i)

        return self._predict_monte_carlo_sampling(predict_func)

    # def _shap_for_one_fold(self,
    #                        X_base_list, Y_base_list, V_base_list,
    #                        X_shap_list, Y_shap_list, V_shap_list, i):
    #     """Compute averaged SHAP values for a specific fold, over multiple iterations with Monte Carlo dropout."""

    #     model = self._load_model(i)
    #     input_base_data = self._prepare_input_data(X_base_list, V_base_list, i)
    #     input_shap_data = self._prepare_input_data(X_shap_list, V_shap_list, i)

    #     # ===== create an explainer based on base_data =====
    #     if Y_base_list[i].size != 0:
    #         Y_base_normalised = self.scaler_testing.transform(Y_base_list[i])
    #         explainer = shap.KernelExplainer(
    #             model, [input_base_data, Y_base_normalised])
    #     else:
    #         explainer = shap.KernelExplainer(model, input_base_data)

    #     # Calculate shap values for one fold
    #     if Y_shap_list[i].size != 0:
    #         Y_shap_normalised = self.scaler_testing.transform(Y_shap_list[i])
    #         shap_values_oneFold = explainer.shap_values(
    #             [input_shap_data, Y_shap_normalised])
    #     else:
    #         shap_values_oneFold = explainer.shap_values(input_shap_data)

    #     # when model contains 2 inputs, the shap_for_one_fold provides a list of 2 shap arrays
    #     # so I will concatenate the 2 shap arrays into 1
    #     if Y_shap_list[i].size != 0:
    #         # print(len(shap_values_oneFold[0]))
    #         return np.concatenate([shap_values_oneFold[0][0], shap_values_oneFold[0][1]], axis=1)
    #     else:
    #         # print('SHAP shape : ', shap_values_oneFold[0].shape)
    #         return shap_values_oneFold[0]

    def _shap_for_one_fold(self,
                           X_base_list, Y_base_list, V_base_list,
                           X_shap_list, Y_shap_list, V_shap_list,
                           i):
        """Compute averaged SHAP values for a specific fold, over multiple iterations with Monte Carlo dropout."""
        model = self._load_model(i)
        input_base_data = self._prepare_input_data(X_base_list, V_base_list, i)
        input_shap_data = self._prepare_input_data(X_shap_list, V_shap_list, i)

        # Initialize an array to store SHAP values for each repeat
        shap_values_all_repeats = []

        for _ in range(self.mc_repeat):
            # Enable dropout in your model here if it's not enabled by default

            # ----- Create an explainer based on base_data -----
            if Y_base_list[i].size != 0:
                Y_base_normalised = self.scaler_testing.transform(
                    Y_base_list[i])
                explainer = shap.GradientExplainer(
                    model, [input_base_data, Y_base_normalised])
            else:
                explainer = shap.GradientExplainer(model, [input_base_data])

            # Calculate SHAP values for one iteration
            if Y_shap_list[i].size != 0:
                Y_shap_normalised = self.scaler_testing.transform(
                    Y_shap_list[i])
                shap_values = explainer.shap_values(
                    [input_shap_data, Y_shap_normalised])
            else:
                shap_values = explainer.shap_values(input_shap_data)

            # Concatenate the SHAP arrays if there are 2 inputs
            if Y_shap_list[i].size != 0:
                shap_values = np.concatenate(
                    [shap_values[0][0], shap_values[0][1]], axis=1)
            else:
                shap_values = shap_values[0]

            shap_values_all_repeats.append(shap_values)

        # end of for loop
        # Average the SHAP values over all repeats
        averaged_shap_values = np.mean(
            np.array(shap_values_all_repeats), axis=0)

        return averaged_shap_values

    # def _attributions_interactions_for_one_fold(self,
    #                                             X_base_list, Y_base_list, V_base_list,
    #                                             X_shap_list, Y_shap_list, V_shap_list, i):
    #     """Compute attributions and interactions for a specific fold."""
    #     model = self._load_model(i)
    #     input_base_data = self._prepare_input_data(X_base_list, V_base_list, i)
    #     input_shap_data = self._prepare_input_data(X_shap_list, V_shap_list, i)

    #     # Initialize an array to store SHAP values for each repeat
    #     attributions_values_all_repeats = []
    #     interactions_values_all_repeats = []

    #     for _ in range(self.mc_repeat):

    #         # ===== create an explainer based on base_data =====
    #         if Y_base_list[i].size != 0:
    #             pass
    #         else:
    #             explainer = PathExplainerTF(model)

    #         # Calculate feature attribution/interaction (Janizek) values for one fold
    #         if Y_shap_list[i].size != 0:
    #             warnings.warn(
    #                 "feature attribution doesn't support multi-input models.")
    #             attributions_values, interactions_values = np.empty(
    #                 (0, 0)), np.empty((0, 0))
    #         else:
    #             np.random.seed(42)
    #             background_X_indices = np.random.permutation(
    #                 input_base_data.shape[0])
    #             background_X = input_base_data[background_X_indices].astype(
    #                 np.float32)

    #             attributions_values = explainer.attributions(inputs=input_shap_data.astype(np.float32),
    #                                                          baseline=background_X,
    #                                                          batch_size=500,
    #                                                          num_samples=1000,
    #                                                          use_expectation=True,
    #                                                          output_indices=0,
    #                                                          verbose=True)

    #             interactions_values = explainer.interactions(inputs=input_shap_data.astype(np.float32),
    #                                                          baseline=background_X,
    #                                                          batch_size=500,
    #                                                          num_samples=1000,
    #                                                          use_expectation=True,
    #                                                          output_indices=0,
    #                                                          verbose=True)

    #         attributions_values_all_repeats.append(attributions_values)
    #         interactions_values_all_repeats.append(interactions_values)

    #     # end of for loop
    #     average_attributions_values = np.mean(
    #         np.array(attributions_values_all_repeats), axis=0)
    #     average_interactions_values = np.mean(
    #         np.array(interactions_values_all_repeats), axis=0)

    #     return average_attributions_values, average_interactions_values

    def _attributions_interactions_for_one_fold(self,
                                                X_base_list, Y_base_list, V_base_list,
                                                X_shap_list, Y_shap_list, V_shap_list, i):
        """Compute attributions and interactions for a specific fold."""
        model = self._load_model(i)
        input_base_data = self._prepare_input_data(X_base_list, V_base_list, i)
        input_shap_data = self._prepare_input_data(X_shap_list, V_shap_list, i)

        # Initialize an array to store SHAP values for each repeat
        attributions_values_all_repeats = []
        interactions_values_all_repeats = []

        # ===== create an explainer based on base_data =====
        if Y_base_list[i].size != 0:
            pass
        else:
            explainer = PathExplainerTF(model)

        # Calculate feature attribution/interaction (Janizek) values for one fold
        if Y_shap_list[i].size != 0:
            warnings.warn(
                "feature attribution doesn't support multi-input models.")
            attributions_values, interactions_values = np.empty(
                (0, 0)), np.empty((0, 0))
        else:
            np.random.seed(42)
            background_X_indices = np.random.permutation(
                input_base_data.shape[0])
            background_X = input_base_data[background_X_indices].astype(
                np.float32)

            attributions_values = explainer.attributions(inputs=input_shap_data.astype(np.float32),
                                                         baseline=background_X,
                                                         batch_size=500,
                                                         num_samples=1000,
                                                         use_expectation=True,
                                                         output_indices=0,
                                                         verbose=True)

            interactions_values = explainer.interactions(inputs=input_shap_data.astype(np.float32),
                                                         baseline=background_X,
                                                         batch_size=500,
                                                         num_samples=1000,
                                                         use_expectation=True,
                                                         output_indices=0,
                                                         verbose=True)

        return attributions_values, interactions_values

    def predict_shap_bootstrap_norm(self,
                                    X1_base_list, Y1_base_list, V1_base_list,
                                    X1_shap_list, Y1_shap_list, V1_shap_list):
        """Predict, compute SHAP values, attributions, and interactions using bootstrap method."""
        results_predict_base = Parallel(n_jobs=-1)(delayed(self._predict_for_one_fold)(X1_base_list, Y1_base_list, V1_base_list, i)
                                                   for i in range(self.k_folds * self.n_CVrepeats))

        results_predict_shap = Parallel(n_jobs=-1)(delayed(self._predict_for_one_fold)(X1_shap_list, Y1_shap_list, V1_shap_list, i)
                                                   for i in range(self.k_folds * self.n_CVrepeats))

        results_shap = Parallel(n_jobs=-1)(delayed(self._shap_for_one_fold)(X1_base_list, Y1_base_list, V1_base_list,
                                                                            X1_shap_list, Y1_shap_list, V1_shap_list, i)
                                           for i in range(self.k_folds * self.n_CVrepeats))

        predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std = zip(
            *results_predict_base)
        predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std = zip(
            *results_predict_shap)
        shap_list = results_shap

        tf.keras.backend.clear_session()

        return (predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std,
                predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std,
                shap_list)

    def predict_shap_attributions_interactions_bootstrap_norm(self,
                                                              X1_base_list, Y1_base_list, V1_base_list,
                                                              X1_shap_list, Y1_shap_list, V1_shap_list):
        """Predict, compute SHAP values, attributions, and interactions using bootstrap method."""
        results_predict_base = Parallel(n_jobs=-1)(delayed(self._predict_for_one_fold)(X1_base_list, Y1_base_list, V1_base_list, i)
                                                   for i in range(self.k_folds * self.n_CVrepeats))

        results_predict_shap = Parallel(n_jobs=-1)(delayed(self._predict_for_one_fold)(X1_shap_list, Y1_shap_list, V1_shap_list, i)
                                                   for i in range(self.k_folds * self.n_CVrepeats))

        results_shap = Parallel(n_jobs=-1)(delayed(self._shap_for_one_fold)(X1_base_list, Y1_base_list, V1_base_list,
                                                                            X1_shap_list, Y1_shap_list, V1_shap_list, i)
                                           for i in range(self.k_folds * self.n_CVrepeats))

        # Unpacking results
        predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std = zip(
            *results_predict_base)
        predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std = zip(
            *results_predict_shap)
        shap_list = results_shap

        # Calculating attributions and interactions
        attributions_list, interactions_list = [], []
        total_folds = self.k_folds * self.n_CVrepeats

        # Loop with a progress bar
        for i in tqdm(range(total_folds), desc="Processing Folds"):

            results_attributions_interactions = Parallel(n_jobs=-1)(delayed(self._attributions_interactions_for_one_fold)(X1_base_list, Y1_base_list, V1_base_list,
                                                                                                                          X1_shap_list, Y1_shap_list, V1_shap_list, i)
                                                                    for _ in range(self.mc_repeat))

            attributions_mc, interactions_mc = zip(
                *results_attributions_interactions)
            attributions_list.append(
                np.mean(np.array(attributions_mc), axis=0))
            interactions_list.append(
                np.mean(np.array(interactions_mc), axis=0))

        # results_attributions_interactions = Parallel(n_jobs=-1)(delayed(self._attributions_interactions_for_one_fold)(X1_base_list, Y1_base_list, V1_base_list,
        #                                                                                                               X1_shap_list, Y1_shap_list, V1_shap_list, i)
        #                                                         for i in range(self.k_folds * self.n_CVrepeats))

        # attributions_list, interactions_list = zip(
        #     *results_attributions_interactions)

        tf.keras.backend.clear_session()

        return (predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std,
                predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std,
                shap_list, attributions_list, interactions_list)

# end of the ModelExplainer class


def inverse_norm_shap(pred_norm_base_stack,
                      pred_norm_shap_stack,
                      shap_norm_stack,
                      scaler_output):
    """
    Transform predictions and Shapley values from normalised to original space.

    Parameters:
    - pred_norm_base_stack (list of np.array): normalised predictions for base data.
    - pred_norm_shap_stack (list of np.array): normalised predictions for Shap data.
    - shap_norm_stack (list of np.array): normalised Shapley values.
    - scaler_output: Scaler for the model output.

    Returns:
    - tuple: Mean and standard deviation of predictions, Shapley values, attribution and interactions values in both spaces.
    """

    # Process normalised predictions for base input data (baseline)
    pred_norm_base_conc = np.concatenate(pred_norm_base_stack, axis=0)
    pred_norm_base_mean = np.mean(pred_norm_base_conc, axis=0).reshape(-1)
    pred_norm_base_std = np.std(pred_norm_base_conc, axis=0).reshape(-1)

    # Inverse transform the predictions for the base input data
    pred_base_list = [scaler_output.inverse_transform(
        pred) for pred in pred_norm_base_conc]
    pred_base_mean = np.mean(pred_base_list, axis=0)
    pred_base_std = np.std(pred_base_list, axis=0)

    # Process normalised predictions for Shap input data (target)
    pred_norm_shap_conc = np.concatenate(pred_norm_shap_stack, axis=0)
    pred_norm_shap_mean = np.mean(pred_norm_shap_conc, axis=0).reshape(-1)
    pred_norm_shap_std = np.std(pred_norm_shap_conc, axis=0).reshape(-1)

    # Inverse transform the predictions for the Shap input data
    pred_shap_list = [scaler_output.inverse_transform(
        pred) for pred in pred_norm_shap_conc]
    pred_shap_mean = np.mean(pred_shap_list, axis=0)
    pred_shap_std = np.std(pred_shap_list, axis=0)

    # Compute mean and standard deviation of Shapley values in the normalised space
    shap_norm_mean = np.mean(shap_norm_stack, axis=0)
    shap_norm_std = np.std(shap_norm_stack, axis=0)

    # Sanity check: Ensure that the sum of Shapley values and the mean prediction for the base input data
    # is approximately equal to the mean prediction for the Shap input data in the normalised space
    pred_norm_base_mean_AVG = pred_norm_base_mean.mean()
    # print(pred_norm_base_mean_AVG)
    # print(np.abs(pred_norm_base_mean_AVG +
    #              shap_norm_mean.sum(axis=1) - pred_norm_shap_mean))
    # -----<if epsilon is not passing assert: try to increase the mc repeats>-----
    epsilon = np.abs(pred_norm_base_mean_AVG*0.1)  # 1e-1
    # print(epsilon)
    assert (np.abs(pred_norm_base_mean_AVG +
            shap_norm_mean.sum(axis=1) - pred_norm_shap_mean) < epsilon).all()

    # Inverse transform the Shapley values
    pred_base_mean_AVG = pred_base_mean.mean()
    diff = pred_shap_mean - pred_base_mean_AVG
    shapley_scaler = np.divide(diff, shap_norm_mean.sum(axis=1).reshape(-1, 1))
    # print(shap_norm_mean.shape, shapley_scaler.shape)

    # I now use the shapley_scaler to inverse normalise both the shap
    shap_mean = np.multiply(shap_norm_mean, shapley_scaler)

    return (pred_norm_base_mean, pred_norm_shap_mean, shap_norm_mean,
            pred_base_mean, pred_shap_mean, shap_mean)


def inverse_norm_shap_attributions_interactions(pred_norm_base_stack, pred_norm_shap_stack,
                                                shap_norm_stack,
                                                attributions_norm_stack,
                                                interactions_norm_stack, scaler_output):
    """
    Transform predictions and Shapley values from normalised to original space.

    Parameters:
    - pred_norm_base_stack (list of np.array): normalised predictions for base data.
    - pred_norm_shap_stack (list of np.array): normalised predictions for Shap data.
    - shap_norm_stack (list of np.array): normalised Shapley values.
    - attributions_norm_stack (list of np.array): normalised attribution values.
    - interactions_norm_stack (list of np.array): normalised interaction values.
    - scaler_output: Scaler for the model output.

    Returns:
    - tuple: Mean and standard deviation of predictions, Shapley values, attribution and interactions values in both spaces.
    """

    # Process normalised predictions for base input data (baseline)
    pred_norm_base_conc = np.concatenate(pred_norm_base_stack, axis=0)
    pred_norm_base_mean = np.mean(pred_norm_base_conc, axis=0).reshape(-1)
    pred_norm_base_std = np.std(pred_norm_base_conc, axis=0).reshape(-1)

    # Inverse transform the predictions for the base input data
    pred_base_list = [scaler_output.inverse_transform(
        pred) for pred in pred_norm_base_conc]
    pred_base_mean = np.mean(pred_base_list, axis=0)
    pred_base_std = np.std(pred_base_list, axis=0)

    # Process normalised predictions for Shap input data (target)
    pred_norm_shap_conc = np.concatenate(pred_norm_shap_stack, axis=0)
    pred_norm_shap_mean = np.mean(pred_norm_shap_conc, axis=0).reshape(-1)
    pred_norm_shap_std = np.std(pred_norm_shap_conc, axis=0).reshape(-1)

    # Inverse transform the predictions for the Shap input data
    pred_shap_list = [scaler_output.inverse_transform(
        pred) for pred in pred_norm_shap_conc]
    pred_shap_mean = np.mean(pred_shap_list, axis=0)
    pred_shap_std = np.std(pred_shap_list, axis=0)

    # Compute mean and standard deviation of Shapley values in the normalised space
    shap_norm_mean = np.mean(shap_norm_stack, axis=0)
    shap_norm_std = np.std(shap_norm_stack, axis=0)

    # Compute mean and standard deviation of Attribution values in the normalised space
    attributions_norm_mean = np.mean(attributions_norm_stack, axis=0)
    attributions_norm_std = np.std(attributions_norm_stack, axis=0)

    # Compute mean and standard deviation of Interaction values values in the normalised space
    interactions_norm_mean = np.mean(interactions_norm_stack, axis=0)
    interactions_norm_std = np.std(interactions_norm_stack, axis=0)

    # Sanity check: Ensure that the sum of Shapley values and the mean prediction for the base input data
    # is approximately equal to the mean prediction for the Shap input data in the normalised space
    pred_norm_base_mean_AVG = pred_norm_base_mean.mean()

    # print(pred_norm_base_mean_AVG)
    # print(pred_norm_base_mean_AVG)

    # print(np.abs(pred_norm_base_mean_AVG +
    #              shap_norm_mean.sum(axis=1) - pred_norm_shap_mean))
    epsilon = np.abs(pred_norm_base_mean_AVG*0.05)  # 1e-1
    assert (np.abs(pred_norm_base_mean_AVG +
            shap_norm_mean.sum(axis=1) - pred_norm_shap_mean) < epsilon).all()

    # Inverse transform the Shapley values
    pred_base_mean_AVG = pred_base_mean.mean()
    diff = pred_shap_mean - pred_base_mean_AVG
    shapley_scaler = np.divide(diff, shap_norm_mean.sum(axis=1).reshape(-1, 1))
    # print(shap_norm_mean.shape, shapley_scaler.shape)

    # I now use the shapley_scaler to inverse normalise both the shap and attributions because I know they are almost the same
    shap_mean = np.multiply(shap_norm_mean, shapley_scaler)
    attributions_mean = np.multiply(attributions_norm_mean, shapley_scaler)

    # I now use the shapley_scaler to inverse normalise the interactions (I always check if it is almost equivalent to the attributions by plotting)
    # below I need to reshape the shapley scaler to work with the dimensions of interactions array
    shapley_scaler_reshaped = shapley_scaler[:, np.newaxis]
    # print(shapley_scaler_reshaped.shape)
    interactions_mean = np.multiply(
        interactions_norm_mean, shapley_scaler_reshaped)

    return (pred_norm_base_mean, pred_norm_shap_mean, shap_norm_mean, attributions_norm_mean, interactions_norm_mean,
            pred_base_mean, pred_shap_mean, shap_mean, attributions_mean, interactions_mean)


def data_for_shap_force(X1_shap_data, Y1_shap_data, V1_shap_data,
                        compo_column, C_specific_testing_column, specific_features_sel_column,
                        pred_norm_base_KFold_mean, shap_KFold_mean,
                        sample_index=[0, 1]):
    """
    Extracts and displays SHAP values for specific samples from the dataset.

    Parameters:
    - X1_shap_data, Y1_shap_data, V1_shap_data: Input data arrays.
    - compo_column, C_specific_testing_column, specific_features_sel_column: Column names.
    - pred_norm_base_KFold_mean, pred_norm_shap_KFold_mean, shap_KFold_mean: Prediction and SHAP data.
    - sample_index: Indices of samples to extract.

    Returns:
    - Average of pred_norm_base_KFold_mean.
    - SHAP values of the selected samples.
    - Columns used for the data.
    """

    # Combine input data
    # sample_dataset = np.hstack((X1_shap_data, Y1_shap_data, V1_shap_data))

    # Determine columns based on input data
    columns = compo_column
    if Y1_shap_data.size != 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column + C_specific_testing_column
    if Y1_shap_data.size == 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column
    if Y1_shap_data.size != 0 and V1_shap_data.size == 0:
        columns = compo_column + C_specific_testing_column

    # Extract and display SHAP values for the selected samples
    sample_shap_values = shap_KFold_mean[sample_index, :]
    # print(sample_shap_values.shape)
    # print(len(columns))
    # display(pd.DataFrame(data=sample_shap_values, columns=columns))

    return pred_norm_base_KFold_mean.mean(), sample_shap_values, columns

# --------------------------------------------------
# plotting the force plot for corrosion network (individual)
# --------------------------------------------------
# import shap
# from utils.postprocessing_explainer import data_for_shap_force
# from IPython.display import display

# shap.initjs()
# sample_indices = [49, 51, 53, 55]
# sample_indices = [x-1 for x in sample_indices]

# for sample_index in sample_indices:
#     A_baseline, B_shap_values, C_column_names = data_for_shap_force(X2_shap_data, Z2_shap_data, W2_shap_data,
#                                                                     compo_column, C_specific_testing_column, specific_features_sel_column,
#                                                                     C2_pred_X2_base_KFold_mean, C2_shap_X2_KFold_mean,
#                                                                     sample_index=[sample_index])
#     # shap.initjs()
#     shap_html = shap.force_plot(
#         A_baseline,
#         B_shap_values,
#         C_column_names,
#         link='identity',
#         matplotlib=False,
#         figsize=(5, 2.6),
#         text_rotation=45,
#         contribution_threshold=0.1)

#     display(shap_html)  # Display the plot in th
#     shap.save_html(
#         model_path_bo + f"shap_force_{shap_fname}_NNC_{sample_index+1}.html", shap_html)


# --------------------------------------------------
# plotting the force plot for corrosion network (merged)
# --------------------------------------------------
# sample_index = [49, 51, 53, 55]
# # sample_index = [12]

# sample_index = [x-1 for x in sample_index]
# A_baseline, B_shap_values, C_column_names = data_for_shap_force(X2_shap_data, Z2_shap_data, W2_shap_data,
#                                                                 compo_column, C_specific_testing_column, specific_features_sel_column,
#                                                                 C2_pred_X2_base_KFold_mean, C2_shap_X2_KFold_mean,
#                                                                 sample_index=sample_index)
# shap.initjs()
# shap.force_plot(
#     A_baseline,
#     B_shap_values,
#     C_column_names,
#     link='identity',
#     matplotlib=False,
#     figsize=(25, 3),
#     text_rotation=45,
#     contribution_threshold=0.001)


def plot_shap_summary(shap_KFold_mean, feature_names, is_abs=True,
                      title='Feature Importance', figsize=(5, 5), palette='twilight_shifted_r'):
    """
    Plot SHAP values in descending order of importance.
    """
    if is_abs:
        shap_KFold_mean_avg = np.abs(shap_KFold_mean).mean(axis=0)
    else:
        shap_KFold_mean_avg = shap_KFold_mean.mean(axis=0)

    # Create DataFrame from SHAP values and features
    shap_df = pd.DataFrame(
        data={'SHAP Value': shap_KFold_mean_avg, 'Feature': feature_names})

    # Sort SHAP values and filter out zeros
    shap_df = shap_df.sort_values(by='SHAP Value', ascending=False)
    shap_df = shap_df[shap_df['SHAP Value'] != 0]

    # Assuming shap_df, figsize, palette, and title are defined
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=shap_df['SHAP Value'],
                     y=shap_df['Feature'], palette=palette)
    ax.set_xlabel('Mean Absolute SHAP Attribution Value')
    ax.set_title(title)

    # Set the aspect ratio here
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

# plot_shap_summary(H1_shap_X1_KFold_mean, compo_column, is_abs=True,
#                   title='Feature Importance - Hardness', figsize=(4, 4), palette='twilight_shifted_r')

# plot_shap_summary(C2_shap_X2_KFold_mean, compo_column, is_abs=True,
#                   title='Feature Importance - Corrosion', figsize=(4, 4), palette='twilight_shifted_r')


def plot_interactions_summary(interactions_values, feature_names, is_abs=True,
                              title='Feature Importance', figsize=(5, 5), palette='twilight_shifted_r'):
    """
    Plots the interaction values in a descending order of importance.
    """

    # Calculate average of interactions_values
    if is_abs:
        interactions_avg = np.abs(interactions_values).mean(axis=0)
    else:
        interactions_avg = interactions_values.mean(axis=0)

    # Filter non-zero rows and columns
    # non_zero_rows = np.any(interactions_avg != 0, axis=1)
    # non_zero_cols = np.any(interactions_avg != 0, axis=0)
    threshold = 1e-5
    non_zero_rows = np.any(np.abs(interactions_avg) > threshold, axis=1)
    non_zero_cols = np.any(np.abs(interactions_avg) > threshold, axis=0)
    filtered_interactions = interactions_avg[non_zero_rows][:, non_zero_cols]
    filtered_labels = np.array(feature_names)[non_zero_rows]

    # Generate annotations
    annotations = np.array([[f'({filtered_labels[i]},{filtered_labels[j]})'
                             for j in range(filtered_interactions.shape[1])]
                            for i in range(filtered_interactions.shape[0])], dtype=object)

    def get_upper_triangle_mask(matrix):
        """Return a mask for the upper triangle of a matrix, excluding the diagonal."""
        mask_inverse = np.triu(np.ones_like(matrix, dtype=bool))
        np.fill_diagonal(mask_inverse, True)
        return mask_inverse

    mask_inverse = get_upper_triangle_mask(filtered_interactions)

    # Extract values using mask
    filtered_interactions_masked = filtered_interactions[mask_inverse]
    annotations_masked = annotations[mask_inverse]

    # Organize into DataFrame
    interactions_df = pd.DataFrame(
        data={'Interactions Value': filtered_interactions_masked, 'Feature': annotations_masked})

    # Sort and filter
    interactions_df = interactions_df.sort_values(
        by='Interactions Value', ascending=False)
    interactions_df = interactions_df[interactions_df['Interactions Value'] != 0]

    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x=interactions_df['Interactions Value'],
                y=interactions_df['Feature'], palette=palette)
    plt.xlabel('Mean Absolute Interactions Value')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_interactions_summary_split(interactions_values, feature_names, is_abs=True,
                                    title='Feature Importance', figsize=(12, 3), palette='twilight_shifted_r'):
    """
    Plots the interaction values in a descending order of importance.
    Two subplots: one for diagonal (self-interactions) and one for non-diagonal interactions.
    """

    # Calculate average of interactions_values
    if is_abs:
        interactions_avg = np.abs(interactions_values).mean(axis=0)
    else:
        interactions_avg = interactions_values.mean(axis=0)

    # Filter non-zero rows and columns
    threshold = 1e-5
    non_zero_rows = np.any(np.abs(interactions_avg) > threshold, axis=1)
    non_zero_cols = np.any(np.abs(interactions_avg) > threshold, axis=0)
    filtered_interactions = interactions_avg[non_zero_rows][:, non_zero_cols]
    filtered_labels = np.array(feature_names)[non_zero_rows]

    # Generate annotations
    annotations = np.array([[f'({filtered_labels[i]},{filtered_labels[j]})'
                             for j in range(filtered_interactions.shape[1])]
                            for i in range(filtered_interactions.shape[0])], dtype=object)

    # Diagonal (self-interactions)
    diagonal_values = np.diag(filtered_interactions)
    diagonal_annotations = [
        f'({filtered_labels[i]},{filtered_labels[i]})' for i in range(len(diagonal_values))]
    diagonal_df = pd.DataFrame(
        {'Interactions Value': diagonal_values, 'Feature': diagonal_annotations})
    diagonal_df = diagonal_df.sort_values(
        by='Interactions Value', ascending=False)
    diagonal_df = diagonal_df[diagonal_df['Interactions Value'] != 0]

    # Non-diagonal interactions

    def get_upper_triangle_mask(matrix):
        """Return a mask for the upper triangle of a matrix, excluding the diagonal."""
        mask_inverse = np.triu(np.ones_like(matrix, dtype=bool))
        np.fill_diagonal(mask_inverse, False)
        return mask_inverse

    mask_inverse = get_upper_triangle_mask(filtered_interactions)

    # Extract values using mask
    non_diagonal_values = filtered_interactions[mask_inverse]
    non_diagonal_annotations = annotations[mask_inverse]

    non_diagonal_df = pd.DataFrame(
        {'Interactions Value': non_diagonal_values, 'Feature': non_diagonal_annotations})
    non_diagonal_df = non_diagonal_df.sort_values(
        by='Interactions Value', ascending=False)
    non_diagonal_df = non_diagonal_df[non_diagonal_df['Interactions Value'] != 0]

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Diagonal plot
    sns.barplot(x='Interactions Value', y='Feature',
                data=diagonal_df, palette=palette, ax=axs[0])
    axs[0].set_title('Diagonal (Self-Interactions)')
    axs[0].set_xlabel('Mean Absolute Interactions Value')
    axs[0].set_box_aspect(1)

    # Non-diagonal plot
    sns.barplot(x='Interactions Value', y='Feature',
                data=non_diagonal_df.head(5), palette=palette, ax=axs[1])
    axs[1].set_title('Non-Diagonal Interactions')
    axs[1].set_xlabel('Mean Absolute Interactions Value')
    axs[1].set_box_aspect(1)

    # Overall layout adjustments
    # plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# plot_interactions_summary(H1_interactions_X1_KFold_mean, compo_column, is_abs=True,
#                           title='Interaction Importance - Hardness', figsize=(4, 4), palette='twilight_shifted_r')

# plot_interactions_summary(C2_interactions_X2_KFold_mean, compo_column, is_abs=True,
#                           title='Interaction Importance - Corrosion', figsize=(4, 4), palette='twilight_shifted_r')
