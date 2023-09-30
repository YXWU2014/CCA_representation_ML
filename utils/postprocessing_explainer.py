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


def compute_shap_attributions_interactions(model, X1_base_normalized, X1_shap_normalized):
    """
    Compute Shapley values, attributions, and interactions for the provided model and data.

    Parameters:
    - model: Trained model for which explanations are needed.
    - X1_base_normalized (array): Data for background/reference set.
    - X1_shap_normalized (array): Data for which explanations are computed.

    Returns:
    - tuple: Shapley values, attributions, and interactions.
    """

    np.random.seed(42)  # Ensure reproducibility

    # Create random background data subset
    background_X = X1_base_normalized[np.random.permutation(
        X1_base_normalized.shape[0])]

    # Obtain Shapley values using KernelExplainer
    shap_values_all = shap.KernelExplainer(
        model, background_X).shap_values(X1_shap_normalized)

    # Use PathExplainerTF for attributions and interactions
    explainer = PathExplainerTF(model)

    attributions = explainer.attributions(
        inputs=X1_shap_normalized,
        baseline=background_X,
        batch_size=500,
        num_samples=1000,
        use_expectation=True,
        output_indices=0,
        verbose=True
    )

    interactions = explainer.interactions(
        inputs=X1_shap_normalized,
        baseline=background_X,
        batch_size=500,
        num_samples=1000,
        use_expectation=True,
        output_indices=0,
        verbose=True
    )

    return shap_values_all[0], attributions, interactions


def plot_shap_attributions_interactions(shap_values, attributions, interactions, i_KFold, compo_column):
    """
    Visualize the attributions, Shapley values, diagonal interaction values, and their scatter plot.

    Parameters:
    - shap_values (array): Shapley values for each feature.
    - attributions (array): Attributions for each feature.
    - interactions (array): Interaction values between features.
    - i_KFold (int): Index for the fold being visualized.
    - compo_column (list): Column names or labels for features.

    Returns:
    - None: Displays the plots.
    """

    # Print aggregated values
    print(
        f"Sum of Shapley values for sample {i_KFold}: {shap_values[i_KFold, :].sum()}")
    print(
        f"Sum of attributions for sample {i_KFold}: {attributions[i_KFold, :].sum()}")
    print(
        f"Sum of interactions for sample {i_KFold}: {interactions[i_KFold, :].sum(axis=-1).sum()}")

    fig, axs = plt.subplots(1, 4, figsize=(
        16, 4), sharex=True, constrained_layout=True)

    # Plot attributions, Shapley values, and diagonal interactions
    axs[0].barh(compo_column, attributions[i_KFold, :], color='steelblue')
    axs[1].barh(compo_column, shap_values[i_KFold, :], color='steelblue')
    axs[2].barh(compo_column, np.diag(
        interactions[i_KFold, :, :]), color='steelblue')

    axs[0].set_title("Attributions by Janizek")
    axs[1].set_title("Shapley Values")
    axs[2].set_title('Diagonal Interactions')

    # Scatter plot of attributions against summed interactions
    reshaped_attributions = np.reshape(attributions, -1)
    summed_interactions = np.reshape(np.sum(interactions, axis=-1), -1)

    axs[3].scatter(reshaped_attributions, summed_interactions)
    axs[3].set(
        xlim=[reshaped_attributions.min(), reshaped_attributions.max()],
        ylim=[reshaped_attributions.min(), reshaped_attributions.max()],
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

    plt.show()


def plot_interactions_heatmap(interactions_values, sample_indices, col_labels,
                              cmap, vmin, vmax):
    """
    Plot heatmaps for interaction values, highlighting non-zero interactions.

    Parameters:
    - interactions_values (array): Array containing interaction values.
    - sample_indices (list): List of indices to generate heatmaps for.
    - col_labels (list): Labels for heatmap columns and rows.
    - vmin (float): Minimum value for heatmap color scale.
    - vmax (float): Maximum value for heatmap color scale.

    Returns:
    - None: Displays the heatmap.
    """

    def get_upper_triangle_mask(matrix):
        """Return a mask for the upper triangle of a matrix, excluding the diagonal."""
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        np.fill_diagonal(mask, True)
        return mask

    fig, axs = plt.subplots(1, len(sample_indices), figsize=(
        4 * len(sample_indices), 4), constrained_layout=True)

    for idx, sample_index in enumerate(sample_indices):
        interactions_sample = interactions_values[sample_index, :, :]

        # Filter non-zero rows and columns
        non_zero_rows = np.any(interactions_sample != 0, axis=1)
        non_zero_cols = np.any(interactions_sample != 0, axis=0)
        filtered_interactions = interactions_sample[non_zero_rows][:, non_zero_cols]
        filtered_labels = np.array(col_labels)[non_zero_rows]

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
        sns.heatmap(filtered_interactions, ax=ax, mask=mask, annot=annotations, fmt="",
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    xticklabels=filtered_labels, yticklabels=filtered_labels,
                    cbar=cbar)

        ax.set_aspect('equal')
        ax.set_title(f"Correlation matrix - Sample {sample_index}")

        # Fine-tune visualization
        for t in ax.texts:
            t.set_size(10)
        ax.tick_params(axis='x', labelrotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    plt.show()


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
        """Prepare and normalize input data for prediction."""
        if V_list[i].size != 0:
            X_normalized = self.scaler_compo.transform(X_list[i])
            V_normalized = self.scaler_specific.transform(V_list[i])
            return np.concatenate([X_normalized, V_normalized], axis=1)
        else:
            return self.scaler_compo.transform(X_list[i])

    def _define_predict_norm_function(self, model, input_data, Y_list, i):
        """Define the normalized prediction function for the model."""
        if Y_list[i].size != 0:
            Y_normalized = self.scaler_testing.transform(Y_list[i])
            return lambda: model.predict([input_data, Y_normalized], verbose=0)
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

    def _shap_for_one_fold(self,
                           X_base_list, Y_base_list, V_base_list,
                           X_shap_list, Y_shap_list, V_shap_list, i):
        """Compute SHAP values for a specific fold."""
        model = self._load_model(i)
        input_base_data = self._prepare_input_data(X_base_list, V_base_list, i)
        input_shap_data = self._prepare_input_data(X_shap_list, V_shap_list, i)

        # ===== create an explainer based on base_data =====
        if Y_base_list[i].size != 0:
            Y_base_normalized = self.scaler_testing.transform(Y_base_list[i])
            explainer = shap.KernelExplainer(
                model, [input_base_data, Y_base_normalized])
        else:
            explainer = shap.KernelExplainer(model, input_base_data)

        # Calculate shap values for one fold
        if Y_shap_list[i].size != 0:
            Y_shap_normalized = self.scaler_testing.transform(Y_shap_list[i])
            shap_values_oneFold = explainer.shap_values(
                [input_shap_data, Y_shap_normalized])
        else:
            shap_values_oneFold = explainer.shap_values(input_shap_data)

        # when model contains 2 inputs, the shap_for_one_fold provides a list of 2 shap arrays
        # so I will concatenate the 2 shap arrays into 1
        if Y_shap_list[i].size != 0:
            # print(len(shap_values_oneFold[0]))
            return np.concatenate([shap_values_oneFold[0][0], shap_values_oneFold[0][1]], axis=1)
        else:
            # print('SHAP shape : ', shap_values_oneFold[0].shape)
            return shap_values_oneFold[0]

    def _attributions_interactions_for_one_fold(self,
                                                X_base_list, Y_base_list, V_base_list,
                                                X_shap_list, Y_shap_list, V_shap_list, i):
        """Compute attributions and interactions for a specific fold."""
        model = self._load_model(i)
        input_base_data = self._prepare_input_data(X_base_list, V_base_list, i)
        input_shap_data = self._prepare_input_data(X_shap_list, V_shap_list, i)

        # ===== create an explainer based on base_data =====
        if Y_base_list[i].size != 0:
            pass
        else:
            explainer = PathExplainerTF(model)

        # Calculate feature attribution/interaction (Janizek) values for one fold
        if Y_shap_list[i].size != 0:
            pass
        else:
            np.random.seed(42)
            background_X_indices = np.random.permutation(
                input_base_data.shape[0])
            background_X = input_base_data[background_X_indices].astype(
                np.float32)

            attributions_values_oneFold = explainer.attributions(inputs=input_shap_data.astype(np.float32),
                                                                 baseline=background_X,
                                                                 batch_size=500,
                                                                 num_samples=1000,
                                                                 use_expectation=True,
                                                                 output_indices=0,
                                                                 verbose=True)

            interactions_values_oneFold = explainer.interactions(inputs=input_shap_data.astype(np.float32),
                                                                 baseline=background_X,
                                                                 batch_size=500,
                                                                 num_samples=1000,
                                                                 use_expectation=True,
                                                                 output_indices=0,
                                                                 verbose=True)

        if Y_shap_list[i].size != 0:
            warnings.warn(
                "feature attribution doesn't support multi-input models.")
            attributions_values_oneFold, interactions_values_oneFold = np.empty(
                (0, 0)), np.empty((0, 0))
        else:
            # print('feature attribution shape : ',
            #       attributions_values_oneFold.shape)
            return attributions_values_oneFold, interactions_values_oneFold

    def predict_norm_shap_bootstrap(self,
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

        results_attributions_interactions = Parallel(n_jobs=-1)(delayed(self._attributions_interactions_for_one_fold)(X1_base_list, Y1_base_list, V1_base_list,
                                                                                                                      X1_shap_list, Y1_shap_list, V1_shap_list, i)
                                                                for i in range(self.k_folds * self.n_CVrepeats))

        predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std = zip(
            *results_predict_base)
        predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std = zip(
            *results_predict_shap)
        shap_list = results_shap
        attributions_list, interactions_list = zip(
            *results_attributions_interactions)

        tf.keras.backend.clear_session()

        return (predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std,
                predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std,
                shap_list, attributions_list, interactions_list)


def process_inverse_norm_explainer_data(pred_norm_base_stack, pred_norm_shap_stack,
                                        shap_norm_stack,
                                        attributions_norm_stack,
                                        interactions_norm_stack, scaler_output):
    """
    Transform predictions and Shapley values from normalized to original space.

    Parameters:
    - pred_norm_base_stack (list of np.array): Normalized predictions for base data.
    - pred_norm_shap_stack (list of np.array): Normalized predictions for Shap data.
    - shap_norm_stack (list of np.array): Normalized Shapley values.
    - attributions_norm_stack (list of np.array): Normalized attribution values.
    - interactions_norm_stack (list of np.array): Normalized interaction values.
    - scaler_output (MinMaxScaler): Scaler for the model output.

    Returns:
    - tuple: Mean and standard deviation of predictions, Shapley values, attribution and interactions values in both spaces.
    """

    # Process normalized predictions for base input data (baseline)
    pred_norm_base_conc = np.concatenate(pred_norm_base_stack, axis=0)
    pred_norm_base_mean = np.mean(pred_norm_base_conc, axis=0).reshape(-1)
    pred_norm_base_std = np.std(pred_norm_base_conc, axis=0).reshape(-1)

    # Inverse transform the predictions for the base input data
    pred_base_list = [scaler_output.inverse_transform(
        pred) for pred in pred_norm_base_conc]
    pred_base_mean = np.mean(pred_base_list, axis=0)
    pred_base_std = np.std(pred_base_list, axis=0)

    # Process normalized predictions for Shap input data (target)
    pred_norm_shap_conc = np.concatenate(pred_norm_shap_stack, axis=0)
    pred_norm_shap_mean = np.mean(pred_norm_shap_conc, axis=0).reshape(-1)
    pred_norm_shap_std = np.std(pred_norm_shap_conc, axis=0).reshape(-1)

    # Inverse transform the predictions for the Shap input data
    pred_shap_list = [scaler_output.inverse_transform(
        pred) for pred in pred_norm_shap_conc]
    pred_shap_mean = np.mean(pred_shap_list, axis=0)
    pred_shap_std = np.std(pred_shap_list, axis=0)

    # Compute mean and standard deviation of Shapley values in the normalized space
    shap_norm_mean = np.mean(shap_norm_stack, axis=0)
    shap_norm_std = np.std(shap_norm_stack, axis=0)

    # Compute mean and standard deviation of Attribution values in the normalized space
    attributions_norm_mean = np.mean(attributions_norm_stack, axis=0)
    attributions_norm_std = np.std(attributions_norm_stack, axis=0)

    # Compute mean and standard deviation of Interaction values values in the normalized space
    interactions_norm_mean = np.mean(interactions_norm_stack, axis=0)
    interactions_norm_std = np.std(interactions_norm_stack, axis=0)

    # Sanity check: Ensure that the sum of Shapley values and the mean prediction for the base input data
    # is approximately equal to the mean prediction for the Shap input data in the normalized space
    pred_norm_base_mean_AVG = pred_norm_base_mean.mean()
    epsilon = 1e-4
    assert (np.abs(pred_norm_base_mean_AVG +
            shap_norm_mean.sum(axis=1) - pred_norm_shap_mean) < epsilon).all()

    # Inverse transform the Shapley values
    pred_base_mean_AVG = pred_base_mean.mean()
    diff = pred_shap_mean - pred_base_mean_AVG
    shapley_scaler = np.divide(diff, shap_norm_mean.sum(axis=1).reshape(-1, 1))
    # print(shap_norm_mean.shape, shapley_scaler.shape)

    # I now use the shapley_scaler to inverse normalize both the shap and attributions because I know they are almost the same
    shap_mean = np.multiply(shap_norm_mean, shapley_scaler)
    attributions_mean = np.multiply(attributions_norm_mean, shapley_scaler)

    # I now use the shapley_scaler to inverse normalize the interactions (I always check if it is almost equivalent to the attributions by plotting)
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
        columns = compo_column + C_specific_testing_column + specific_features_sel_column
    if Y1_shap_data.size == 0 and V1_shap_data.size != 0:
        columns = compo_column + specific_features_sel_column

    # Extract and display SHAP values for the selected samples
    sample_shap_values = shap_KFold_mean[sample_index, :]
    # print(sample_shap_values.shape)
    # print(len(columns))
    display(pd.DataFrame(data=sample_shap_values, columns=columns))

    return pred_norm_base_KFold_mean.mean(), sample_shap_values, columns


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

    # Plot SHAP values
    plt.figure(figsize=figsize)
    sns.barplot(x=shap_df['SHAP Value'], y=shap_df['Feature'], palette=palette)
    plt.xlabel('Mean Absolute SHAP Value')
    plt.title(title)
    plt.tight_layout()
    plt.show()


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
    non_zero_rows = np.any(interactions_avg != 0, axis=1)
    non_zero_cols = np.any(interactions_avg != 0, axis=0)
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
