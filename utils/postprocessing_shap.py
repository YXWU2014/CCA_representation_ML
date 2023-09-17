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


def predict_norm_shap_bootstrap(model_path_bo, model_name,
                                X1_base_list, Y1_base_list, V1_base_list,
                                X1_shap_list, Y1_shap_list, V1_shap_list,
                                k_folds, n_CVrepeats, mc_repeat,
                                scaler_compo, scaler_testing, scaler_specific):
    """
    Perform predictions and Shapley value calculations using bootstrapped models.

    Parameters:
    - model_path_bo (str): Path to the directory containing the models.
    - model_name (str): Name format of the models.
    - X1_base_list, Y1_base_list, V1_base_list (list of np.array): Lists of base input data.
    - X1_shap_list, Y1_shap_list, V1_shap_list (list of np.array): Lists of Shap input data.
    - k_folds (int): Number of cross-validation folds.
    - n_CVrepeats (int): Number of cross-validation repetitions.
    - mc_repeat (int): Number of Monte Carlo repetitions.
    - scaler_compo, scaler_testing, scaler_specific (MinMaxScaler): Scalers used for the model's input and output.

    Returns:
    - tuple: Contains lists of predictions and Shapley values for each fold.
    """

    # Load model from path
    def load_model(i):
        return keras.models.load_model(os.path.join(model_path_bo, model_name.format(i+1)))

    # Normalize and prepare input data

    def prepare_input_base_data(i):
        if V1_base_list[i].size != 0:
            X1_base_normalized = scaler_compo.transform(X1_base_list[i])
            V1_base_normalized = scaler_specific.transform(V1_base_list[i])
            return np.concatenate([X1_base_normalized, V1_base_normalized], axis=1)
        else:
            X1_base_normalized = scaler_compo.transform(X1_base_list[i])
            return X1_base_normalized

    def prepare_input_shap_data(i):
        if V1_shap_list[i].size != 0:
            X1_shap_normalized = scaler_compo.transform(X1_shap_list[i])
            V1_shap_normalized = scaler_specific.transform(V1_shap_list[i])
            return np.concatenate([X1_shap_normalized, V1_shap_normalized], axis=1)
        else:
            X1_shap_normalized = scaler_compo.transform(X1_shap_list[i])
            return X1_shap_normalized

    # Define prediction function based on input
    def define_predict_norm_function_base(model, input_base_data, i):
        if Y1_base_list[i].size != 0:
            Y1_base_normalized = scaler_testing.transform(Y1_base_list[i])
            # the output model predition is what I need for validate SHAP and it will not be inversely transformed
            return lambda: model.predict([input_base_data, Y1_base_normalized], verbose=0)
        else:
            return lambda: model.predict(input_base_data, verbose=0)

    # Define prediction function based on input
    def define_predict_norm_function_shap(model, input_shap_data, i):
        if Y1_shap_list[i].size != 0:
            Y1_shap_normalized = scaler_testing.transform(Y1_shap_list[i])
            # the output model predition is what I need for validate SHAP and it will not be inversely transformed
            return lambda: model.predict([input_shap_data, Y1_shap_normalized], verbose=0)
        else:
            return lambda: model.predict(input_shap_data, verbose=0)

    # Monte Carlo Sampling for predictions
    @tf.autograph.experimental.do_not_convert
    def predict_monte_carlo_sampling(predict_func):
        predictions = tf.map_fn(lambda _: predict_func(),
                                tf.range(mc_repeat),
                                dtype=tf.float32,
                                parallel_iterations=mc_repeat)
        # print(len(predictions.numpy()[0]))
        return predictions.numpy(), predictions.numpy().mean(axis=0).reshape((-1,)), predictions.numpy().std(axis=0).reshape((-1,))

    # Prediction on one fold of the cross-validated model
    def predict_for_one_fold_base(i):
        model = load_model(i)
        input_base_data = prepare_input_base_data(i)
        predict_func = define_predict_norm_function_base(
            model, input_base_data, i)
        return predict_monte_carlo_sampling(predict_func)

    def predict_for_one_fold_shap(i):
        model = load_model(i)
        input_shap_data = prepare_input_shap_data(i)
        predict_func = define_predict_norm_function_shap(
            model, input_shap_data, i)
        return predict_monte_carlo_sampling(predict_func)

    # SHAP calculation on one fold of the cross-validated model
    def shap_for_one_fold(i):
        model = load_model(i)
        # model.summary()
        input_base_data = prepare_input_base_data(i)
        input_shap_data = prepare_input_shap_data(i)

        # ===== create an explainer based on base_data =====
        if Y1_base_list[i].size != 0:
            Y1_base_normalized = scaler_testing.transform(Y1_base_list[i])
            # print(Y1_normalized)
            explainer = shap.DeepExplainer(
                model, [input_base_data, Y1_base_normalized])
            # print(explainer.expected_value[0])
        else:
            explainer = shap.DeepExplainer(model, input_base_data)
        # ==================================================

        # Calculate shap values for one fold
        if Y1_shap_list[i].size != 0:
            Y1_shap_normalized = scaler_testing.transform(Y1_shap_list[i])
            shap_values_oneFold = explainer.shap_values(
                [input_shap_data, Y1_shap_normalized])
            # print("SHAP value dim:", shap_values_oneFold[0].shape)
        else:
            shap_values_oneFold = explainer.shap_values(input_shap_data)
            # print("SHAP value dim:", shap_values_oneFold[0].shape)

        return shap_values_oneFold[0]

    # Perform parallel prediction on each fold
    results_predict_base = Parallel(n_jobs=-1)(delayed(predict_for_one_fold_base)(i)
                                               for i in range(k_folds * n_CVrepeats))

    results_predict_shap = Parallel(n_jobs=-1)(delayed(predict_for_one_fold_shap)(i)
                                               for i in range(k_folds * n_CVrepeats))

    results_shap = Parallel(n_jobs=-1)(delayed(shap_for_one_fold)(i)
                                       for i in range(k_folds * n_CVrepeats))

    # Extract results
    predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std = zip(
        *results_predict_base)

    predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std = zip(
        *results_predict_shap)

    if Y1_shap_list[0].size != 0:
        # when model contains 2 inputs, the shap_for_one_fold provides a list of 2 shap arrays
        # so I will concatenate the 2 shap arrays into 1
        results_shap_new = []

        # print(len(results_shap))
        # print(len(results_shap[0]))

        for j in range(len(results_shap)):
            concatenated_array = np.concatenate(
                [results_shap[j][0], results_shap[j][1]], axis=1)
            results_shap_new.append(concatenated_array)

        shap_values_list = results_shap_new
    else:
        shap_values_list = results_shap

    # Clear TensorFlow session to free resources
    tf.keras.backend.clear_session()

    return (predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std,
            predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std,
            shap_values_list)


def process_predict_norm_shap_data(pred_norm_base_stack, pred_norm_shap_stack, shap_norm_stack, scaler_output):
    """
    Process predictions and Shapley values in the normalized space and inverse transform them to the original space.

    Parameters:
    - pred_norm_base_stack (list of np.array): List of normalized predictions for the base input data.
    - pred_norm_shap_stack (list of np.array): List of normalized predictions for the Shap input data.
    - shap_norm_stack (list of np.array): List of normalized Shapley values.
    - scaler_output (MinMaxScaler): Scaler used for the model's output.

    Returns:
    - tuple: Contains mean and standard deviation of predictions and Shapley values in both normalized and original spaces.
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
    shap_mean = np.multiply(shap_norm_mean, shapley_scaler)

    return (pred_norm_base_mean, pred_norm_shap_mean, shap_norm_mean,
            pred_base_mean, pred_shap_mean, shap_mean)


def data_for_shap_force(X1_shap_data, Y1_shap_data, V1_shap_data,
                        compo_column, C_specific_testing_column, specific_features_sel_column,
                        pred_norm_base_KFold_mean, pred_norm_shap_KFold_mean, shap_KFold_mean,
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


def plot_shap_summary(shap_KFold_mean, feature_names,
                      title='Feature Importance', figsize=(5, 5), palette='twilight_shifted_r'):
    """
    Plots the SHAP values in a descending order of importance.
    """
    shap_KFold_mean_avg = np.abs(shap_KFold_mean).mean(axis=0)
    # shap_KFold_mean_avg = shap_KFold_mean.mean(axis=0)

    # Organize the calculated SHAP values and their corresponding feature names into a DataFrame.
    shap_df = pd.DataFrame(
        data={'SHAP Value': shap_KFold_mean_avg, 'Feature': feature_names})

    # Order the DataFrame based on the SHAP values to understand which features have the greatest influence.
    shap_df = shap_df.sort_values(by='SHAP Value', ascending=False)
    shap_df = shap_df[shap_df['SHAP Value'] != 0]

    # Use Seaborn's barplot function to plot the data
    plt.figure(figsize=figsize)
    sns.barplot(x=shap_df['SHAP Value'], y=shap_df['Feature'], palette=palette)

    plt.xlabel('Mean Absolute SHAP Value')
    plt.title(title)

    # Ensure the plot layout is optimized to prevent overlapping or cutting off labels.
    plt.tight_layout()

    # Display the plotted chart.
    plt.show()
