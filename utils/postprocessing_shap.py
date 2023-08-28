import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tabulate import tabulate
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow import keras
import shap
import seaborn as sns


def predict_norm_shap_bootstrap(model_path_bo, model_name,
                                X1_base_list, Y1_base_list, V1_base_list,
                                X1_shap_list, Y1_shap_list, V1_shap_list,
                                k_folds, n_CVrepeats, mc_repeat,
                                scaler_compo, scaler_testing, scaler_specific):

    # Load model from path
    def load_model(i):
        return keras.models.load_model(os.path.join(model_path_bo, model_name.format(i+1)))

    # Normalize and prepare input data
    def prepare_input_base_data(i):
        X1_base_normalized = scaler_compo.transform(X1_base_list[i])
        V1_base_normalized = scaler_specific.transform(V1_base_list[i])
        return np.concatenate([X1_base_normalized, V1_base_normalized], axis=1)

    def prepare_input_shap_data(i):
        X1_shap_normalized = scaler_compo.transform(X1_shap_list[i])
        V1_shap_normalized = scaler_specific.transform(V1_shap_list[i])
        return np.concatenate([X1_shap_normalized, V1_shap_normalized], axis=1)

    # Define prediction function based on input
    def define_predict_norm_function_base(model, input_base_data, i):
        if Y1_base_list:
            Y1_base_normalized = scaler_testing.transform(Y1_base_list[i])
            # the output model predition is what I need for validate SHAP and it will not be inversely transformed
            return lambda: model.predict([input_base_data, Y1_base_normalized], verbose=0)
        else:
            return lambda: model.predict(input_base_data, verbose=0)

    # Define prediction function based on input
    def define_predict_norm_function_shap(model, input_shap_data, i):
        if Y1_shap_list:
            Y1_shap_normalized = scaler_testing.transform(Y1_shap_list[i])
            # the output model predition is what I need for validate SHAP and it will not be inversely transformed
            return lambda: model.predict([input_shap_data, Y1_shap_normalized], verbose=0)
        else:
            return lambda: model.predict(input_shap_data, verbose=0)

    # Monte Carlo Sampling for predictions
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
        if Y1_base_list:
            Y1_base_normalized = scaler_testing.transform(Y1_base_list[i])
            # print(Y1_normalized)
            explainer = shap.DeepExplainer(
                model, [input_base_data, Y1_base_normalized])
            # print(explainer.expected_value[0])
        else:
            explainer = shap.DeepExplainer(model, input_base_data)
        # ==================================================

        # Calculate shap values for one fold
        if Y1_shap_list:
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

    if Y1_shap_list:
        # when model contains 2 inputs, the shap_for_one_fold provides a list of 2 shap arrays
        # so I will concatenate the 2 shap arrays into 1
        results_shap_new = []

        # print(len(results_shap))
        # print(len(results_shap[0]))

        for i in range(len(results_shap)):
            concatenated_array = np.concatenate(
                [results_shap[i][0], results_shap[i][1]], axis=1)
            results_shap_new.append(concatenated_array)

        shap_values_list = results_shap_new
    else:
        shap_values_list = results_shap

    # Clear TensorFlow session to free resources
    tf.keras.backend.clear_session()

    return (predictions_base_list, predictions_base_mc_mean, predictions_base_mc_std,
            predictions_shap_list, predictions_shap_mc_mean, predictions_shap_mc_std, shap_values_list)


def process_predict_norm_shap_data(pred_norm_base_stack, pred_norm_shap_stack, shap_stack):
    pred_norm_base_conc = np.concatenate(pred_norm_base_stack, axis=0)
    pred_norm_base_mean = np.mean(pred_norm_base_conc, axis=0).reshape(-1)
    pred_norm_base_std = np.std(pred_norm_base_conc, axis=0).reshape(-1)

    pred_norm_shap_conc = np.concatenate(pred_norm_shap_stack, axis=0)
    pred_norm_shap_mean = np.mean(pred_norm_shap_conc, axis=0).reshape(-1)
    pred_norm_shap_std = np.std(pred_norm_shap_conc, axis=0).reshape(-1)

    shap_mean = np.mean(shap_stack, axis=0)
    shap_std = np.std(shap_stack, axis=0)

    return (pred_norm_base_mean, pred_norm_base_std, pred_norm_shap_mean, pred_norm_shap_std, shap_mean, shap_std)


def plot_shap_force(X1_shap_data, Y1_shap_data, V1_shap_data,
                    compo_column, C_specific_testing_column, specific_features_sel_column,
                    pred_norm_base_KFold_mean, pred_norm_shap_KFold_mean, shap_KFold_mean,
                    sample_index=1):
    """
    Plots a SHAP force plot for a specific sample from the dataset.
    """
    shap.initjs()

    if len(Y1_shap_data) > 0:
        sample_dataset = np.hstack((X1_shap_data, Y1_shap_data, V1_shap_data))
    else:
        sample_dataset = np.hstack((X1_shap_data, V1_shap_data))

    sample_index = sample_index-1

    # Adjust for 0-indexing
    columns = compo_column + \
        (C_specific_testing_column if len(Y1_shap_data) > 0 else []) + \
        specific_features_sel_column

    # Display sample features
    sample_feature_values = sample_dataset[sample_index, :]
    sample_feature_values_df = pd.DataFrame(
        data=[sample_feature_values], columns=columns)
    display(sample_feature_values_df)

    print('Pred calculated from model:',
          pred_norm_shap_KFold_mean[sample_index])

    # Display SHAP values for the sample
    sample_shap_values = shap_KFold_mean[sample_index, :].reshape(-1)
    sample_shap_values_df = pd.DataFrame(
        data=[sample_shap_values], columns=columns)
    display(sample_shap_values_df)

    # Validate the predicted value for the selected sample
    pred_norm_base_KFold_mean_AVG = pred_norm_base_KFold_mean.mean()
    # print('Validate the pred calculated using SHAP values: ',
    #       pred_norm_base_KFold_mean_AVG)
    predicted_value_sample = pred_norm_base_KFold_mean_AVG + sample_shap_values.sum()
    print('Validate the pred calculated using SHAP values: ', predicted_value_sample)

    # Visualize SHAP force plot
    shap.force_plot(
        pred_norm_base_KFold_mean_AVG,
        sample_shap_values,
        columns,
        link='identity',
        matplotlib=True,
        figsize=(25, 3),
        text_rotation=45,
        contribution_threshold=0.000
    )


def plot_shap_summary(shap_KFold_mean, feature_names,
                      title='Feature Importance', figsize=(5, 5), palette='twilight_shifted_r'):
    """
    Plots the SHAP values in a descending order of importance.
    """
    # shap_KFold_mean_avg = np.abs(shap_KFold_mean.mean(axis=0))
    shap_KFold_mean_avg = shap_KFold_mean.mean(axis=0)

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
