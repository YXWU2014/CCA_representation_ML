import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tabulate import tabulate
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow import keras
import shap


def display_saved_models(model_path_bo, mc_state=True):
    """
    This function displays the saved NNH and NNC models in a tabular format.

    Parameters:
    model_path_bo: str
        The path of the directory where the model files are stored.

    The function sorts the models based on their type (NNH or NNC) and their index 
    (as mentioned in the filename after 'RepeatedKFold_'). It then displays them in a table.
    """

    # Get all h5 files from the specified directory
    files = sorted([f for f in os.listdir(model_path_bo) if f.endswith('.h5')])

    # Separate NNH and NNC model files
    if mc_state:
        nnh_files = [f for f in files if f.startswith(
            'NNH_model_mc_RepeatedKFold')]
        nnc_files = [f for f in files if f.startswith(
            'NNC_model_mc_RepeatedKFold')]
    else:
        nnh_files = [f for f in files if f.startswith(
            'NNH_model_RepeatedKFold')]
        nnc_files = [f for f in files if f.startswith(
            'NNC_model_RepeatedKFold')]

    # Sort model files based on the index present in the filename
    nnh_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    nnc_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Prepare the table data with model filenames
    if mc_state:
        table_data = [["NNH_model_mc", "NNC_model_mc"]]
    else:
        table_data = [["NNH_model", "NNC_model"]]

    for i in range(12):
        nnh_file = nnh_files[i] if i < len(nnh_files) else ""
        nnc_file = nnc_files[i] if i < len(nnc_files) else ""
        table_data.append([nnh_file, nnc_file])

    # Display the model filenames in tabular format
    print()
    print(tabulate(table_data, headers="firstrow"))


# def predict_bootstrap(model_path_bo, model_name,
#                       X1_list, Y1_list, V1_list,
#                       k_folds, n_CVrepeats, mc_repeat,
#                       scaler_compo, scaler_testing, scaler_specific, scaler_output):
#     """
#     Perform bootstrap predictions on a given model with specified parameters.

#     Parameters:
#     model_path_bo: str
#         The path of the directory where the model file is stored.
#     model_name: str
#         The filename of the model.
#     X1_list, Y1_list, V1_list: list of np.array
#         List of test data arrays.
#     k_folds: int
#         The number of folds for k-fold cross-validation.
#     n_CVrepeats: int
#         The number of repeats for k-fold cross-validation.
#     mc_repeat: int
#         The number of Monte Carlo simulations to perform.
#     scaler_compo, scaler_testing, scaler_specific, scaler_output: sklearn.preprocessing Scalers
#         Scalers for normalizing input data and inverse transforming output data.

#     Returns:
#     H1_pred_X1_list, H1_pred_X1_mc_mean, H1_pred_X1_mc_std: lists
#         Lists of predictions, means, and standard deviations of predictions.
#     """
#     # Initialize lists for storing prediction results
#     H1_pred_X1_list = []
#     H1_pred_X1_mc_mean = []
#     H1_pred_X1_mc_std = []

#     # Function to predict on one fold of the cross-validated model
#     def predict_one_model(i):

#         # Load the ith model
#         NNH_model_loaded_temp = keras.models.load_model(
#             os.path.join(model_path_bo, model_name.format(i+1)))

#         # Normalize input data using the provided scalers
#         X1_temp_norm = scaler_compo.transform(X1_list[i])
#         V1_temp_norm = scaler_specific.transform(V1_list[i])

#         # Concatenate normalized X1 and V1 data
#         X1_V1_temp_norm = np.concatenate([X1_temp_norm, V1_temp_norm], axis=1)

#         # Check if testing condition for C is defined
#         if len(Y1_list) != 0:
#             Y1_temp_norm = scaler_testing.transform(Y1_list[i])

#             def predict_one_sample(): return scaler_output.inverse_transform(
#                 NNH_model_loaded_temp.predict(
#                     [X1_V1_temp_norm, Y1_temp_norm], verbose=0)
#             )
#         else:  # if testing condition for H is NOT defined
#             def predict_one_sample(): return scaler_output.inverse_transform(
#                 NNH_model_loaded_temp.predict(X1_V1_temp_norm, verbose=0)
#             )

#         # Perform Monte Carlo Sampling for predictions
#         H1_pred_X1_mc_stack_temp = tf.map_fn(lambda _: predict_one_sample(),
#                                              tf.range(mc_repeat),
#                                              dtype=tf.float32,
#                                              parallel_iterations=mc_repeat)

#         # Compute mean and standard deviation of the predictions
#         H1_pred_X1_mc_mean_temp = np.mean(
#             H1_pred_X1_mc_stack_temp, axis=0).reshape((-1,))
#         H1_pred_X1_mc_std_temp = np.std(
#             H1_pred_X1_mc_stack_temp,  axis=0).reshape((-1,))

#         return H1_pred_X1_mc_stack_temp, H1_pred_X1_mc_mean_temp, H1_pred_X1_mc_std_temp

#     # Perform parallel prediction on each fold of the cross-validated model
#     results = Parallel(n_jobs=-1)(delayed(predict_one_model)(i)
#                                   for i in range(k_folds * n_CVrepeats))

#     # Clear TensorFlow session to free up resources
#     tf.keras.backend.clear_session()

#     # Extract and return results
#     for mc_stack, mean, std in results:
#         H1_pred_X1_list.append(mc_stack)
#         H1_pred_X1_mc_mean.append(mean)
#         H1_pred_X1_mc_std.append(std)

#     return H1_pred_X1_list, H1_pred_X1_mc_mean, H1_pred_X1_mc_std


def predict_bootstrap(model_path_bo, model_name,
                      X1_list, Y1_list, V1_list,
                      k_folds, n_CVrepeats, mc_repeat,
                      scaler_compo, scaler_testing, scaler_specific, scaler_output):
    """
    Perform bootstrap predictions on a given model with specified parameters.

    Parameters:
    model_path_bo: str
        The path of the directory where the model file is stored.
    model_name: str
        The filename of the model.
    X1_list, Y1_list, V1_list: list of np.array
        List of model input data arrays.
    k_folds: int
        The number of folds for k-fold cross-validation.
    n_CVrepeats: int
        The number of repeats for k-fold cross-validation.
    mc_repeat: int
        The number of Monte Carlo simulations to perform.
    scaler_compo, scaler_testing, scaler_specific, scaler_output: sklearn.preprocessing Scalers
        Scalers for normalizing input data and inverse transforming output data.

    Returns:
    predictions_list, predictions_mc_mean, predictions_mc_std: lists
        Lists of predictions, means, and standard deviations of predictions.
        - predictions_list: tuple for 12 RepeatedKFold, each element's shape (mc_repeat, 680, 1)
        - predictions_mc_mean: tuple for 12 RepeatedKFold, each element's shape (680,) averaged over mc repeats
        - predictions_mc_std: tuple for 12 RepeatedKFold, each element's shape (680,) averaged over mc repeats   
    """

    # Load model from path
    def load_model(i):
        return keras.models.load_model(os.path.join(model_path_bo, model_name.format(i+1)))

    # Normalize and prepare input data
    def prepare_input_data(i):
        X1_normalized = scaler_compo.transform(X1_list[i])
        V1_normalized = scaler_specific.transform(V1_list[i])
        return np.concatenate([X1_normalized, V1_normalized], axis=1)

    # Define prediction function based on input
    def define_predict_function(model, input_data, i):
        if Y1_list:
            Y1_normalized = scaler_testing.transform(Y1_list[i])
            return lambda: scaler_output.inverse_transform(model.predict([input_data, Y1_normalized], verbose=0))
        else:
            return lambda: scaler_output.inverse_transform(model.predict(input_data, verbose=0))

    # Monte Carlo Sampling for predictions
    def predict_monte_carlo_sampling(predict_func):
        predictions = tf.map_fn(lambda _: predict_func(),
                                tf.range(mc_repeat),
                                dtype=tf.float32,
                                parallel_iterations=mc_repeat)
        return predictions.numpy(), predictions.numpy().mean(axis=0).reshape((-1,)), predictions.numpy().std(axis=0).reshape((-1,))

    # Prediction on one fold of the cross-validated model
    def predict_for_one_fold(i):
        model = load_model(i)
        input_data = prepare_input_data(i)
        predict_func = define_predict_function(model, input_data, i)
        return predict_monte_carlo_sampling(predict_func)

    # Perform parallel prediction on each fold
    results = Parallel(n_jobs=-1)(delayed(predict_for_one_fold)(i)
                                  for i in range(k_folds * n_CVrepeats))

    # Extract results
    predictions_list, predictions_mc_mean, predictions_mc_std = zip(*results)

    # Clear TensorFlow session to free resources
    tf.keras.backend.clear_session()

    return predictions_list, predictions_mc_mean, predictions_mc_std


def predict_norm_shap_bootstrap(model_path_bo, model_name,
                                X1_list, Y1_list, V1_list,
                                k_folds, n_CVrepeats, mc_repeat,
                                scaler_compo, scaler_testing, scaler_specific):

    # Load model from path
    def load_model(i):
        return keras.models.load_model(os.path.join(model_path_bo, model_name.format(i+1)))

    # Normalize and prepare input data
    def prepare_input_data(i):
        X1_normalized = scaler_compo.transform(X1_list[i])
        V1_normalized = scaler_specific.transform(V1_list[i])
        return np.concatenate([X1_normalized, V1_normalized], axis=1)

    # Define prediction function based on input
    def define_predict_norm_function(model, input_data, i):
        if Y1_list:
            Y1_normalized = scaler_testing.transform(Y1_list[i])
            # the output model predition is what I need for validate SHAP and it will not be inversely transformed
            return lambda: model.predict([input_data, Y1_normalized], verbose=0)
        else:
            return lambda: model.predict(input_data, verbose=0)

    # Monte Carlo Sampling for predictions
    def predict_monte_carlo_sampling(predict_func):
        predictions = tf.map_fn(lambda _: predict_func(),
                                tf.range(mc_repeat),
                                dtype=tf.float32,
                                parallel_iterations=mc_repeat)
        # print(len(predictions.numpy()[0]))
        return predictions.numpy(), predictions.numpy().mean(axis=0).reshape((-1,)), predictions.numpy().std(axis=0).reshape((-1,))

    # Prediction on one fold of the cross-validated model
    def predict_for_one_fold(i):
        model = load_model(i)
        input_data = prepare_input_data(i)
        predict_func = define_predict_norm_function(model, input_data, i)
        return predict_monte_carlo_sampling(predict_func)

    # SHAP calculation on one fold of the cross-validated model
    def shap_for_one_fold(i):
        model = load_model(i)
        # model.summary()
        input_data = prepare_input_data(i)

        # create an explainer
        if Y1_list:
            Y1_normalized = scaler_testing.transform(Y1_list[i])
            explainer = shap.DeepExplainer(model, [input_data, Y1_normalized])
        else:
            explainer = shap.DeepExplainer(model, input_data)

        # Calculate shap values for one fold
        if Y1_list:
            shap_values_oneFold = explainer.shap_values(
                [input_data, Y1_normalized])
            # print("SHAP value dim:", shap_values_oneFold[0].shape)
        else:
            shap_values_oneFold = explainer.shap_values(input_data)
            # print("SHAP value dim:", shap_values_oneFold[0].shape)

        return shap_values_oneFold[0]

    # Perform parallel prediction on each fold
    results_predict = Parallel(n_jobs=-1)(delayed(predict_for_one_fold)(i)
                                          for i in range(k_folds * n_CVrepeats))

    results_shap = Parallel(n_jobs=-1)(delayed(shap_for_one_fold)(i)
                                       for i in range(k_folds * n_CVrepeats))
    # Extract results
    predictions_list, predictions_mc_mean, predictions_mc_std = zip(
        *results_predict)
    shap_values_list = results_shap

    # Clear TensorFlow session to free resources
    tf.keras.backend.clear_session()

    return predictions_list, predictions_mc_mean, predictions_mc_std, shap_values_list


def plot_test_true_vs_pred(k_folds, n_CVrepeats, test_KFold, test_pred_mean, test_pred_std, lims, label, color, model_path_bo):
    """
    Plot true vs predicted values for test data across multiple cross-validation folds.

    Parameters:
    k_folds: int
        The number of folds for k-fold cross-validation.
    n_CVrepeats: int
        The number of repeats for k-fold cross-validation.
    test_KFold: list of np.array
        List of ground truth values for each fold.
    test_pred_mean: list of np.array
        List of mean prediction values for each fold.
    test_pred_std: list of np.array
        List of standard deviation of prediction values for each fold.
    lims: tuple
        Tuple specifying the limits of the plot (xmin, xmax, ymin, ymax).
    label: str
        Label for the data being plotted.
    color: str
        Color for the data points.
    model_path_bo: str
        The path to the directory where the plot will be saved.

    Returns:
    None
    """
    # Initialize a new matplotlib figure
    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(18, 7))

    # Iterate over each fold
    for i in range(k_folds * n_CVrepeats):
        # Determine the row and column index for the subplot
        row_idx = i // 6
        col_idx = i % 6

        # Set the limits of the subplot and draw the line y=x
        ax[row_idx, col_idx].set_xlim(lims)
        ax[row_idx, col_idx].set_ylim(lims)
        ax[row_idx, col_idx].plot(lims, lims, color='grey')

        # Scatter plot of true vs predicted values
        ax[row_idx, col_idx].scatter(
            test_KFold[i], test_pred_mean[i], label=label, color=color, alpha=0.5)

        # Add error bars to the scatter plot
        ax[row_idx, col_idx].errorbar(x=test_KFold[i], y=test_pred_mean[i], yerr=test_pred_std[i], fmt='none',
                                      ecolor=color, capsize=3, alpha=0.5)

        # Compute and add R^2 score to the subplot
        r = r2_score(test_KFold[i], test_pred_mean[i])
        ax[row_idx, col_idx].text(.05, .7, 'r2={:.2f}'.format(
            r), transform=ax[row_idx, col_idx].transAxes, color=color)

        # Set labels and aspect ratio for the subplot
        ax[row_idx, col_idx].set_xlabel('True values in training dataset')
        ax[row_idx, col_idx].set_ylabel('Predictions')
        ax[row_idx, col_idx].set_aspect('equal', 'box')

        # Add a legend to the subplot
        ax[row_idx, col_idx].legend(loc=4, prop={'size': 8})

        # Enable grid for the subplot
        ax[row_idx, col_idx].grid()

    # Adjust spacing and add title
    fig.tight_layout()
    axs_title = label + '_RepeatedKFold_True_Prediction_testdata'
    fig.suptitle(axs_title, fontsize=18)

    # Save the figure and display the plot
    plt.savefig(os.path.join(model_path_bo, axs_title + '.png'),
                bbox_inches='tight')
    plt.show()


def plot_full_true_vs_pred(HC_list, HC_pred_stack_list, model_path_bo, lims):
    """
    Plot true vs predicted values for a model's full output.

    Parameters:
    HC_list: list
        A list of numpy arrays representing the true outputs of the models.
    HC_pred_stack_list: list
        A list of numpy arrays representing the predicted outputs of the models.
    model_path_bo: str
        The path to the directory where the plot will be saved.
    lims: tuple
        Tuple specifying the limits of the plot (xmin, xmax, ymin, ymax).

    Returns:
    None
    """

    # Compute mean and standard deviation of the first model's predictions
    H1_pred_X1_conc = np.concatenate(HC_pred_stack_list[0], axis=0)
    H1_pred_X1_KFold_mean = np.mean(H1_pred_X1_conc, axis=0).reshape(-1)
    H1_pred_X1_KFold_std = np.std(H1_pred_X1_conc, axis=0).reshape(-1)

    # Compute mean and standard deviation of the second model's predictions
    C2_pred_X2_conc = np.concatenate(HC_pred_stack_list[1], axis=0)
    C2_pred_X2_KFold_mean = np.mean(C2_pred_X2_conc, axis=0).reshape(-1)
    C2_pred_X2_KFold_std = np.std(C2_pred_X2_conc, axis=0).reshape(-1)

    # Initialize a new matplotlib figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # Data, labels and colors for each subplot
    data_labels_colors = [
        ((HC_list[0], H1_pred_X1_KFold_mean, H1_pred_X1_KFold_std),
         'NNH_model', 'steelblue'),
        ((HC_list[1], C2_pred_X2_KFold_mean,
         C2_pred_X2_KFold_std), 'NNC_model', 'firebrick')
    ]

    # Create each subplot
    for i, (data, label, color) in enumerate(data_labels_colors):
        ax[i].set(xlim=lims[i], ylim=lims[i], aspect='equal', box_aspect=1,
                  xlabel='True values in training dataset', ylabel='Predictions',
                  title=f'{label} - r2={r2_score(data[0], data[1]):.2f}')
        ax[i].plot(lims[i], lims[i], color='grey')
        ax[i].scatter(*data[:2], label=label, color=color, alpha=0.5)
        ax[i].errorbar(x=data[0], y=data[1], yerr=data[2],
                       fmt='none', ecolor=color, capsize=3, alpha=0.3)
        ax[i].legend(loc=4, prop={'size': 8})
        ax[i].grid()

    # Adjust spacing and save the figure
    fig.tight_layout()
    plt.savefig(os.path.join(
        model_path_bo, 'NN_full_RepeatedKFold_True_Prediction_fulldata.png'), bbox_inches='tight')
    plt.show()
