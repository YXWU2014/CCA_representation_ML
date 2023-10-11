import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tabulate import tabulate
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow import keras
import re
import shap
import warnings


def display_saved_models(model_path_bo, NNH_model_name, NNC_model_name,
                         mc_state=True, islean=False, act='relu'):
    """
    This function displays the saved NNH and NNC models in a tabular format.

    Parameters:
    model_path_bo: str
        The path of the directory where the model files are stored.

    The function sorts the models based on their type (NNH or NNC) and their index 
    (as mentioned in the filename after 'RepeatedKFold_'). It then displays them in a table.
    """

    # Get all h5 files from the specified directory
    files = sorted([f for f in os.listdir(
        model_path_bo) if f.endswith(f'_{act}.h5')])

    # Separate NNH and NNC model files
    if mc_state and not islean:

        table_data = [["NNH_model_mc", "NNC_model_mc"]]
        nnh_files = [f for f in files if f.startswith(
            NNH_model_name) and f.endswith(f'_mc_{act}.h5')]
        nnc_files = [f for f in files if f.startswith(
            NNC_model_name) and f.endswith(f'_mc_{act}.h5')]

    elif not mc_state and not islean:

        table_data = [["NNH_model", "NNC_model"]]
        nnh_files = [f for f in files if f.startswith(
            NNH_model_name) and not f.endswith(f'_mc_{act}.h5') and not f.endswith(f'_lean_{act}.h5')]
        nnc_files = [f for f in files if f.startswith(
            NNC_model_name) and not f.endswith(f'_mc_{act}.h5') and not f.endswith(f'_lean_{act}.h5')]

    elif not mc_state and islean:
        table_data = [["NNH_model_lean", "NNC_model_lean"]]
        nnh_files = [f for f in files if f.startswith(
            NNH_model_name) and f.endswith(f'_lean_{act}.h5')]
        nnc_files = [f for f in files if f.startswith(
            NNC_model_name) and f.endswith(f'_lean_{act}.h5')]
    else:
        warnings.warn("error on finding saved models.")

    # print(nnh_files)

    def extract_number_from_filename(filename):
        match = re.search(r'_(\d+)', filename)
        return int(match.group(1)) if match else 0

    nnh_files.sort(key=extract_number_from_filename)
    nnc_files.sort(key=extract_number_from_filename)

    # # Sort model files based on the index present in the filename
    # nnh_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # nnc_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Prepare the table data with model filenames

    for i in range(12):
        nnh_file = nnh_files[i] if i < len(nnh_files) else ""
        nnc_file = nnc_files[i] if i < len(nnc_files) else ""
        table_data.append([nnh_file, nnc_file])

    # Display the model filenames in tabular format
    print()
    print(tabulate(table_data, headers="firstrow"))


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
        if V1_list[i].size != 0:
            X1_normalized = scaler_compo.transform(X1_list[i])
            V1_normalized = scaler_specific.transform(V1_list[i])
            return np.concatenate([X1_normalized, V1_normalized], axis=1)
        else:
            X1_normalized = scaler_compo.transform(X1_list[i])
            return X1_normalized

    # Define prediction function based on input
    def define_predict_function(model, input_data, i):
        if Y1_list[i].size != 0:
            Y1_normalized = scaler_testing.transform(Y1_list[i])
            return lambda: scaler_output.inverse_transform(model.predict([input_data, Y1_normalized], verbose=0))
        else:
            return lambda: scaler_output.inverse_transform(model.predict(input_data, verbose=0))

    # Monte Carlo Sampling for predictions
    @tf.autograph.experimental.do_not_convert
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
        ax[row_idx, col_idx].set_xlabel('True values in the testing datasets')
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
         'hardness model', 'steelblue'),
        ((HC_list[1], C2_pred_X2_KFold_mean, C2_pred_X2_KFold_std),
         'corrosion model', 'firebrick')
    ]

    # Create each subplot
    for i, (data, label, color) in enumerate(data_labels_colors):

        min, max = lims[i]
        ticks = np.arange(min, max, 200)

        ax[i].set_xticks(ticks)
        ax[i].set_yticks(ticks)

        # ax[i].set(xlim=lims[i], ylim=lims[i], aspect='equal', box_aspect=1,
        #           xlabel='True values in the testing datasets', ylabel='Predictions',
        #           title=f'{label} - r2={r2_score(data[0], data[1]):.2f}')
        ax[i].set(xlim=lims[i], ylim=lims[i], aspect='equal', box_aspect=1,
                  xlabel='True values in the testing datasets', ylabel='Predictions',
                  title=r'{} - $R^2={:.2f}$'.format(label, r2_score(data[0], data[1])))
        ax[i].plot(lims[i], lims[i], color='grey')
        ax[i].scatter(*data[:2], label=label, color=color, alpha=0.5)
        ax[i].errorbar(x=data[0], y=data[1], yerr=data[2],
                       fmt='none', ecolor=color, capsize=3, alpha=0.3)
        ax[i].legend(loc=4, prop={'size': 8})
        ax[i].grid(alpha=0.5, linewidth=0.5)

    # Adjust spacing and save the figure
    fig.tight_layout()
    plt.savefig(os.path.join(
        model_path_bo, 'NN_full_RepeatedKFold_True_Prediction_fulldata.pdf'), bbox_inches='tight')
    plt.show()
