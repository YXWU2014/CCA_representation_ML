import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tabulate import tabulate
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow import keras


def display_saved_models(model_path_bo):
    # list all files in the directory
    files = sorted([f for f in os.listdir(model_path_bo) if f.endswith('.h5')])

    # create a table with the headers "NNH_model" and "NNC_model"
    table = [["NNH_model", "NNC_model"]]

    # loop through all files and add the filenames to the appropriate list
    nnh_files = [f for f in files if f.startswith('NNH_model_RepeatedKFold')]
    nnc_files = [f for f in files if f.startswith('NNC_model_RepeatedKFold')]

    # sort NNH_model files and NNC_model files by their integer suffix
    nnh_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    nnc_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # add the filenames to the table
    for i in range(12):
        nnh_file = nnh_files[i] if i < len(nnh_files) else ""
        nnc_file = nnc_files[i] if i < len(nnc_files) else ""
        table.append([nnh_file, nnc_file])

    # display the table
    print(tabulate(table, headers="firstrow"))


def predict_bootstrap(model_path, model_name,
                      X1_list, Y1_list, V1_list,
                      k_folds, n_CVrepeats, mc_repeat,
                      scaler_compo, scaler_testing, scaler_specific, scaler_output):

    H1_pred_X1_list = []
    H1_pred_X1_mc_mean = []
    H1_pred_X1_mc_std = []

    def predict_one_model(i):

        # loading saved models
        NNH_model_loaded_temp = keras.models.load_model(
            os.path.join(model_path, model_name.format(i+1)))

        # concatenating the X1_list and V1_list and normalisation
        X1_temp_norm = scaler_compo.transform(X1_list[i])
        if len(Y1_list) != 0:  # if testing condition for C is defined
            Y1_temp_norm = scaler_testing.transform(Y1_list[i])
        V1_temp_norm = scaler_specific.transform(V1_list[i])

        X1_V1_temp_norm = np.concatenate([X1_temp_norm, V1_temp_norm], axis=1)

        # make predictions

        if len(Y1_list) != 0:  # if testing condition for C is defined

            def predict_one_sample():
                return scaler_output.inverse_transform(
                    NNH_model_loaded_temp.predict(
                        [X1_V1_temp_norm, Y1_temp_norm], verbose=0)
                )
        elif len(Y1_list) == 0:  # if testing condition for H is NOT defined
            def predict_one_sample():
                return scaler_output.inverse_transform(
                    NNH_model_loaded_temp.predict(X1_V1_temp_norm, verbose=0)
                )

        H1_pred_X1_mc_stack_temp = tf.map_fn(lambda _: predict_one_sample(),
                                             tf.range(mc_repeat),
                                             dtype=tf.float32,
                                             parallel_iterations=mc_repeat)

        H1_pred_X1_mc_mean_temp = np.mean(
            H1_pred_X1_mc_stack_temp, axis=0).reshape((-1,))

        H1_pred_X1_mc_std_temp = np.std(
            H1_pred_X1_mc_stack_temp,  axis=0).reshape((-1,))

        return H1_pred_X1_mc_stack_temp, H1_pred_X1_mc_mean_temp, H1_pred_X1_mc_std_temp

    results = Parallel(n_jobs=-1)(delayed(predict_one_model)(i)
                                  for i in range(k_folds * n_CVrepeats))

    # clear TensorFlow session
    tf.keras.backend.clear_session()

    for mc_stack, mean, std in results:
        H1_pred_X1_list.append(mc_stack)
        H1_pred_X1_mc_mean.append(mean)
        H1_pred_X1_mc_std.append(std)

    return H1_pred_X1_list, H1_pred_X1_mc_mean, H1_pred_X1_mc_std


def plot_test_true_vs_pred(k_folds, n_CVrepeats, test_KFold, test_pred_mean, test_pred_std, lims, label, color, model_path):

    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(18, 7))

    for i in range(k_folds * n_CVrepeats):
        # ----- plot true vs prediction -----
        row_idx = i // 6
        col_idx = i % 6
        ax[row_idx, col_idx].set_xlim(lims)
        ax[row_idx, col_idx].set_ylim(lims)
        ax[row_idx, col_idx].plot(lims, lims, color='grey')
        ax[row_idx, col_idx].scatter(
            test_KFold[i], test_pred_mean[i], label=label, color=color, alpha=0.5)
        ax[row_idx, col_idx].errorbar(x=test_KFold[i], y=test_pred_mean[i], yerr=test_pred_std[i], fmt='none',
                                      ecolor=color, capsize=3, alpha=0.5)
        r = r2_score(test_KFold[i], test_pred_mean[i])
        ax[row_idx, col_idx].text(.05, .7, 'r2={:.2f}'.format(
            r), transform=ax[row_idx, col_idx].transAxes, color=color)
        ax[row_idx, col_idx].set_xlabel('True values in training dataset')
        ax[row_idx, col_idx].set_ylabel('Predictions')
        ax[row_idx, col_idx].set_aspect('equal', 'box')
        ax[row_idx, col_idx].legend(loc=4, prop={'size': 8})
        ax[row_idx, col_idx].grid()

    # adjust spacing and show plot
    fig.tight_layout()
    axs_title = label + '_RepeatedKFold_True_Prediction_testdata'
    fig.suptitle(axs_title, fontsize=18)
    plt.savefig(model_path + axs_title + '.png', bbox_inches='tight')
    plt.show()


def plot_full_true_vs_pred(HC_list, HC_pred_stack_list, model_path_bo, lims):

    # concatenate along the first axis
    H1_pred_X1_conc = np.concatenate(HC_pred_stack_list[0], axis=0)
    H1_pred_X1_KFold_mean = np.mean(H1_pred_X1_conc, axis=0).reshape(-1)
    H1_pred_X1_KFold_std = np.std(H1_pred_X1_conc, axis=0).reshape(-1)

    # concatenate along the first axis
    C2_pred_X2_conc = np.concatenate(HC_pred_stack_list[1], axis=0)
    C2_pred_X2_KFold_mean = np.mean(C2_pred_X2_conc, axis=0).reshape(-1)
    C2_pred_X2_KFold_std = np.std(C2_pred_X2_conc, axis=0).reshape(-1)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    for i, (data, label, color) in enumerate(zip([(HC_list[0], H1_pred_X1_KFold_mean, H1_pred_X1_KFold_std),
                                                  (HC_list[1], C2_pred_X2_KFold_mean, C2_pred_X2_KFold_std)],
                                                 ['NNH_model', 'NNC_model'],
                                                 ['steelblue', 'firebrick'])):
        ax[i].set(xlim=lims[i], ylim=lims[i], aspect='equal', box_aspect=1, xlabel='True values in training dataset',
                  ylabel='Predictions', title=f'{label} - r2={r2_score(data[0], data[1]):.2f}')
        ax[i].plot(lims[i], lims[i], color='grey')
        ax[i].scatter(*data[:2], label=label, color=color, alpha=0.5)
        ax[i].errorbar(x=data[0], y=data[1], yerr=data[2],
                       fmt='none', ecolor=color, capsize=3, alpha=0.3)
        ax[i].legend(loc=4, prop={'size': 8})
        ax[i].grid()

    fig.tight_layout()
    plt.savefig(model_path_bo +
                'NN_full_RepeatedKFold_True_Prediction_fulldata.png', bbox_inches='tight')
    plt.show()
