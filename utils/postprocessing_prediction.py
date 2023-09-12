from typing import List
import numpy as np
import pandas as pd
from utils.feature_calculator import FeatureCalculator
from utils.postprocessing_evalutation import predict_bootstrap
import concurrent.futures
from matplotlib import pyplot as plt


def read_new_data_feature_calc(df_new_wt: pd.DataFrame, vars_ele: List[str],
                               specific_features_sel_column: List[str] = ['delta_a', 'Tm', 'sigma_Tm',
                                                                          'Hmix', 'sigma_Hmix', 'sigma_elec_nega',
                                                                          'VEC', 'sigma_VEC'],
                               C_testing: np.array = np.array([25, 1, 7, 0.333])) -> tuple:
    """
    Function to calculate features for new data. 

    Parameters:
    data_path (str): The path where the data file is located.
    KW_name (str): The keyword name used in the file name.
    vars_ele (List[str]): A list of elements to consider in calculations.
    specific_features_sel_column (List[str]): A list of specific features for selection, default is ['delta_a', 'Tm', 'sigma_Tm',
                                                                           'Hmix', 'sigma_Hmix', 'sigma_elec_nega', 
                                                                           'VEC', 'sigma_VEC'].
    C_testing (np.array): The array for corrosion testing conditions, default is np.array([25, 1, 7, 0.333]).

    Returns:
    tuple: A tuple of three elements: composition, features and corrosion testing conditions for the new data.
    """

    # Load the data
    # df_new_wt = pd.read_excel(file_name)

    # Prepare a new dataframe for compositions
    element_columns = ['Fe', 'Cr', 'Ni', 'Mo', 'W', 'N', 'Nb', 'C', 'Si', 'Mn',
                       'Cu', 'Al', 'V', 'Ta', 'Ti', 'Co', 'Mg', 'Y', 'Zr', 'Hf']
    df_compo = pd.DataFrame(columns=element_columns)
    df_compo_new = pd.concat(
        [df_compo, df_new_wt[vars_ele]], axis=0, ignore_index=True)
    df_compo_new = df_compo_new.fillna(0)  # replace NaNs with zero
    compo_new = np.asarray(df_compo_new.values)

    # Calculate features for each composition
    compositions = [(element_columns, element_fraction)
                    for element_fraction in df_compo_new.values]
    feature_calculator = FeatureCalculator(compositions)
    calculated_features = feature_calculator.calculate_features()

    # Organize the calculated features into a DataFrame
    feature_columns = ["a", "delta_a", "Tm", "sigma_Tm", "Hmix", "sigma_Hmix", "ideal_S",
                       "elec_nega", "sigma_elec_nega", "VEC", "sigma_VEC", "bulk_modulus", "sigma_bulk_modulus"]
    df_features = pd.DataFrame(calculated_features, columns=feature_columns)

    # Select specific features
    df_feature_sel = df_features[specific_features_sel_column]
    HC_specific_features_new = np.asarray(df_feature_sel.values)

    # Set the corrosion testing conditions
    C_specific_testing_new = np.ones(
        (len(df_compo_new.index), len(C_testing))) * C_testing
    df_C_specific_new = pd.DataFrame(C_specific_testing_new, columns=[
                                     'TestTemperature_C', 'ChlorideIonConcentration', 'pH', 'ScanRate_mVs'])

    return compo_new, HC_specific_features_new, C_specific_testing_new


def prediction_new_composition(fname, compo, data_path, model_path_bo,
                               NNH_model_name, NNC_model_name, islean, scalers,
                               specific_features_sel_column, C_testing,
                               k_folds, n_CVrepeats, mc_repeat):
    """
    Predict new compositions using provided models.

    Parameters:
    - fname (str): Filename for the composition.
    - compo (str): Composition details.
    - data_path (str): Path to the data directory.
    - model_path_bo (str): Path to the model directory.
    - specific_features_sel_column (list): List of specific features selected.
    - C_testing (array-like): Testing data for C.
    - NNH_model_name (str): Name of the NNH model.
    - NNC_model_name (str): Name of the NNC model.
    - islean (bool): Flag to determine if the model is lean.
    - scalers (object): Scalers for data normalization.
    - k_folds (int): Number of k-folds for cross-validation.
    - n_CVrepeats (int): Number of cross-validation repeats.
    - mc_repeat (int): Number of Monte Carlo repetitions.

    Returns:
    - tuple: A tuple containing composition data, testing data, feature data, and predictions.
    """

    # Load the input data
    file_name_input = f'{data_path}MultiTaskModel_{fname}_wt_pct.xlsx'
    df_new_wt = pd.read_excel(file_name_input)

    # Extract and calculate features from the new data
    compo_new, HC_specific_features, C_specific_testing = read_new_data_feature_calc(
        df_new_wt, compo,
        specific_features_sel_column=specific_features_sel_column,
        C_testing=C_testing)
    H_specific_testing = np.empty((0, 0))

    # Prepare data based on the 'islean' flag
    if not islean:
        compo_data, H_testing_data, C_testing_data, HC_feature_data = compo_new, H_specific_testing, C_specific_testing, HC_specific_features
    else:
        compo_data, H_testing_data, C_testing_data, HC_feature_data = compo_new, np.empty(
            (0, 0)), np.empty((0, 0)), np.empty((0, 0))

    # Predict using the NNH and NNC models
    H1_new_pred_stack, H1_new_pred_mean, H1_new_pred_std, C2_new_pred_stack, C2_new_pred_mean, C2_new_pred_std = predict_bootstrap_NNH_NNC(
        model_path_bo, NNH_model_name, NNC_model_name,
        compo_data, H_testing_data, C_testing_data, HC_feature_data,
        scalers, k_folds, n_CVrepeats, mc_repeat)

    # Aggregate predictions
    H1_new_pred_KFold_mean = np.mean(np.concatenate(
        H1_new_pred_stack, axis=0), axis=0).reshape(-1)
    H1_new_pred_KFold_std = np.std(np.concatenate(
        H1_new_pred_stack, axis=0), axis=0).reshape(-1)
    C2_new_pred_KFold_mean = np.mean(np.concatenate(
        C2_new_pred_stack, axis=0), axis=0).reshape(-1)
    C2_new_pred_KFold_std = np.std(np.concatenate(
        C2_new_pred_stack, axis=0), axis=0).reshape(-1)

    # Update the dataframe with predictions
    df_new_wt['H1_new_pred_KFold_mean'] = H1_new_pred_KFold_mean
    df_new_wt['H1_new_pred_KFold_std'] = H1_new_pred_KFold_std
    df_new_wt['C2_new_pred_KFold_mean'] = C2_new_pred_KFold_mean
    df_new_wt['C2_new_pred_KFold_std'] = C2_new_pred_KFold_std

    # Save the updated dataframe
    file_name_output = f'{model_path_bo}MultiTaskModel_{fname}_wt_pct_ML.xlsx'
    df_new_wt.to_excel(file_name_output, index=False)

    return (compo_data, H_testing_data, C_testing_data, HC_feature_data,
            H1_new_pred_mean, H1_new_pred_std, C2_new_pred_mean, C2_new_pred_std,
            H1_new_pred_KFold_mean, H1_new_pred_KFold_std,
            C2_new_pred_KFold_mean, C2_new_pred_KFold_std)


def predict_bootstrap_NNH_NNC(model_path_bo, NNH_model_name, NNC_model_name,
                              compo_new, H_specific_testing, C_specific_testing, HC_specific_features,
                              scalers,
                              k_folds, n_CVrepeats, mc_repeat):
    """
    Run predictions for NNH and NNC models with bootstrap resampling in parallel.

    Parameters
    ----------
    model_path_bo : str
        The path where the model is located.
    NNH_model_name : str
        The name of the NNH model.
    NNC_model_name : str
        The name of the NNC model.
    compo_new : array_like
        The new composition data to be used for prediction.
    HC_specific_features : array_like
        The specific features for HC prediction.
    C_specific_testing : array_like
        The specific testing for C prediction.
    scalers : dict
        The scalers for each component of the input and output.
    k_folds : int
        The number of folds for cross-validation.
    n_CVrepeats : int
        The number of times the cross-validation is to be repeated.
    mc_repeat : int
        The number of Monte Carlo repetitions to be performed.

    Returns
    -------
    tuple
        A tuple containing prediction stacks, means, and standard deviations for H1 and C2.
    """

    # Create lists of the new data, each repeated as many times as there are CV repetitions
    compo_new_list = [compo_new]*k_folds*n_CVrepeats
    H_specific_testing_list = [H_specific_testing]*k_folds*n_CVrepeats
    C_specific_testing_list = [C_specific_testing]*k_folds*n_CVrepeats

    HC_specific_features_list = [HC_specific_features]*k_folds*n_CVrepeats

    # Run bootstrap predictions in parallel using a thread pool executor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit NNH model prediction task
        future1 = executor.submit(predict_bootstrap, model_path_bo, NNH_model_name,
                                  compo_new_list, H_specific_testing_list, HC_specific_features_list,
                                  k_folds, n_CVrepeats, mc_repeat,
                                  scalers["compo"], scalers["H_specific_testing"], scalers["specific_features"], scalers["H_output"])

        # Submit NNC model prediction task
        future2 = executor.submit(predict_bootstrap, model_path_bo, NNC_model_name,
                                  compo_new_list, C_specific_testing_list, HC_specific_features_list,
                                  k_folds, n_CVrepeats, mc_repeat,
                                  scalers["compo"], scalers["C_specific_testing"], scalers["specific_features"], scalers["C_output"])

    # Extract results from the completed tasks, including prediction stacks, means, and standard deviations
    H1_new_pred_stack, H1_new_pred_mean, H1_new_pred_std = future1.result()
    C2_new_pred_stack, C2_new_pred_mean, C2_new_pred_std = future2.result()

    return H1_new_pred_stack, H1_new_pred_mean, H1_new_pred_std, \
        C2_new_pred_stack, C2_new_pred_mean, C2_new_pred_std


def plot_prediction_uncertainty(model_path_bo, coord_x, coord_y, index_PVD_x_y,
                                pred_mean, pred_std, pred_label, unc_label,
                                title, vmin1, vmax1, vmin2, vmax2):
    """
    This function generates two subplots per iteration: one for predicted values and one for their uncertainties.
    Each point in the scatter plot corresponds to an observation with its color denoting the predicted value or uncertainty.

    Parameters:
    model_path_bo (str): Path to save the plot image.
    coord_x, coord_y (array-like): X and Y coordinates for scatter plot.
    index_PVD_x_y (array-like): Labels for each point to be annotated on the scatter plot.
    pred_mean, pred_std (array-like): Mean and standard deviation of predictions. 
                                     These arrays should have same first dimension size.
    pred_label, unc_label (str): Labels for color bars of prediction and uncertainty plots.
    title (str): Title for the plots.
    vmin1, vmax1, vmin2, vmax2 (float): Min and max values for color scaling in prediction and uncertainty plots.

    Returns:
    None. A plot is generated and saved as an image in the provided model_path_bo.
    """

    # Creating subplots grid
    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(18, 10), dpi=200)

    # Each iteration creates two subplots: one for predictions and one for uncertainty.
    for i, (ax_pred, ax_unc) in enumerate(zip(axs[::2].flat, axs[1::2].flat)):
        # ----- Subplot 1: Prediction -----
        # Scatter plot where color represents predicted mean.
        cax1 = ax_pred.scatter(
            coord_x, coord_y, c=pred_mean[i], s=400, marker='.', cmap='RdBu_r', vmin=vmin1, vmax=vmax1)

        # Setting labels and titles
        ax_pred.set_xlabel('X')
        ax_pred.set_ylabel('Y')
        ax_pred.set_title(f'Prediction {i+1}', fontsize=8)
        ax_pred.set_aspect('equal', 'box')

        # Annotating each point with its corresponding label
        for i_pvd, txt in enumerate(index_PVD_x_y):
            ax_pred.annotate(
                txt, (coord_x[i_pvd]-2, coord_y[i_pvd]-1), color="grey", alpha=1, fontsize=8)

        # Creating colorbar and setting its label
        cbar1 = fig.colorbar(cax1, ax=ax_pred)
        cbar1.set_label(pred_label, size=8)
        cbar1.ax.tick_params(labelsize=8)

        # ----- Subplot 2: Prediction Uncertainty -----
        # Scatter plot where color represents prediction uncertainty (standard deviation).
        cax2 = ax_unc.scatter(
            coord_x, coord_y, c=pred_std[i], s=400, marker='.', cmap='RdGy_r', vmin=vmin2, vmax=vmax2)

        # Setting labels and titles
        ax_unc.set_xlabel('X')
        ax_unc.set_ylabel('Y')
        ax_unc.set_title(f'Prediction Uncertainty {i+1}', fontsize=8)
        ax_unc.set_aspect('equal', 'box')

        # Annotating each point with its corresponding label
        for i, txt in enumerate(index_PVD_x_y):
            ax_unc.annotate(
                txt, (coord_x[i]-3, coord_y[i]-1.5), color="grey", alpha=1, fontsize=8)

        # Creating colorbar and setting its label
        cbar2 = fig.colorbar(cax2, ax=ax_unc)
        cbar2.set_label(unc_label, size=8)
        cbar2.ax.tick_params(labelsize=8)

    # Setting main title for all plots and adjusting layout
    plt.suptitle(title, fontsize=18)
    fig.tight_layout()

    # Saving figure as .png image
    plt.savefig(model_path_bo + title + '.png', bbox_inches='tight')

    # Displaying the plot
    plt.show()


def plot_prediction_uncertainty_AVG(model_path_bo, coord_x, coord_y, index_PVD_x_y,
                                    H1_new_pred_KFold_mean, H1_new_pred_KFold_std,
                                    C2_new_pred_KFold_mean, C2_new_pred_KFold_std,
                                    title):
    """
    This function plots the average prediction uncertainty for two types of predictions (Hardness and Pitting Potential).

    Parameters:
    model_path_bo (str): Path for saving the figure.
    coord_x, coord_y (ndarray): Coordinates for data points.
    index_PVD_x_y (list): Indices for the data points.
    H1_new_pred_stack, C2_new_pred_stack (list): Lists containing the prediction stacks for Hardness and Pitting Potential respectively.
    KW_name (str): Keyword used in title and filename.
    title (str): Title of the figure.

    Returns:
    None. Displays and saves the generated plot.
    """

    # Set font size for the entire figure
    plt.rcParams.update({'font.size': 8})

    # Initialize the figure with 2x2 grid of subplots
    fig, ax = plt.subplots(2, 2, figsize=(6, 5), dpi=200)

    # Define properties for each subplot
    plot_details = [(H1_new_pred_KFold_mean, H1_new_pred_KFold_std, 'Hardness', ''),
                    (C2_new_pred_KFold_mean, C2_new_pred_KFold_std, 'Pitting potential', '(mV)')]

    # Iterate through each subplot and populate with data and details
    for i, (mean, std, name, unit) in enumerate(plot_details):
        row, col = i // 2, i % 2
        cmap1, cmap2 = plt.get_cmap('RdBu_r'), plt.get_cmap('RdGy_r')

        # Scatter plots for mean and standard deviation
        cax1 = ax[row, col].scatter(
            coord_x, coord_y, c=mean, s=400, marker='.', cmap=cmap1)
        cax2 = ax[row+1, col].scatter(coord_x, coord_y,
                                      c=std, s=400, marker='.', cmap=cmap2)

        # Set titles and aspect ratios
        ax[row, col].set_title(f'{name} ')
        ax[row+1, col].set_title(f'{name}  uncertainty')
        ax[row, col].set_aspect('equal')
        ax[row+1, col].set_aspect('equal')

        # Set x and y labels
        ax[row, col].set_xlabel('X')
        ax[row, col].set_ylabel('Y')
        ax[row+1, col].set_xlabel('X')
        ax[row+1, col].set_ylabel('Y')

        # Annotate the data points
        for i_pvd, txt in enumerate(index_PVD_x_y):
            ax[row, col].annotate(
                txt, (coord_x[i_pvd]-3, coord_y[i_pvd]-1.5), color="grey", alpha=1)
            ax[row+1, col].annotate(txt, (coord_x[i_pvd]-3,
                                    coord_y[i_pvd]-1.5), color="grey", alpha=1)

        # Add colorbars to the plots
        cbar1, cbar2 = fig.colorbar(
            cax1, ax=ax[row, col]), fig.colorbar(cax2, ax=ax[row+1, col])
        cbar1.set_label(f'{name}  {unit}')
        cbar2.set_label(f'{name}  uncertainty {unit}')

    # Set the main title for the figure
    plt.suptitle(title, fontsize=10)

    # Adjust layout for neatness
    fig.tight_layout()

    # Save the figure to the specified path
    plt.savefig(model_path_bo + title, bbox_inches='tight')

    # Display the plot
    plt.show()
