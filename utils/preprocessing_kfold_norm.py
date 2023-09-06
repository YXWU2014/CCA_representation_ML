from sklearn.model_selection import RepeatedKFold
from matplotlib import cm
import matplotlib.pyplot as plt


def kfold_split_and_normalize(data_list, scalers_list, n_splits=2, n_repeats=6, random_state=None):
    """
    Perform k-fold cross-validation on the given data sets and apply normalization using the provided scalers.

    Parameters
    ----------
    data_list : list of ndarray
        List of data sets to be split. Each data set is expected to be an ndarray.
    scalers_list : list of sklearn.preprocessing scaler
        List of scalers corresponding to the data sets in data_list. Each scaler is used to normalize the corresponding data set.
    n_splits : int, default=2
        Number of folds for cross-validation.
    n_repeats : int, default=6
        Number of times cross-validator needs to be repeated.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the cv splitter.

    Returns
    -------
    Tuple of lists
        The train/test split data for each data set in data_list and their normalized versions.
    """

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=random_state)

    train_KFold_list, test_KFold_list = [], []
    train_norm_KFold_list, test_norm_KFold_list = [], []

    for data, scaler in zip(data_list, scalers_list):
        # temporary lists to store split data and normalized data for each data set
        temp_train_KFold, temp_test_KFold, temp_train_norm_KFold, temp_test_norm_KFold = [], [], [], []

        # split is based on the first data set
        for _, (train_ix, test_ix) in enumerate(cv.split(data_list[0])):

            # prepare that Y1 can be empty np array
            if data.size == 0:  # Check if the array is empty
                temp_train_KFold.append(data)
                temp_test_KFold.append(data)
                temp_train_norm_KFold.append(data)
                temp_test_norm_KFold.append(data)
                continue  # Skip the rest of the loop for this empty array

            # split data set
            train_array, test_array = data[train_ix], data[test_ix]

            # normalize split data
            train_norm, test_norm = scaler.transform(
                train_array), scaler.transform(test_array)

            # store split data and normalized data
            temp_train_KFold.append(train_array)
            temp_test_KFold.append(test_array)
            temp_train_norm_KFold.append(train_norm)
            temp_test_norm_KFold.append(test_norm)

        # add the split and normalized data for each data set to the output lists
        train_KFold_list.append(temp_train_KFold)
        test_KFold_list.append(temp_test_KFold)
        train_norm_KFold_list.append(temp_train_norm_KFold)
        test_norm_KFold_list.append(temp_test_norm_KFold)

    return train_KFold_list, test_KFold_list, train_norm_KFold_list, test_norm_KFold_list


def kfold_with_norm(X, Z, W, C,
                    scaler_compo, scaler_testing, scaler_features, scaler_output,
                    n_splits, n_repeats, random_state):
    """
    Wrapper function for kfold_split_and_normalize. Applies k-fold cross-validation and normalization to the input data sets.

    Returns four lists containing the split and normalized data for X, Z, W, and C respectively.
    """
    return kfold_split_and_normalize([X, Z, W, C],
                                     [scaler_compo, scaler_testing,
                                         scaler_features, scaler_output],
                                     n_splits, n_repeats, random_state)


def plot_hist_kfold_with_norm(train_data, test_data, x_min, x_max, axs_title, n_splits, n_repeats, nrows=3):
    """
    This function plots histograms for each fold of cross-validation in the K-Fold normalization scheme.

    Parameters:
    train_data (list of np.array): List of training data arrays for each feature.
    test_data (list of np.array): List of test data arrays for each feature.
    x_min (list): List of minimum x-values for the plots corresponding to each feature.
    x_max (list): List of maximum x-values for the plots corresponding to each feature.
    axs_title (str): Title for the plot.
    n_splits (int): Number of folds in the K-Fold split.
    n_repeats (int): Number of repetitions in the K-Fold split.
    nrows (int, optional): Number of rows in the subplot grid. Default is 3.
    dataset (str, optional): Identifier for the type of dataset ('H' or 'C'). Default is 'H'.

    Returns:
    None
    """

    # Create subplot grid
    fig, axs = plt.subplots(nrows=nrows, ncols=n_splits *
                            n_repeats, figsize=(30, 4*nrows))

    # Define color maps
    colors_1, colors_2 = cm.get_cmap('Blues', 10), cm.get_cmap('Reds', 10)

    # Select the data to be plotted based on the dataset type

    dataset_plot = [(train_data[0], test_data[0], x_min[0], x_max[0], 'compo'),
                    (train_data[1], test_data[1],
                     x_min[1], x_max[1], 'H/C specific_testing'),
                    (train_data[2], test_data[2],
                     x_min[2], x_max[2], 'H/C specific_features'),
                    (train_data[3], test_data[3], x_min[3], x_max[3], 'output')]

    # Loop over each split and repeat in the K-Fold scheme
    for i in range(n_splits*n_repeats):
        # Loop over each feature to be plotted
        for j, data in enumerate(dataset_plot):
            # Unpack data
            train_data_j, test_data_j, x_min_j, x_max_j, axs_title_j = data

            if train_data_j[0].size == 0:  # Check if the array is empty
                continue

            # Plot histogram for the training data
            axs[j, i].hist(train_data_j[i], bins=10, edgecolor='black',
                           color=colors_1(range(len(train_data_j[i][0]))))
            # Plot histogram for the test data
            axs[j, i].hist(test_data_j[i],  bins=10, edgecolor='black',
                           color=colors_2(range(len(test_data_j[i][0]))))
            # Set title and x-axis limits for the plot
            axs[j, i].set_title(f'Fold {i+1}_' + axs_title_j)
            axs[j, i].set_xlim([x_min_j, x_max_j])

    # Set main title and adjust layout for the entire figure
    fig.suptitle(axs_title, fontsize=24)
    fig.tight_layout()
    plt.show()
