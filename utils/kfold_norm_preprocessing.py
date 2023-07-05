from sklearn.model_selection import RepeatedKFold


def kfold_split_and_normalize(data_list, scalers_list, n_splits=2, n_repeats=6, random_state=None):
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=random_state)

    train_KFold_list, test_KFold_list = [], []
    train_norm_KFold_list, test_norm_KFold_list = [], []


# first lets unpack the X_list and scalers_list
    for data, scaler in zip(data_list, scalers_list):
        temp_train_KFold = []  # each will temperarily as X or Y or H
        temp_test_KFold = []
        temp_train_norm_KFold = []
        temp_test_norm_KFold = []

        # split based on the first array for all arrays in the X_list(e.g. X, Y, H)
        for _, (train_ix, test_ix) in enumerate(cv.split(data_list[0])):

            train_array, test_array = data[train_ix], data[test_ix]
            train_norm, test_norm = scaler.transform(
                train_array), scaler.transform(test_array)

            temp_train_KFold.append(train_array)  # contains the 6 CV split * 2 repeats
            temp_test_KFold.append(test_array)
            temp_train_norm_KFold.append(train_norm)
            temp_test_norm_KFold.append(test_norm)

        train_KFold_list.append(temp_train_KFold)  # contains X, Y, H
        test_KFold_list.append(temp_test_KFold)
        train_norm_KFold_list.append(temp_train_norm_KFold)
        test_norm_KFold_list.append(temp_test_norm_KFold)

    return train_KFold_list, test_KFold_list, train_norm_KFold_list, test_norm_KFold_list


# when call the kfold_with_norm_H, one expect the return to be
# ([X_trian_KFold, Y_train_KFold, H_train_KFold], [X_test_KFold, Y_test_KFold, H_test_KFold],
# [X_train_norm_KFold, Y_train_norm_KFold, H_train_norm_KFold], [X_test_norm_KFold, Y_test_norm_KFold, H_test_norm_KFold])
def kfold_with_norm_H(X, Y, H,
                      scaler_X, scaler_features, scaler_H,
                      n_splits, n_repeats, random_state):
    return kfold_split_and_normalize([X, Y, H],
                                     [scaler_X, scaler_features, scaler_H],
                                     n_splits, n_repeats, random_state)

# when call the kfold_with_norm_C, one expect the return to be
# ([X_trian_KFold, Z_train_KFold, W_train_KFold, C_train_KFold], [X_test_KFold, Z_test_KFold, W_test_KFold, C_test_KFold],
# [X_train_norm_KFold, Z_train_norm_KFold, W_train_norm_KFold, C_train_norm_KFold], [X_test_norm_KFold, Z_test_norm_KFold, W_test_norm_KFold, C_test_norm_KFold])


def kfold_with_norm_C(X, Z, W, C, scaler_X, scaler_testing, scaler_features, scaler_C, n_splits, n_repeats, random_state):
    return kfold_split_and_normalize([X, Z, W, C], [scaler_X, scaler_testing, scaler_features, scaler_C], n_splits, n_repeats, random_state)
