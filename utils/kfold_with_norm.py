from sklearn.model_selection import RepeatedKFold

# function for KFold split and normalisation


def kfold_with_norm_H(X, Y, H,
                      scaler_X, scaler_features, scaler_H,
                      n_splits, n_repeats, random_state):

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=random_state)

    X_train_KFold, X_test_KFold = [], []
    Y_train_KFold, Y_test_KFold = [], []
    H_train_KFold, H_test_KFold = [], []

    X_train_norm_KFold, X_test_norm_KFold = [], []
    Y_train_norm_KFold, Y_test_norm_KFold = [], []
    H_train_norm_KFold, H_test_norm_KFold = [], []

    for i, (train_ix, test_ix) in enumerate(cv.split(X)):

        # train-test split
        X_train, X_test = X[train_ix], X[test_ix]
        Y_train, Y_test = Y[train_ix], Y[test_ix]
        H_train, H_test = H[train_ix], H[test_ix]

        X_train_KFold.append(X_train)
        X_test_KFold.append(X_test)
        Y_train_KFold.append(Y_train)
        Y_test_KFold.append(Y_test)
        H_train_KFold.append(H_train)
        H_test_KFold.append(H_test)

        # MinMaxScaler normalisation
        X_train_norm, X_test_norm = scaler_X.transform(
            X_train), scaler_X.transform(X_test)
        Y_train_norm, Y_test_norm = scaler_features.transform(
            Y_train), scaler_features.transform(Y_test)
        H_train_norm, H_test_norm = scaler_H.transform(H_train.reshape(
            (-1, 1))), scaler_H.transform(H_test.reshape((-1, 1)))

        X_train_norm_KFold.append(X_train_norm)
        X_test_norm_KFold.append(X_test_norm)
        Y_train_norm_KFold.append(Y_train_norm)
        Y_test_norm_KFold.append(Y_test_norm)
        H_train_norm_KFold.append(H_train_norm)
        H_test_norm_KFold.append(H_test_norm)

    return (X_train_KFold, X_test_KFold, Y_train_KFold, Y_test_KFold, H_train_KFold, H_test_KFold,
            X_train_norm_KFold, X_test_norm_KFold, Y_train_norm_KFold, Y_test_norm_KFold, H_train_norm_KFold, H_test_norm_KFold)



# function for KFold split and normalisation
def kfold_with_norm_C(X, Z, W, C,
                      scaler_X, scaler_testing, scaler_features, scaler_C,
                      n_splits, n_repeats, random_state):

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=random_state)

    X_train_KFold, X_test_KFold = [], []
    Z_train_KFold, Z_test_KFold = [], []
    W_train_KFold, W_test_KFold = [], []
    C_train_KFold, C_test_KFold = [], []

    X_train_norm_KFold, X_test_norm_KFold = [], []
    Z_train_norm_KFold, Z_test_norm_KFold = [], []
    W_train_norm_KFold, W_test_norm_KFold = [], []
    C_train_norm_KFold, C_test_norm_KFold = [], []

    for i, (train_ix, test_ix) in enumerate(cv.split(X)):

        # train-test split
        X_train, X_test = X[train_ix], X[test_ix]
        Z_train, Z_test = Z[train_ix], Z[test_ix]
        W_train, W_test = W[train_ix], W[test_ix]
        C_train, C_test = C[train_ix], C[test_ix]

        X_train_KFold.append(X_train)
        X_test_KFold.append(X_test)
        Z_train_KFold.append(Z_train)
        Z_test_KFold.append(Z_test)
        W_train_KFold.append(W_train)
        W_test_KFold.append(W_test)
        C_train_KFold.append(C_train)
        C_test_KFold.append(C_test)

        # MinMaxScaler normalisation
        X_train_norm, X_test_norm = scaler_X.transform(
            X_train), scaler_X.transform(X_test)
        
        Z_train_norm, Z_test_norm = scaler_testing.transform(
            Z_train), scaler_testing.transform(Z_test)
        
        W_train_norm, W_test_norm = scaler_features.transform(
            W_train), scaler_features.transform(W_test)
        
        C_train_norm, C_test_norm = scaler_C.transform(C_train.reshape(
            (-1, 1))), scaler_C.transform(C_test.reshape((-1, 1)))

        X_train_norm_KFold.append(X_train_norm)
        X_test_norm_KFold.append(X_test_norm)
        Z_train_norm_KFold.append(Z_train_norm)
        Z_test_norm_KFold.append(Z_test_norm)
        W_train_norm_KFold.append(W_train_norm)
        W_test_norm_KFold.append(W_test_norm)
        C_train_norm_KFold.append(C_train_norm)
        C_test_norm_KFold.append(C_test_norm)

    return (X_train_KFold, X_test_KFold, Z_train_KFold, Z_test_KFold, W_train_KFold, W_test_KFold, C_train_KFold, C_test_KFold,
            X_train_norm_KFold, X_test_norm_KFold, Z_train_norm_KFold, Z_test_norm_KFold, W_train_norm_KFold, W_test_norm_KFold, 
            C_train_norm_KFold, C_test_norm_KFold)