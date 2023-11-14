import numpy as np


# def create_empty_datasets(datasets_to_empty, all_datasets):
#     # Ensure the input is in the correct format
#     return {key: np.empty((np.array(value).shape[0], 0)) if key in datasets_to_empty else value for key, value in all_datasets.items()}


def create_empty_arrays(arr_list):
    return [np.empty((arr.shape[0], 0)) for arr in arr_list]


def evaluate_model_with_empty_datasets(model, datasets_to_empty, all_datasets, k_folds, n_CVrepeats):
    datasets = create_empty_datasets(datasets_to_empty, all_datasets)

    return model.evaluate_NN_full_model(
        datasets['X1_train'], datasets['X1_test'], datasets['Y1_train'], datasets['Y1_test'],
        datasets['V1_train'], datasets['V1_test'], datasets['H1_train'], datasets['H1_test'],
        datasets['X2_train'], datasets['X2_test'], datasets['Z2_train'], datasets['Z2_test'],
        datasets['W2_train'], datasets['W2_test'], datasets['C2_train'], datasets['C2_test'],
        k_folds, n_CVrepeats
    )

# Example of how you would call evaluate_model_with_empty_datasets from another file where your data variables are defined


def evaluate_model(model, all_datasets, k_folds, n_CVrepeats):
    return evaluate_model_with_empty_datasets(model, [], all_datasets, k_folds, n_CVrepeats)


def evaluate_model_compo_testing(model, all_datasets, k_folds, n_CVrepeats):
    datasets_to_empty = ['V1_train', 'V1_test',
                         'W2_train', 'W2_test']
    return evaluate_model_with_empty_datasets(model, datasets_to_empty, all_datasets, k_folds, n_CVrepeats)


def evaluate_model_compoOnly(model, all_datasets, k_folds, n_CVrepeats):
    datasets_to_empty = ['Y1_train', 'Y1_test', 'V1_train', 'V1_test',
                         'Z2_train', 'Z2_test', 'W2_train', 'W2_test']
    return evaluate_model_with_empty_datasets(model, datasets_to_empty, all_datasets, k_folds, n_CVrepeats)


def evaluate_model_compo_features(model, all_datasets, k_folds, n_CVrepeats):
    datasets_to_empty = ['Y1_train', 'Y1_test',
                         'Z2_train', 'Z2_test']
    return evaluate_model_with_empty_datasets(model, datasets_to_empty, all_datasets, k_folds, n_CVrepeats)

 # # Function to create empty arrays for datasets
    # def create_empty_arrays(arr_list):
    #     return [np.empty((arr.shape[0], 0)) for arr in arr_list]

    # # Helper function to evaluate the model
    # def evaluate_model(model):
    #     return model.evaluate_NN_full_model(
    #         X1_train_norm_KFold, X1_test_norm_KFold, Y1_train_norm_KFold, Y1_test_norm_KFold, V1_train_norm_KFold, V1_test_norm_KFold, H1_train_norm_KFold, H1_test_norm_KFold,
    #         X2_train_norm_KFold, X2_test_norm_KFold, Z2_train_norm_KFold, Z2_test_norm_KFold, W2_train_norm_KFold, W2_test_norm_KFold, C2_train_norm_KFold, C2_test_norm_KFold,
    #         k_folds, n_CVrepeats, scalers)

    # # Helper function to evaluate the model, but replace the testing and feature inputs to be empty

    # def evaluate_model_compo_testing(model):

    #     datasets = [V1_train_norm_KFold, V1_test_norm_KFold,
    #                 W2_train_norm_KFold, W2_test_norm_KFold]

    #     V1_train_empty, V1_test_empty, \
    #         W2_train_empty, W2_test_empty = map(
    #             create_empty_arrays, datasets)

    #     return model.evaluate_NN_full_model(
    #         X1_train_norm_KFold, X1_test_norm_KFold, Y1_train_norm_KFold, Y1_test_norm_KFold, V1_train_empty, V1_test_empty, H1_train_norm_KFold, H1_test_norm_KFold,
    #         X2_train_norm_KFold, X2_test_norm_KFold, Z2_train_norm_KFold, Z2_test_norm_KFold, W2_train_empty, W2_test_empty, C2_train_norm_KFold, C2_test_norm_KFold,
    #         k_folds, n_CVrepeats, scalers)

    # # Helper function to evaluate the model, but replace the testing and feature inputs to be empty
    # def evaluate_model_compoOnly(model):

    #     datasets = [Y1_train_norm_KFold, Y1_test_norm_KFold, V1_train_norm_KFold, V1_test_norm_KFold,
    #                 Z2_train_norm_KFold, Z2_test_norm_KFold, W2_train_norm_KFold, W2_test_norm_KFold]

    #     Y1_train_empty, Y1_test_empty, V1_train_empty, V1_test_empty, \
    #         Z2_train_empty, Z2_test_empty, W2_train_empty, W2_test_empty = map(
    #             create_empty_arrays, datasets)

    #     return model.evaluate_NN_full_model(
    #         X1_train_norm_KFold, X1_test_norm_KFold, Y1_train_empty, Y1_test_empty, V1_train_empty, V1_test_empty, H1_train_norm_KFold, H1_test_norm_KFold,
    #         X2_train_norm_KFold, X2_test_norm_KFold, Z2_train_empty, Z2_test_empty, W2_train_empty, W2_test_empty, C2_train_norm_KFold, C2_test_norm_KFold,
    #         k_folds, n_CVrepeats, scalers)

    # # Helper function to evaluate the model, but replace the testing and feature inputs to be empty

    # def evaluate_model_compo_features(model):

    #     datasets = [Y1_train_norm_KFold, Y1_test_norm_KFold,
    #                 Z2_train_norm_KFold, Z2_test_norm_KFold]

    #     Y1_train_empty, Y1_test_empty, \
    #         Z2_train_empty, Z2_test_empty = map(
    #             create_empty_arrays, datasets)

    #     return model.evaluate_NN_full_model(
    #         X1_train_norm_KFold, X1_test_norm_KFold, Y1_train_empty, Y1_test_empty, V1_train_norm_KFold, V1_test_norm_KFold, H1_train_norm_KFold, H1_test_norm_KFold,
    #         X2_train_norm_KFold, X2_test_norm_KFold, Z2_train_empty, Z2_test_empty, W2_train_norm_KFold, W2_test_norm_KFold, C2_train_norm_KFold, C2_test_norm_KFold,
    #         k_folds, n_CVrepeats, scalers)

    # ---------- model training in parallel ----------
    # Using ProcessPoolExecutor to run in parallel
    # futures_dict = {}
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures_dict['mc_shared'] = executor.submit(
    #         evaluate_model, mt_nn_BObest_mc_shared)
    #     futures_dict['mc_separate'] = executor.submit(
    #         evaluate_model, mt_nn_BObest_mc_separate)
    #     futures_dict['Nomc_shared'] = executor.submit(
    #         evaluate_model, mt_nn_BObest_Nomc_shared)
    #     futures_dict['Nomc_separate'] = executor.submit(
    #         evaluate_model, mt_nn_BObest_Nomc_separate)
    #     futures_dict['Nomc_NoEng'] = executor.submit(
    #         evaluate_model_compo_testing, mt_nn_BObest_Nomc_NoEng)
    #     futures_dict['compo_XAI'] = executor.submit(
    #         evaluate_model_compoOnly, mt_nn_BObest_compo_XAI)
    #     futures_dict['compo_features_XAI'] = executor.submit(
    #         evaluate_model_compo_features, mt_nn_BObest_compo_features_XAI)
