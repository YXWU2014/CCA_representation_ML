# BO_hyper_objective.py
from utils.multitask_nn import MultiTaskNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class BayesianOptimizationObjective:
    """
    A class to define the objective function for Bayesian Optimization when optimizing 
    hyperparameters for a MultiTask Neural Network.
    """

    def __init__(self, bo_ens_num: int, model_path_bo: str,
                 all_hyper_names: list, search_hyper_names: list):
        """
        Initialize the BayesianOptimizationObjective class.

        Parameters:
        - bo_ens_num: Ensemble number for Bayesian Optimization.
        - model_path_bo: Path where the trained model will be saved.
        - all_hyper_names: List containing names of all hyperparameters.
        - search_hyper_names: List containing names of hyperparameters that will be optimized.
        """
        self.bo_iteration = 0
        self.bo_ens_num = bo_ens_num
        self.model_path_bo = model_path_bo
        self.all_hyper_names = all_hyper_names
        self.search_hyper_names = search_hyper_names
        self.hypertable = pd.DataFrame(columns=['score_r2_HC', 'score_r2_HC_best', 'score_loss_HC',
                                                'score_r2_H', 'score_r2_C', 'score_loss_H', 'score_loss_C'] + self.all_hyper_names)

    def BO_NNS_NNH_NNC_objective(self, search_hyper_params, fixed_hyper_space, search_hyper_names, fixed_hyper_names,
                                 n_initial_points: int, n_iterations: int,
                                 mc_state: bool, act: str,
                                 total_epochs: int,
                                 model_save_flag: bool, model_path_bo: str,
                                 NNH_model_name: str, NNC_model_name: str,
                                 X1_train_norm_KFold: list, X1_test_norm_KFold: list, Y1_train_norm_KFold: list, Y1_test_norm_KFold: list, V1_train_norm_KFold: list, V1_test_norm_KFold: list, H1_train_norm_KFold: list, H1_test_norm_KFold: list,
                                 X2_train_norm_KFold: list, X2_test_norm_KFold: list, Z2_train_norm_KFold: list, Z2_test_norm_KFold: list, W2_train_norm_KFold: list, W2_test_norm_KFold: list, C2_train_norm_KFold: list, C2_test_norm_KFold: list,
                                 k_folds: int, n_CVrepeats: int,
                                 score_r2_HC_list: list, score_loss_HC_list: list,
                                 score_r2_H_list: list, score_r2_C_list: list,
                                 score_loss_H_list: list, score_loss_C_list: list):
        """
        Objective function for Bayesian Optimization: Trains a MultiTaskNN model using given hyperparameters and returns evaluation scores.

        Args:
            search_hyper_params (list): Hyperparameters that are subject to search.
            fixed_hyper_space (list): Fixed hyperparameters not under search.
            search_hyper_names (list): Names of hyperparameters under search.
            fixed_hyper_names (list): Names of fixed hyperparameters.
            n_initial_points (int): Number of initial points.
            n_iterations (int): Number of iterations.
            mc_state (bool): The state of Monte Carlo dropout.
            act (str): The activation function to use.
            total_epochs (int): Total number of epochs for training.
            model_save_flag (bool): Flag to decide if the model should be saved or not.
            model_path_bo (str): Path to save the model.
            X1_train_norm_KFold, X1_test_norm_KFold, etc.: Lists of training and testing data for different tasks.
            k_folds (int): Number of folds for cross validation.
            n_CVrepeats (int): Number of repeats for cross validation.
            score_r2_HC_list, score_loss_HC_list, etc.: Lists to store scores for different tasks.

        Returns:
            tuple: A tuple of mean r2 score and mean loss score across tasks.
        """
        global bo_iteration  # Global variable to keep track of iterations

        # get the search space name and parameters
        search_hyper_params = search_hyper_params[0]
        search_hyper_dict = {key: value for key,
                             value in zip(search_hyper_names, search_hyper_params)}

        # get the fixed space name and parameters
        fixed_hyper_params = [entry['domain'] for entry in fixed_hyper_space]
        fixed_hyper_dict = {key: value for key,
                            value in zip(fixed_hyper_names, fixed_hyper_params)}

        # create the `all_hyper_dict` by merging `search_hyper_dict` and `fixed_hyper_dict`
        all_hyper_dict = {**search_hyper_dict, **fixed_hyper_dict}

        # Convert the categorical encoding of loss function to tensorflow loss function
        loss_encoder = int(all_hyper_dict['loss_encoder'])
        if loss_encoder == 0:
            loss_func = tf.keras.metrics.mean_squared_error
        elif loss_encoder == 1:
            loss_func = tf.keras.metrics.mean_absolute_error
        else:
            raise ValueError(f"Invalid loss function '{loss_encoder}' ")

        # Create a MultiTaskNN model
        mt_nn_bo = MultiTaskNN(NNS_num_nodes=int(all_hyper_dict['NNS_num_nodes']), NNS_num_layers=int(all_hyper_dict['NNS_num_layers']),
                               NNH_num_nodes=int(all_hyper_dict['NNH_num_nodes']), NNH_num_layers=int(all_hyper_dict['NNH_num_layers']),
                               NNC_num_nodes=int(all_hyper_dict['NNC_num_nodes']), NNC_num_layers=int(all_hyper_dict['NNC_num_layers']),
                               mc_state=mc_state, act=act,
                               NNS_dropout=all_hyper_dict['NNS_dropout'], NNH_dropout=all_hyper_dict[
                                   'NNH_NNC_dropout'], NNC_dropout=all_hyper_dict['NNH_NNC_dropout'],
                               loss_func=loss_func,
                               learning_rate_H=all_hyper_dict['learning_rate_H'], learning_rate_C=all_hyper_dict['learning_rate_C'],
                               batch_size_H=int(
                                   all_hyper_dict['batch_size_H']),
                               N_epochs_local=int(all_hyper_dict['N_epochs_local']), total_epochs=total_epochs,
                               model_save_flag=model_save_flag, model_path_bo=model_path_bo,
                               NNH_model_name=NNH_model_name, NNC_model_name=NNC_model_name)

        # Train the model and get scores
        (_, _, _, _,
         score_loss_H,  score_loss_C,
         score_r2_H,    score_r2_C) = mt_nn_bo.evaluate_NN_full_model(X1_train_norm_KFold, X1_test_norm_KFold, Y1_train_norm_KFold, Y1_test_norm_KFold, V1_train_norm_KFold, V1_test_norm_KFold, H1_train_norm_KFold, H1_test_norm_KFold,
                                                                      X2_train_norm_KFold, X2_test_norm_KFold, Z2_train_norm_KFold, Z2_test_norm_KFold, W2_train_norm_KFold, W2_test_norm_KFold, C2_train_norm_KFold, C2_test_norm_KFold,
                                                                      k_folds, n_CVrepeats)

        # Calculate mean scores for the objective function
        score_r2_HC = np.mean([score_r2_H, score_r2_C])
        score_loss_HC = np.mean([score_loss_H, score_loss_C])

        # Append the scores to the corresponding lists
        score_r2_HC_list.append(score_r2_HC)
        score_loss_HC_list.append(score_loss_HC)
        score_r2_H_list.append(np.mean(score_r2_H))
        score_r2_C_list.append(np.mean(score_r2_C))
        score_loss_H_list.append(np.mean(score_loss_H))
        score_loss_C_list.append(np.mean(score_loss_C))

        # Display the progress
        self.bo_iteration += 1
        print(
            f'Iteration: {self.bo_iteration}/{n_initial_points + n_iterations}')

        return score_r2_HC, score_loss_HC

    def update_hypertable(self, fixed_hyper_space,
                          bo, score_r2_HC_list, score_loss_HC_list,
                          score_r2_H_list, score_r2_C_list, score_loss_H_list, score_loss_C_list):
        """
        This function updates the hypertable with the current Bayesian Optimization scores.

        Args:
            bo: The BayesianOptimization object.
            score_r2_HC_list, score_loss_HC_list, etc.: Lists to store scores for different tasks.
        """

        # print(self.hypertable)

        # reformat the fixed_hyper_space_dict
        fixed_hyper_space_dict = {
            entry['name']: entry['domain'] for entry in fixed_hyper_space}

        for i in range(len(bo.X)):
            row = dict(zip(self.search_hyper_names, bo.X[i]))

            # adding fixed_space parameters to the row
            row.update(fixed_hyper_space_dict)

            row['score_r2_HC_best'] = -bo.Y_best.flatten()[i]
            row['score_r2_HC'] = np.array(score_r2_HC_list)[i]
            row['score_loss_HC'] = np.array(score_loss_HC_list)[i]
            row['score_r2_H'] = np.array(score_r2_H_list)[i]
            row['score_r2_C'] = np.array(score_r2_C_list)[i]
            row['score_loss_H'] = np.array(score_loss_H_list)[i]
            row['score_loss_C'] = np.array(score_loss_C_list)[i]

            self.hypertable.loc[len(self.hypertable)] = row

    def plot_best_r2_score(self):
        """
        This function plots the best R2 scores.
        """
        def _plot(ax, col_name):
            data = self.hypertable[col_name].values
            best_scores = np.maximum.accumulate(data)
            ax.plot(best_scores, label=col_name, linestyle='-', marker='o')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Objective: R2 score')
            ax.grid(alpha=0.5, color='lightgrey')
            ax.legend()

        # Create a figure with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(
            12, 4), sharex=True, sharey=True)

        # Call the function for each column and subplot
        col_names = ['score_r2_HC', 'score_r2_H', 'score_r2_C']
        for ax, col_name in zip(axs, col_names):
            _plot(ax, col_name)

        # Add a title to the figure
        fig.suptitle(f'Best R2 Scores for bo_ens {self.bo_ens_num}')
        fig.tight_layout()
        plt.savefig(
            self.model_path_bo + f'BO_score_r2_bo_ens_{self.bo_ens_num}.png', format='png', dpi=200)
        plt.show()
