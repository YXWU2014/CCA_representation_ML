# multitask_nn.py
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import r2_score
import warnings
import logging
logging.basicConfig(filename='model_saving_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class MultiTaskNN:
    """
    This class implements a multi-task neural network architecture, including methods to create and train the network.
    The network consists of a shared feature network (NNS), an H-specific network (NNH), and a C-specific network (NNC).
    """
    # function for initializing the class

    def __init__(self,
                 NNS_num_nodes: int, NNS_num_layers: int, NNH_num_nodes: int, NNH_num_layers: int, NNC_num_nodes: int, NNC_num_layers: int,
                 mc_state: bool, act: str, NNS_dropout: float, NNH_dropout: float, NNC_dropout: float,
                 loss_func: str, learning_rate_H: float, learning_rate_C: float,
                 batch_size_H: int, N_epochs_local: int, total_epochs: int,
                 model_save_flag: bool, model_path_bo: str,
                 NNH_model_name: str = 'NNH_model_RepeatedKFold_{}',
                 NNC_model_name: str = 'NNC_model_RepeatedKFold_{}'):
        """
        Initialize the MultiTaskNN class with the given parameters.

        :param NNS_num_nodes: Number of nodes in the NNS layers
        :param NNS_num_layers: Number of layers in the NNS
        :param NNH_num_nodes: Number of nodes in the NNH layers
        :param NNH_num_layers: Number of layers in the NNH
        :param NNC_num_nodes: Number of nodes in the NNC layers
        :param NNC_num_layers: Number of layers in the NNC
        :param mc_state: Dropout state for Monte Carlo dropout
        :param act: Activation function to be used in the network
        :param NNS_dropout: Dropout rate for the NNS
        :param NNH_dropout: Dropout rate for the NNH
        :param NNC_dropout: Dropout rate for the NNC
        :param loss_func: Loss function for the network training
        :param learning_rate_H: Learning rate for the NNH
        :param learning_rate_C: Learning rate for the NNC
        :param batch_size_H: Batch size for the NNH
        :param N_epochs_global: Number of epochs for global training
        :param N_epochs_local: Number of epochs for local training
        :param model_save_flag: Flag to save the model after training
        :param model_path_bo: Path to save the trained model
        """
        # NN architecture parameters
        self.NNS_num_nodes = NNS_num_nodes
        self.NNS_num_layers = NNS_num_layers
        self.NNH_num_nodes = NNH_num_nodes
        self.NNH_num_layers = NNH_num_layers
        self.NNC_num_nodes = NNC_num_nodes
        self.NNC_num_layers = NNC_num_layers

        # NN training parameters
        self.mc_state = mc_state
        self.act = act  # activation function
        self.NNS_dropout = NNS_dropout
        self.NNH_dropout = NNH_dropout
        self.NNC_dropout = NNC_dropout
        self.loss_func = loss_func
        self.learning_rate_H = learning_rate_H
        self.learning_rate_C = learning_rate_C
        self.batch_size_H = batch_size_H

        # NN training parameters
        self.total_epochs = total_epochs
        self.N_epochs_local = N_epochs_local
        self.N_epochs_global = int(self.total_epochs/self.N_epochs_local)

        # NN saving parameters
        self.model_save_flag = model_save_flag
        self.model_path_bo = model_path_bo
        self.NNH_model_name = NNH_model_name
        self.NNC_model_name = NNC_model_name

    #  function for Monte Carlo dropout

    def get_dropout(self, input_tensor, p=0.5):
        """
        Apply Dropout to the given input tensor, with dropout rate `p`. 
        :param input_tensor: The input tensor to apply Dropout to
        :param p: The dropout rate, default is 0.5
        :return: Tensor with Dropout applied
        """
        if self.mc_state:
            return Dropout(p)(input_tensor, training=True)
        else:
            return Dropout(p)(input_tensor)

    #  function for creating shared feature network

    def create_NNS_model(self, input1_compo_features_shape):
        """
        Creates the shared feature network (NNS).

        :param input1_compo_features_shape: The shape of the component features input
        :return: The NNS model
        """
        input1_compo_features_layer = layers.Input(
            shape=input1_compo_features_shape)

        if self.NNS_num_layers == 0:
            return models.Model(inputs=input1_compo_features_layer, outputs=input1_compo_features_layer)

        NNS_l = input1_compo_features_layer

        for _ in range(self.NNS_num_layers):
            NNS_l = layers.Dense(self.NNS_num_nodes,
                                 activation=self.act)(NNS_l)
            # NNS_l = BatchNormalization()(NNS_l)
            NNS_l = self.get_dropout(NNS_l, p=self.NNS_dropout)

        return models.Model(inputs=input1_compo_features_layer, outputs=NNS_l)

    # function for creating H-specific network

    def create_NNH_model(self, NNS_model, input2_H_specific_shape):
        """
        Creates the H-specific network (NNH), which includes the NNS and additional layers.

        :param NNS_model: The shared feature network (NNS) model
        :param input2_H_specific_shape: The shape of the H-specific input
        :return: The NNH model
        """
        NNS_output = NNS_model.output
        model_inputs = [NNS_model.input]

        if input2_H_specific_shape != (0,):
            input2_H_specific_layer = layers.Input(
                shape=input2_H_specific_shape)
            NNH_l = Concatenate()([input2_H_specific_layer, NNS_output])
            model_inputs.append(input2_H_specific_layer)
        else:
            NNH_l = NNS_output

        for _ in range(self.NNH_num_layers):
            NNH_l = layers.Dense(self.NNH_num_nodes,
                                 activation=self.act)(NNH_l)
            # NNH_l = BatchNormalization()(NNH_l)
            NNH_l = self.get_dropout(NNH_l, p=self.NNH_dropout)

        # NNH_output = layers.Dense(1, activation='sigmoid')(NNH_l)
        NNH_output = layers.Dense(1)(NNH_l)

        return models.Model(inputs=model_inputs, outputs=NNH_output)

    #  function for creating C-specific network

    def create_NNC_model(self, NNS_model, input3_C_specific_shape):
        """
        Creates the C-specific network (NNC), which includes the NNS and additional layers.

        :param NNS_model: The shared feature network (NNS) model
        :param input3_C_specific_shape: The shape of the C-specific input
        :return: The NNC model
        """
        NNS_output = NNS_model.output
        model_inputs = [NNS_model.input]

        if input3_C_specific_shape != (0,):
            input3_C_specific_layer = layers.Input(
                shape=input3_C_specific_shape)
            NNC_l = Concatenate()([input3_C_specific_layer, NNS_output])
            model_inputs.append(input3_C_specific_layer)
        else:
            NNC_l = NNS_output

        for _ in range(self.NNC_num_layers):
            NNC_l = layers.Dense(self.NNC_num_nodes,
                                 activation=self.act)(NNC_l)
            # NNC_l = BatchNormalization()(NNC_l)
            NNC_l = self.get_dropout(NNC_l, p=self.NNC_dropout)

        # NNC_output = layers.Dense(1, activation='sigmoid')(NNC_l)
        NNC_output = layers.Dense(1)(NNC_l)

        return models.Model(inputs=model_inputs, outputs=NNC_output)

    # function for creating the full model

    def get_NN_full_model(self, input1_compo_features_shape, input2_H_specific_shape, input3_C_specific_shape):
        """
        This function constructs the full multi-task network model. It is composed of the shared features network (NNS) 
        and two task-specific networks (NNH and NNC). 

        :param input1_compo_features_shape: Shape of the input for the shared features network (NNS).
        :param input2_H_specific_shape: Shape of the input specific to the first task (NNH).
        :param input3_C_specific_shape: Shape of the input specific to the second task (NNC).
        :return: Compiled NNH and NNC models.
        """

        # Construct and compile the shared features network (NNS).
        NNS_model = self.create_NNS_model(input1_compo_features_shape)

        # Construct and compile the first task-specific network (NNH).
        NNH_model = self.create_NNH_model(NNS_model, input2_H_specific_shape)
        NNH_model.compile(loss=self.loss_func,
                          optimizer=Adam(self.learning_rate_H))

        # Construct and compile the second task-specific network (NNC).
        NNC_model = self.create_NNC_model(NNS_model, input3_C_specific_shape)
        NNC_model.compile(loss=self.loss_func,
                          optimizer=Adam(self.learning_rate_C))

        return NNH_model, NNC_model

    # function for train model in parallel

    def evaluate_NN_full_model_parallel(self, args):
        """
        This function trains and evaluates the full multi-task network model in parallel. It performs a number of steps 
        including data preprocessing, model definition and compilation, model training, and model evaluation.

        :param args: A tuple containing all necessary input parameters.
        :return: Training and validation losses for NNH and NNC models, test loss and predictions for both models.
        """
        tf.keras.backend.clear_session()

        # Unpack input arguments
        (i_fold,
         X1_train_norm_temp, X1_test_norm_temp, Y1_train_norm_temp, Y1_test_norm_temp, V1_train_norm_temp, V1_test_norm_temp, H1_train_norm_temp, H1_test_norm_temp,
         X2_train_norm_temp, X2_test_norm_temp, Z2_train_norm_temp, Z2_test_norm_temp, W2_train_norm_temp, W2_test_norm_temp, C2_train_norm_temp, C2_test_norm_temp,
         ) = args

        # concatenate arrays for the input1_compo_features_shape
        X1_V1_train_norm_temp = np.concatenate(
            [X1_train_norm_temp, V1_train_norm_temp], axis=1)
        X1_V1_test_norm_temp = np.concatenate(
            [X1_test_norm_temp, V1_test_norm_temp], axis=1)
        X2_W2_train_norm_temp = np.concatenate(
            [X2_train_norm_temp, W2_train_norm_temp], axis=1)
        X2_W2_test_norm_temp = np.concatenate(
            [X2_test_norm_temp, W2_test_norm_temp], axis=1)

        # dimension of input dimension
        input1_compo_features_shape = (
            X1_V1_train_norm_temp.shape[1],)  # X1_V1 vs X2+W2 must have same shape
        # now no H specific input: (0,)
        input2_H_specific_shape = (Y1_train_norm_temp.shape[1],)
        # this is for `C_specific_testing`
        input3_C_specific_shape = (Z2_train_norm_temp.shape[1],)

        # define model + compile model
        NNH_model, NNC_model = self.get_NN_full_model(
            input1_compo_features_shape, input2_H_specific_shape, input3_C_specific_shape)

        if i_fold == 0 and self.model_save_flag:
            plot_model(NNH_model, to_file=self.model_path_bo+'NNH_model_architecture.png',
                       show_shapes=True, show_layer_names=True)
            plot_model(NNC_model, to_file=self.model_path_bo+'NNC_model_architecture.png',
                       show_shapes=True, show_layer_names=True)

        # ----- train model: start ------------------------------------------
        # batch_size_H = 32
        N_batches = H1_train_norm_temp.shape[0] // self.batch_size_H + int(
            H1_train_norm_temp.shape[0] % self.batch_size_H > 0)  # to loop through both datasets
        batch_size_C = int(C2_train_norm_temp.shape[0]/N_batches)

        history_H = []  # loss history
        history_C = []  # loss history

        for N in range(self.N_epochs_global):

            # NNH_model
            X1_V1_train_norm_concat = np.concatenate(
                [X1_V1_train_norm_temp[i*self.batch_size_H: (i+1)*self.batch_size_H] for i in range(N_batches)])
            H1_train_norm_concat = np.concatenate(
                [H1_train_norm_temp[i*self.batch_size_H: (i+1)*self.batch_size_H] for i in range(N_batches)])

            if input2_H_specific_shape == (0,):  # no H_testing input
                history_H_temp = NNH_model.fit(X1_V1_train_norm_concat, H1_train_norm_concat,
                                               validation_data=(
                                                   X1_V1_test_norm_temp, H1_test_norm_temp),
                                               epochs=self.N_epochs_local, verbose=0)
            else:
                warnings.warn("error on 'input2_H_specific_shape'.")

            # NNC_model
            X2_W2_train_norm_concat = np.concatenate(
                [X2_W2_train_norm_temp[i*batch_size_C: (i+1)*batch_size_C] for i in range(N_batches)])
            Z2_train_norm_concat = np.concatenate(
                [Z2_train_norm_temp[i*batch_size_C: (i+1)*batch_size_C] for i in range(N_batches)])
            C2_train_norm_concat = np.concatenate(
                [C2_train_norm_temp[i*batch_size_C: (i+1)*batch_size_C] for i in range(N_batches)])

            if input3_C_specific_shape == (0,):  # no C_testing input
                history_C_temp = NNC_model.fit(X2_W2_train_norm_concat, C2_train_norm_concat,
                                               validation_data=(
                                                   X2_W2_test_norm_temp, C2_test_norm_temp),
                                               epochs=self.N_epochs_local, verbose=0)
            elif input3_C_specific_shape != (0,):  # with C_testing input
                history_C_temp = NNC_model.fit([X2_W2_train_norm_concat, Z2_train_norm_concat], C2_train_norm_concat,
                                               validation_data=(
                                                   [X2_W2_test_norm_temp, Z2_test_norm_temp], C2_test_norm_temp),
                                               epochs=self.N_epochs_local, verbose=0)
            else:
                warnings.warn("error on 'input3_C_specific_shape'.")

            history_H.append(history_H_temp)
            history_C.append(history_C_temp)

        # Extract training and validation losses
        train_loss_H_temp = np.array([a.history['loss']
                                      for a in history_H]).reshape(-1, 1)
        val_loss_H_temp = np.array([a.history['val_loss']
                                    for a in history_H]).reshape(-1, 1)
        train_loss_C_temp = np.array([a.history['loss']
                                      for a in history_C]).reshape(-1, 1)
        val_loss_C_temp = np.array([a.history['val_loss']
                                    for a in history_C]).reshape(-1, 1)

        # ----- save model ------------------------------------------
        if self.mc_state:
            NNH_model_name = f'{self.NNH_model_name.format(i_fold + 1, act=self.act)}.h5'
            NNC_model_name = f'{self.NNC_model_name.format(i_fold + 1, act=self.act)}.h5'
        else:
            NNH_model_name = f'{self.NNH_model_name.format(i_fold + 1, act=self.act)}.h5'
            NNC_model_name = f'{self.NNC_model_name.format(i_fold + 1, act=self.act)}.h5'

        if self.model_save_flag:
            NNH_model.save(os.path.join(self.model_path_bo, NNH_model_name))
            NNC_model.save(os.path.join(self.model_path_bo, NNC_model_name))

            logger.info('Saved model: %s', NNH_model_name)
            logger.info('Saved model: %s', NNC_model_name)

        # ----- evaluate model ------------------------------------------
        # evaluate model on test set

        def compute_loss_error(model, input_data, target_data, iterations):
            return np.mean(np.stack([model.evaluate(input_data, target_data, verbose=0) for _ in range(iterations)]))

        def compute_prediction(model, input_data, iterations):
            return np.mean(np.stack([model.predict(input_data, verbose=0) for _ in range(iterations)]), axis=0)

        iterations = 50 if self.mc_state else 1

        if input2_H_specific_shape == (0,):  # no H_testing input
            NNH_loss_error_temp = compute_loss_error(
                NNH_model, X1_V1_test_norm_temp, H1_test_norm_temp, iterations)  # final loss function value
            NNH_pred_temp = compute_prediction(
                NNH_model, X1_V1_test_norm_temp, iterations)  # make prediction
        else:
            warnings.warn("error on 'input2_H_specific_shape'.")

        if input3_C_specific_shape == (0,):  # no C_testing input

            NNC_loss_error_temp = compute_loss_error(
                NNC_model, X2_W2_test_norm_temp, C2_test_norm_temp, iterations)  # final loss function value
            NNC_pred_temp = compute_prediction(
                NNC_model, X2_W2_test_norm_temp, iterations)  # make prediction

        elif input3_C_specific_shape != (0,):  # with C_testing input

            NNC_loss_error_temp = compute_loss_error(
                NNC_model, [X2_W2_test_norm_temp, Z2_test_norm_temp], C2_test_norm_temp, iterations)  # final loss function value
            NNC_pred_temp = compute_prediction(
                NNC_model, [X2_W2_test_norm_temp, Z2_test_norm_temp], iterations)  # make prediction
        else:
            warnings.warn("error on 'input3_C_specific_shape'.")

        # prepare the output
        NNH_test_pred_temp = np.concatenate(
            [H1_test_norm_temp, NNH_pred_temp], axis=1)
        NNC_test_pred_temp = np.concatenate(
            [C2_test_norm_temp, NNC_pred_temp], axis=1)

        return (train_loss_H_temp, val_loss_H_temp,
                train_loss_C_temp, val_loss_C_temp,
                NNH_loss_error_temp, NNC_loss_error_temp,
                NNH_test_pred_temp, NNC_test_pred_temp)

    # function to call the parallelized training

    def evaluate_NN_full_model(self, X1_train_norm_KFold, X1_test_norm_KFold, Y1_train_norm_KFold, Y1_test_norm_KFold, V1_train_norm_KFold, V1_test_norm_KFold, H1_train_norm_KFold, H1_test_norm_KFold,
                               X2_train_norm_KFold, X2_test_norm_KFold, Z2_train_norm_KFold, Z2_test_norm_KFold, W2_train_norm_KFold, W2_test_norm_KFold, C2_train_norm_KFold, C2_test_norm_KFold,
                               k_folds, n_CVrepeats):
        """
        Orchestrates the training and evaluation of machine learning models. It initiates parallel processes for training 
        and evaluating the models and collects the results.

        Parameters:
        * X1_train_norm_KFold, X1_test_norm_KFold, V1_train_norm_KFold, V1_test_norm_KFold, H1_train_norm_KFold, H1_test_norm_KFold: training and testing data for the NNH model.
        * X2_train_norm_KFold, X2_test_norm_KFold, Z2_train_norm_KFold, Z2_test_norm_KFold, W2_train_norm_KFold, W2_test_norm_KFold, C2_train_norm_KFold, C2_test_norm_KFold: training and testing data for the NNC model.
        * k_folds (int): number of folds for the K-Fold cross-validation.
        * n_CVrepeats (int): number of repetitions of the cross-validation process.

        Returns:
        tuple: Returns tuples of training and validation losses, score losses, and R2 scores for both models.
        """
        train_loss_H, train_loss_C = [], []
        val_loss_H, val_loss_C = [], []

        score_loss_H, score_loss_C = [], []
        score_r2_H, score_r2_C = [], []

        args_list = []

        for i_fold in range(k_folds * n_CVrepeats):
            X1_train_norm_temp, X1_test_norm_temp = X1_train_norm_KFold[
                i_fold], X1_test_norm_KFold[i_fold]
            Y1_train_norm_temp, Y1_test_norm_temp = Y1_train_norm_KFold[
                i_fold], Y1_test_norm_KFold[i_fold]
            V1_train_norm_temp, V1_test_norm_temp = V1_train_norm_KFold[
                i_fold], V1_test_norm_KFold[i_fold]
            H1_train_norm_temp, H1_test_norm_temp = H1_train_norm_KFold[
                i_fold], H1_test_norm_KFold[i_fold]

            X2_train_norm_temp, X2_test_norm_temp = X2_train_norm_KFold[
                i_fold], X2_test_norm_KFold[i_fold]
            Z2_train_norm_temp, Z2_test_norm_temp = Z2_train_norm_KFold[
                i_fold], Z2_test_norm_KFold[i_fold]
            W2_train_norm_temp, W2_test_norm_temp = W2_train_norm_KFold[
                i_fold], W2_test_norm_KFold[i_fold]
            C2_train_norm_temp, C2_test_norm_temp = C2_train_norm_KFold[
                i_fold], C2_test_norm_KFold[i_fold]

            args_list.append((i_fold,
                              X1_train_norm_temp, X1_test_norm_temp, Y1_train_norm_temp, Y1_test_norm_temp, V1_train_norm_temp, V1_test_norm_temp, H1_train_norm_temp, H1_test_norm_temp,
                              X2_train_norm_temp, X2_test_norm_temp, Z2_train_norm_temp, Z2_test_norm_temp, W2_train_norm_temp, W2_test_norm_temp, C2_train_norm_temp, C2_test_norm_temp,
                              ))

        # Initiate parallel processes for model training and evaluation
        with Pool() as p:
            results = p.map(self.evaluate_NN_full_model_parallel, args_list)

        # print("I am groot!")

        # Collect results from parallel processes
        for i_fold in range(k_folds * n_CVrepeats):

            train_loss_H.append(results[i_fold][0])
            val_loss_H.append(results[i_fold][1])
            train_loss_C.append(results[i_fold][2])
            val_loss_C.append(results[i_fold][3])
            score_loss_H.append(results[i_fold][4])
            score_loss_C.append(results[i_fold][5])

            NNH_test_pred_temp = results[i_fold][6]
            NNC_test_pred_temp = results[i_fold][7]

            if np.isnan(NNH_test_pred_temp[:, 0]).any():
                print("NNH_test_pred_temp[:, 0] contains NaN values.")

            if np.isnan(NNH_test_pred_temp[:, 1]).any():
                print("NNH_test_pred_temp[:, 1] contains NaN values.")

            score_r2_H_temp = r2_score(
                NNH_test_pred_temp[:, 0], NNH_test_pred_temp[:, 1])
            score_r2_C_temp = r2_score(
                NNC_test_pred_temp[:, 0], NNC_test_pred_temp[:, 1])

            score_r2_H.append(score_r2_H_temp)
            score_r2_C.append(score_r2_C_temp)

        # clear TensorFlow session
        tf.keras.backend.clear_session()

        # return results
        # train_loss_H: training loss history of the NNH model
        # train_loss_C: training loss history of the NNC model
        # val_loss_H: validation loss history of the NNH model
        # val_loss_C: validation loss history of the NNC model
        # score_loss_H: evaluated loss score of the NNH model
        # score_loss_C: evaluated loss score of the NNC model
        # score_r2_H: computed R2 score of the NNH model
        # score_r2_C: computed R2 score of the NNC model

        return (train_loss_H, train_loss_C,
                val_loss_H,   val_loss_C,
                score_loss_H, score_loss_C,
                score_r2_H,   score_r2_C)

    # plot the training and validation losses for both tasks

    def plot_losses(self, train_loss_H, val_loss_H,
                    train_loss_C, val_loss_C, k_folds, n_CVrepeats, index=0):
        """
        Plot training and validation losses for tasks H and C over multiple epochs.

        Args:
            train_loss_H (list): Training losses for task H.
            val_loss_H (list): Validation losses for task H.
            train_loss_C (list): Training losses for task C.
            val_loss_C (list): Validation losses for task C.
            k_folds (int): The number of folds used for cross-validation.
            n_CVrepeats (int): The number of times cross-validation was repeated.

        Returns:
            None. A plot is displayed and saved to the model_path_bo directory.
        """
        # Initialize a new figure with two subplots side by side
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

        # For each fold in the cross-validation
        for i in range(k_folds * n_CVrepeats):
            # Only label the first set of plots for clarity
            if i == index:
                # Plot training and validation losses for task H
                ax[0].plot(train_loss_H[i], label=f"Train Loss_H",
                           linewidth=1, color='steelblue', alpha=0.5)
                ax[0].plot(val_loss_H[i], label=f"Validation Loss_H",
                           linewidth=1, color='firebrick', alpha=0.5)

                # Plot training and validation losses for task C
                ax[1].plot(train_loss_C[i], label=f"Train Loss_C",
                           linewidth=1, color='steelblue', alpha=0.5)
                ax[1].plot(val_loss_C[i], label=f"Validation Loss_C",
                           linewidth=1, color='firebrick', alpha=0.5)

        # Set labels, title, legend, and grid for each subplot
        for axi in ax.flat:
            axi.set_xlabel("Epochs")  # x-axis label represents training epochs
            # y-axis label represents the loss or error
            axi.set_ylabel("Error")
            axi.set_title('MSE')  # title of the plot
            axi.legend()  # show legend on the plot
            axi.grid()  # display gridlines on the plot
            axi.set_box_aspect(1)  # maintain aspect ratio
            axi.set_ylim(0, 0.03)  # set limits for y-axis

        # Save the figure as a .png image in the model_path_bo directory
        plt.savefig(self.model_path_bo +
                    'NN_full_RepeatedKFold_loss.png', format='png', dpi=200)

        # Display the figure
        plt.show()
