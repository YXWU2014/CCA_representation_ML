# multitask_nn.py
import os
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import r2_score


class MultiTaskNN:

    # function for initializing the class
    def __init__(self,
                 NNF_num_nodes: int, NNF_num_layers: int, NNH_num_nodes: int, NNH_num_layers: int, NNC_num_nodes: int, NNC_num_layers: int,
                 mc_state: bool, act: str, NNF_dropout: float, NNH_dropout: float, NNC_dropout: float,
                 loss_func: str, learning_rate_H: float, learning_rate_C: float,
                 batch_size_H: int, N_epochs_global: int, N_epochs_local: int, model_save_flag: bool, model_path_bo: str):

        # NN architecture parameters
        self.NNF_num_nodes = NNF_num_nodes
        self.NNF_num_layers = NNF_num_layers
        self.NNH_num_nodes = NNH_num_nodes
        self.NNH_num_layers = NNH_num_layers
        self.NNC_num_nodes = NNC_num_nodes
        self.NNC_num_layers = NNC_num_layers

        # NN training parameters
        self.mc_state = mc_state
        self.act = act  # activation function
        self.NNF_dropout = NNF_dropout
        self.NNH_dropout = NNH_dropout
        self.NNC_dropout = NNC_dropout
        self.loss_func = loss_func
        self.learning_rate_H = learning_rate_H
        self.learning_rate_C = learning_rate_C
        self.batch_size_H = batch_size_H

        # NN training parameters
        self.N_epochs_global = N_epochs_global
        self.N_epochs_local = N_epochs_local

        # NN saving parameters
        self.model_save_flag = model_save_flag
        self.model_path_bo = model_path_bo

    #  function for Monte Carlo dropout

    def get_dropout(self, input_tensor, p=0.5):
        if self.mc_state:
            return Dropout(p)(input_tensor, training=True)
        else:
            return Dropout(p)(input_tensor)

    #  function for creating shared feature network

    def create_NNF_model(self, input1_compo_features_shape):
        input1_compo_features_layer = layers.Input(
            shape=input1_compo_features_shape)

        if self.NNF_num_layers == 0:
            return models.Model(inputs=input1_compo_features_layer, outputs=input1_compo_features_layer)

        NNF_l = input1_compo_features_layer

        for _ in range(self.NNF_num_layers):
            NNF_l = layers.Dense(self.NNF_num_nodes,
                                 activation=self.act)(NNF_l)
            NNF_l = BatchNormalization()(NNF_l)
            NNF_l = self.get_dropout(NNF_l, p=self.NNF_dropout)

        return models.Model(inputs=input1_compo_features_layer, outputs=NNF_l)

    # function for creating H-specific network

    def create_NNH_model(self, NNF_model, input2_H_specific_shape):
        NNF_output = NNF_model.output
        model_inputs = [NNF_model.input]

        if input2_H_specific_shape != ():
            input2_H_specific_layer = layers.Input(
                shape=input2_H_specific_shape)
            NNH_l = Concatenate()([input2_H_specific_layer, NNF_output])
            model_inputs.append(input2_H_specific_layer)
        else:
            NNH_l = NNF_output

        for _ in range(self.NNH_num_layers):
            NNH_l = layers.Dense(self.NNH_num_nodes,
                                 activation=self.act)(NNH_l)
            NNH_l = BatchNormalization()(NNH_l)
            NNH_l = self.get_dropout(NNH_l, p=self.NNH_dropout)

        NNH_output = layers.Dense(1, activation='sigmoid')(NNH_l)

        return models.Model(inputs=model_inputs, outputs=NNH_output)

    #  function for creating C-specific network

    def create_NNC_model(self, NNF_model, input3_C_specific_shape):
        NNF_output = NNF_model.output
        model_inputs = [NNF_model.input]

        if input3_C_specific_shape != ():
            input3_C_specific_layer = layers.Input(
                shape=input3_C_specific_shape)
            NNC_l = Concatenate()([input3_C_specific_layer, NNF_output])
            model_inputs.append(input3_C_specific_layer)
        else:
            NNC_l = NNF_output

        for _ in range(self.NNC_num_layers):
            NNC_l = layers.Dense(self.NNC_num_nodes,
                                 activation=self.act)(NNC_l)
            NNC_l = BatchNormalization()(NNC_l)
            NNC_l = self.get_dropout(NNC_l, p=self.NNC_dropout)

        NNC_output = layers.Dense(1, activation='sigmoid')(NNC_l)

        return models.Model(inputs=model_inputs, outputs=NNC_output)

    # function for creating the full model

    def get_NN_full_model(self, input1_compo_features_shape, input2_H_specific_shape, input3_C_specific_shape):
        # Create and compile the NNF model
        NNF_model = self.create_NNF_model(input1_compo_features_shape)

        # Create and compile the NNH model
        NNH_model = self.create_NNH_model(
            NNF_model, input2_H_specific_shape)
        NNH_model.compile(loss=self.loss_func,
                          optimizer=Adam(self.learning_rate_H))

        # Create and compile the NNC model
        NNC_model = self.create_NNC_model(NNF_model, input3_C_specific_shape)
        NNC_model.compile(
            loss=self.loss_func, optimizer=Adam(self.learning_rate_C))

        return NNH_model, NNC_model

    # function ti train model in parallel

    def evaluate_NN_full_model_parallel(self, args):
        (i_fold,
         X1_train_norm_temp, X1_test_norm_temp, Y1_train_norm_temp, Y1_test_norm_temp, H1_train_norm_temp, H1_test_norm_temp,
         X2_train_norm_temp, X2_test_norm_temp, Z2_train_norm_temp, Z2_test_norm_temp, W2_train_norm_temp, W2_test_norm_temp, C2_train_norm_temp, C2_test_norm_temp,
         ) = args

        # concatenate arrays for the input1_compo_features_shape
        X1_Y1_train_norm_temp = np.concatenate(
            [X1_train_norm_temp, Y1_train_norm_temp], axis=1)
        X1_Y1_test_norm_temp = np.concatenate(
            [X1_test_norm_temp, Y1_test_norm_temp], axis=1)
        X2_W2_train_norm_temp = np.concatenate(
            [X2_train_norm_temp, W2_train_norm_temp], axis=1)
        X2_W2_test_norm_temp = np.concatenate(
            [X2_test_norm_temp, W2_test_norm_temp], axis=1)

        # dimension of input dimension
        input1_compo_features_shape = (
            X1_Y1_train_norm_temp.shape[1],)  # or X2+W2
        input2_H_specific_shape = ()  # now no H specific input
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
            X1_Y1_train_norm_concat = np.concatenate(
                [X1_Y1_train_norm_temp[i*self.batch_size_H: (i+1)*self.batch_size_H] for i in range(N_batches)])
            H1_train_norm_concat = np.concatenate(
                [H1_train_norm_temp[i*self.batch_size_H: (i+1)*self.batch_size_H] for i in range(N_batches)])

            if input2_H_specific_shape == ():
                history_H_temp = NNH_model.fit(X1_Y1_train_norm_concat, H1_train_norm_concat,
                                               validation_data=(
                                                   X1_Y1_test_norm_temp, H1_test_norm_temp),
                                               epochs=self.N_epochs_local, verbose=0)

            # NNC_model
            X2_W2_train_norm_concat = np.concatenate(
                [X2_W2_train_norm_temp[i*batch_size_C: (i+1)*batch_size_C] for i in range(N_batches)])
            Z2_train_norm_concat = np.concatenate(
                [Z2_train_norm_temp[i*batch_size_C: (i+1)*batch_size_C] for i in range(N_batches)])
            C2_train_norm_concat = np.concatenate(
                [C2_train_norm_temp[i*batch_size_C: (i+1)*batch_size_C] for i in range(N_batches)])

            history_C_temp = NNC_model.fit([X2_W2_train_norm_concat, Z2_train_norm_concat], C2_train_norm_concat,
                                           validation_data=(
                [X2_W2_test_norm_temp, Z2_test_norm_temp], C2_test_norm_temp),
                epochs=self.N_epochs_local, verbose=0)

            history_H.append(history_H_temp)
            history_C.append(history_C_temp)

        train_loss_H_temp = np.array([a.history['loss']
                                      for a in history_H]).reshape(-1, 1)
        val_loss_H_temp = np.array([a.history['val_loss']
                                    for a in history_H]).reshape(-1, 1)
        train_loss_C_temp = np.array([a.history['loss']
                                      for a in history_C]).reshape(-1, 1)
        val_loss_C_temp = np.array([a.history['val_loss']
                                    for a in history_C]).reshape(-1, 1)

        # ----- train model: finish ------------------------------------------

        # save the model
        NNH_model_name = f'NNH_model_RepeatedKFold_{i_fold+1}.h5'
        NNC_model_name = f'NNC_model_RepeatedKFold_{i_fold+1}.h5'

        if self.model_save_flag:
            NNH_model.save(os.path.join(self.model_path_bo, NNH_model_name))
            NNC_model.save(os.path.join(self.model_path_bo, NNC_model_name))

        # evaluate model on test set
        if input2_H_specific_shape == ():
            NNH_loss_error_temp = np.mean(np.stack([NNH_model.evaluate(
                X1_Y1_test_norm_temp, H1_test_norm_temp, verbose=0) for _ in range(50)]))
        NNC_loss_error_temp = np.mean(np.stack([NNC_model.evaluate(
            [X2_W2_test_norm_temp, Z2_test_norm_temp], C2_test_norm_temp, verbose=0) for _ in range(50)]))

        if input2_H_specific_shape == ():
            NNH_pred_temp = np.mean(np.stack([NNH_model.predict(
                X1_Y1_test_norm_temp, verbose=0) for _ in range(50)]), axis=0)
        NNC_pred_temp = np.mean(np.stack([NNC_model.predict(
            [X2_W2_test_norm_temp, Z2_test_norm_temp], verbose=0) for _ in range(50)]), axis=0)

        NNH_test_pred_temp = np.concatenate(
            [H1_test_norm_temp, NNH_pred_temp], axis=1)
        NNC_test_pred_temp = np.concatenate(
            [C2_test_norm_temp, NNC_pred_temp], axis=1)

        # print(f'NNH_test_pred_temp: {NNH_test_pred_temp.shape}')
        # print(f'NNC_test_pred_temp: {NNC_test_pred_temp.shape}')

        # store result
        # print(f'NNH_Fold*Repeat {i+1}: error={NNH_loss_error_temp:.4f}')
        # print(f'NNC_Fold*Repeat {i+1}: error={NNC_loss_error_temp:.4f}')

        return (train_loss_H_temp, val_loss_H_temp,
                train_loss_C_temp, val_loss_C_temp,
                NNH_loss_error_temp, NNC_loss_error_temp,
                NNH_test_pred_temp, NNC_test_pred_temp)

    # function to call the parallelized training
    def evaluate_NN_full_model(self, X1_train_norm_KFold, X1_test_norm_KFold, Y1_train_norm_KFold, Y1_test_norm_KFold, H1_train_norm_KFold, H1_test_norm_KFold,
                               X2_train_norm_KFold, X2_test_norm_KFold, Z2_train_norm_KFold, Z2_test_norm_KFold, W2_train_norm_KFold, W2_test_norm_KFold, C2_train_norm_KFold, C2_test_norm_KFold,
                               k_folds, n_CVrepeats):

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
                              X1_train_norm_temp, X1_test_norm_temp, Y1_train_norm_temp, Y1_test_norm_temp, H1_train_norm_temp, H1_test_norm_temp,
                              X2_train_norm_temp, X2_test_norm_temp, Z2_train_norm_temp, Z2_test_norm_temp, W2_train_norm_temp, W2_test_norm_temp, C2_train_norm_temp, C2_test_norm_temp,
                              ))

        with Pool() as p:
            results = p.map(self.evaluate_NN_full_model_parallel, args_list)

        for i_fold in range(k_folds * n_CVrepeats):

            train_loss_H.append(results[i_fold][0])
            val_loss_H.append(results[i_fold][1])
            train_loss_C.append(results[i_fold][2])
            val_loss_C.append(results[i_fold][3])
            score_loss_H.append(results[i_fold][4])
            score_loss_C.append(results[i_fold][5])

            NNH_test_pred_temp = results[i_fold][6]
            NNC_test_pred_temp = results[i_fold][7]

            score_r2_H_temp = r2_score(
                NNH_test_pred_temp[:, 0], NNH_test_pred_temp[:, 1])
            score_r2_C_temp = r2_score(
                NNC_test_pred_temp[:, 0], NNC_test_pred_temp[:, 1])

            score_r2_H.append(score_r2_H_temp)
            score_r2_C.append(score_r2_C_temp)

        # clear TensorFlow session
        tf.keras.backend.clear_session()

        return (train_loss_H, train_loss_C,
                val_loss_H,   val_loss_C,
                score_loss_H, score_loss_C,
                score_r2_H,   score_r2_C)
