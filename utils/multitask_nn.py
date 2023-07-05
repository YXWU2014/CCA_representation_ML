# multitask_nn.py
import os
import numpy as np
from multiprocessing import Pool
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


class MultitaskNN:
    def __init__(self, mc_state, act, NNF_dropout, NNH_dropout, NNC_dropout,
                 loss_func, learning_rate_H, learning_rate_C,
                 batch_size_H, N_epochs_global, N_epochs_local, model_save_flag):
        self.mc_state = mc_state
        self.act = act
        self.NNF_dropout = NNF_dropout
        self.NNH_dropout = NNH_dropout
        self.NNC_dropout = NNC_dropout
        self.loss_func = loss_func
        self.learning_rate_H = learning_rate_H
        self.learning_rate_C = learning_rate_C
        self.batch_size_H = batch_size_H
        self.N_epochs_global = N_epochs_global
        self.N_epochs_local = N_epochs_local
        self.model_save_flag = model_save_flag

    def get_dropout(self, input_tensor, p=0.5):
        if self.mc_state:
            return Dropout(p)(input_tensor, training=True)
        else:
            return Dropout(p)(input_tensor)

    def create_NNF_model(self, input1_compo_features_shape, NNF_num_nodes, NNF_num_layers):
        input1_compo_features_layer = layers.Input(
            shape=input1_compo_features_shape)

        if NNF_num_layers == 0:
            return models.Model(inputs=input1_compo_features_layer, outputs=input1_compo_features_layer)

        NNF_l = input1_compo_features_layer

        for _ in range(NNF_num_layers):
            NNF_l = layers.Dense(NNF_num_nodes, activation=self.act)(NNF_l)
            NNF_l = BatchNormalization()(NNF_l)
            NNF_l = self.get_dropout(NNF_l, p=self.NNF_dropout)

        return models.Model(inputs=input1_compo_features_layer, outputs=NNF_l)

    def create_NNH_model(self, NNF_model, input2_H_specific_shape, NNH_num_nodes, NNH_num_layers):
        NNF_output = NNF_model.output
        model_inputs = [NNF_model.input]

        if input2_H_specific_shape != ():
            input2_H_specific_layer = layers.Input(
                shape=input2_H_specific_shape)
            NNH_l = Concatenate()([input2_H_specific_layer, NNF_output])
            model_inputs.append(input2_H_specific_layer)
        else:
            NNH_l = NNF_output

        for _ in range(NNH_num_layers):
            NNH_l = layers.Dense(NNH_num_nodes, activation=self.act)(NNH_l)
            NNH_l = BatchNormalization()(NNH_l)
            NNH_l = self.get_dropout(NNH_l, p=self.NNH_dropout)

        NNH_output = layers.Dense(1, activation='sigmoid')(NNH_l)

        return models.Model(inputs=model_inputs, outputs=NNH_output)

    def create_NNC_model(self, NNF_model, input3_C_specific_shape, NNC_num_nodes, NNC_num_layers):
        NNF_output = NNF_model.output
        model_inputs = [NNF_model.input]

        if input3_C_specific_shape != ():
            input3_C_specific_layer = layers.Input(
                shape=input3_C_specific_shape)
            NNC_l = Concatenate()([input3_C_specific_layer, NNF_output])
            model_inputs.append(input3_C_specific_layer)
        else:
            NNC_l = NNF_output

        for _ in range(NNC_num_layers):
            NNC_l = layers.Dense(NNC_num_nodes, activation=self.act)(NNC_l)
            NNC_l = BatchNormalization()(NNC_l)
            NNC_l = self.get_dropout(NNC_l, p=self.NNC_dropout)

        NNC_output = layers.Dense(1, activation='sigmoid')(NNC_l)

        return models.Model(inputs=model_inputs, outputs=NNC_output)


# adding more functions to the class
