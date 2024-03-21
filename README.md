# Multi-Task Learning Neural Network for Sparsely Labelled Data

## Overview

This project utilises a hard parameter sharing approach in multi-task learning in neural networks to handle sparsely labelled data. The architecture employs shared hidden layers for multiple tasks before branching into task-specific networks.

These networks are fine-tuned for the unique attributes of each task. The primary problem addressed is the training of sparsely labelled data, such as non-concurrent hardness and corrosion measurements in materials science.

This repository contains the complete workflow: exploratory data analysis, data preparation, model compilation, training, evaluation, and explainability steps.


![Fig_2_MTL](Fig_2_MTL.png)  


## Methodology

The training algorithm operates iteratively, alternating between data types (hardness/corrosion) to update the weights in both shared and task-specific networks. This method allows separate yet intrinsically linked models for hardness and corrosion to share weights in the hidden layers.

## Key Features

- **Shared and Task-Specific Networks:** Leverages shared representations to improve learning efficiency and reduce overfitting.
- **Model Ensemble:** Employs ensemble learning for handling heterogeneous small datasets.
- **Hyperparameter Optimization:** Utilises Bayesian Optimisation for efficient hyperparameter selection (subnetwork configurations), optimising model performance based on \(R^{2}\) score.
- **Model Explainability:** Implements a Shapley value-based framework for local explainability of model predictions.

- All above features are unified in one model.

## Implementation Details

The project is implemented using TensorFlow and Keras, with Bayesian Optimisation facilitated by the GPyOpt library for hyperparameter tuning.
