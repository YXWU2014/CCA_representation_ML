# Multi-Task Learning Neural Network for Sparsely Labelled Data

## Overview

This project utilises a hard parameter sharing approach in multi-task learning with neural networks to handle sparsely labelled data. The architecture employs shared hidden layers for multiple tasks before branching into task-specific networks. These networks are fine-tuned for the unique attributes of each task, leveraging TensorFlow and Keras for implementation. The primary challenge addressed is the development of a training algorithm capable of managing sparsely labelled data, such as non-concurrent hardness and corrosion measurements in materials science.

## Methodology

An iterative training approach focuses on subsets of the data, maintaining input-output consistency and adjusting neural network weights through backpropagation. This method allows separate yet intrinsically linked models for hardness and corrosion to share weights in the hidden layers, promoting efficiency and effectiveness in learning.

## Training Algorithm

The training algorithm operates iteratively, alternating between data types (hardness/corrosion) to update the weights in both shared and task-specific networks. This process is detailed in our custom algorithm, which encompasses exploratory data analysis, data preparation, model initiation, compilation, training, evaluation, and explainability steps.

## Key Features

- **Iterative Training Approach:** Focuses on effective weight adjustment in sparsely labelled datasets.
- **Shared and Task-Specific Networks:** Leverages shared representations to improve learning efficiency and model performance.
- **Model Ensemble:** Employs ensemble learning for handling sparsity in labels, improving robustness and accuracy.
- **Algorithmic Workflow:** Detailed steps from data analysis to model evaluation, including model initiation, compilation, and training specifics.
- **Hyperparameter Optimization:** Utilises Bayesian Optimisation for efficient hyperparameter selection, optimising model performance based on \(R^{2}\) score.
- **Model Explainability:** Implements a Shapley value-based framework for local explainability of model predictions.

- all above features are unified in one model

## Implementation Details

The project is implemented using TensorFlow and Keras, with Bayesian Optimisation facilitated by the GPyOpt library for hyperparameter tuning. 