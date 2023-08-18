'''
cd to the .py directory and run this script
'''

import papermill as pm
import numpy as np
from multiprocessing import Pool
import time
import os

# Constants
NOTEBOOK_MASTER = 'NN_full_v3_BO_Train_Eval_Pred_master'
# dir in respect to this batch script
MASTER_PATH = '../03_Model_Train_Evaluate_Predict/'
MODEL_PATH = '../04_Model_Saved/'
CONDA_ENV = '/nethome/yuxiang.wu/anaconda3/envs/tf-env/bin/python'


def execute_notebook(i):
    """Execute a specific Jupyter notebook using Papermill."""

    notebook_fname = f'NN_full_v3_BO_test_{i}'
    model_path_bo = os.path.join(MODEL_PATH, notebook_fname)
    print('full path:', os.path.abspath(model_path_bo))

    # Check if the directory exists, and create if not
    if not os.path.exists(model_path_bo):
        os.makedirs(model_path_bo)
        print(f"Folder '{model_path_bo}' created.")
    else:
        print(f"Folder '{model_path_bo}' already exists.")

    # Execute the notebook
    pm.execute_notebook(
        input_path=os.path.join(MASTER_PATH, f"{NOTEBOOK_MASTER}.ipynb"),
        output_path=os.path.join(model_path_bo, f"{notebook_fname}.ipynb"),
        parameters={'notebook_fname': notebook_fname,
                    # dir in respect to this batch script
                    'data_path': '../01_Dataset_Cleaned/',
                    # dir in respect to this batch script
                    'model_path': '../04_Model_Saved/'
                    },
        conda_env=CONDA_ENV
    )


def main():
    """Main function to execute notebooks in parallel."""

    start_time = time.time()

    file_nums = np.array([7, 8])
    cpu_count = os.cpu_count()
    cpu_count_use = max(1, cpu_count // len(file_nums)
                        )  # Avoid dividing by zero

    with Pool(cpu_count_use) as p:
        p.map(execute_notebook, file_nums)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {int(elapsed_time)} seconds")


if __name__ == '__main__':
    main()
