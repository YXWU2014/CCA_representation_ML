import papermill as pm
import numpy as np
from multiprocessing import Pool
import time
import os

# ------------------------------------------------------

notebook_master = 'NN_full_RepeatedKFold_v3_BO_master'

def execute_notebook(i):
    
    input_fname  = notebook_master
    output_fname = f'NN_full_RepeatedKFold_v3_BO_{i}'
    

    model_path    = './Model_Saved/'
    model_path_bo = f'./{model_path}/{output_fname}/'
    if not os.path.exists(model_path_bo):
        os.makedirs(model_path_bo)
        print(f"Folder '{model_path_bo}' created.")
    else:
        print(f"Folder '{model_path_bo}' already exists.")

    pm.execute_notebook(
        input_path  = input_fname + '.ipynb',
        output_path = model_path_bo + output_fname + '.ipynb',
        parameters  = {'notebook_fname': output_fname},
        conda_env   = '/nethome/yuxiang.wu/anaconda3/envs/tf-env/bin/python'
    )

# ------------------------------------------------------

start_time = time.time()

if __name__ == '__main__':
    
    file_nums = np.array([7, 8])
    cpu_count = os.cpu_count()
    cpu_count_use = cpu_count // len(file_nums)

    with Pool(cpu_count_use) as p:
        p.map(execute_notebook, file_nums)

elapsed_time = time.time() - start_time
print(f"Time taken: {int(elapsed_time)} seconds")

# ------------------------------------------------------
