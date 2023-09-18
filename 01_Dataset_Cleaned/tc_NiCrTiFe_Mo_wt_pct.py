from tc_python import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool

current_directory = os.path.dirname(os.path.abspath(__file__))
cache_fname = os.path.basename(__file__) + "_cache"
output_fname = os.path.splitext(os.path.basename(__file__))[0]

print("Cache Filename: ", cache_fname)
print("Output Filename: ", output_fname)
print("Current Dir: ", current_directory)


def tc_calculation(tk):
    # Setup thermodynamic calculation
    with TCPython() as start:
        calculation = (
            start
            .set_cache_folder(os.path.join(current_directory, cache_fname))
            .set_ges_version(5)
            .select_database_and_elements("tchea4", ["Ni", "Cr", "Mo", "Ti", "Fe"])
            .get_system()
            .with_batch_equilibrium_calculation()
            .set_condition("W(Ni)", 95E-2)
            .set_condition("W(Cr)", 0.1E-2)
            .set_condition("W(Mo)", 0.1E-2)
            .set_condition("W(Ti)", 0.1E-2)
            .set_condition("T", 1000+273.15)
            .disable_global_minimization()
        )

        # Generate condition combinations for equilibrium calculations
        k = 40
        list_of_conditions = [(("W(Ni)", w_ni), ("W(Cr)", w_cr), ("W(Mo)", 0.1E-2), ("W(Ti)", w_ti), ("T", tk))
                              for w_ni in np.linspace(0.1E-2, 0.95, k)
                              for w_cr in np.linspace(0.1E-2, 0.95, k)
                              for w_ti in np.linspace(0.1E-2, 0.95, k)
                              if w_ni + w_cr + 0.1E-2 + w_ti <= 1]

        calculation.set_conditions_for_equilibria(list_of_conditions)

        results = calculation.calculate(
            ["np(FCC_L12)", "np(FCC_L12#1)", "np(FCC_L12#2)"], 100)

        return (
            list_of_conditions,
            results.get_values_of('np(FCC_L12)'),
            results.get_values_of('np(FCC_L12#1)'),
            results.get_values_of('np(FCC_L12#2)')
        )


# Parallelize the computation over different tk values
tk_values = np.arange(900+273.15, 1300+273.15, 50)
with Pool() as pool:
    all_results = pool.map(tc_calculation, tk_values)

# print(len(all_results[0]))

# Merge results from different processes
list_of_conditions = [res[0] for res in all_results]
list_np_FCC_L12 = [res[1] for res in all_results]
list_np_FCC_L12_1 = [res[2] for res in all_results]
list_np_FCC_L12_2 = [res[3] for res in all_results]

print(len(list_of_conditions))

# Flattening the lists
list_of_conditions = [
    item for sublist in list_of_conditions for item in sublist]
list_np_FCC_L12 = [item for sublist in list_np_FCC_L12 for item in sublist]
list_np_FCC_L12_1 = [item for sublist in list_np_FCC_L12_1 for item in sublist]
list_np_FCC_L12_2 = [item for sublist in list_np_FCC_L12_2 for item in sublist]
print(len(list_of_conditions))


# ====== postprocessing of tc calculation ======
list_np_FCC_L12_merge = [max(a, b, c) for a, b, c in zip(
    list_np_FCC_L12, list_np_FCC_L12_1, list_np_FCC_L12_2)]

df = pd.DataFrame(columns=['Ni', 'Cr', 'Mo', 'Ti',
                  'T', 'np(FCC_L12)', 'np(FCC_L12#1)', 'np(FCC_L12#2)',
                           'np(FCC_L12)_merge'])

# Convert conditions and results to DataFrame
df = pd.DataFrame({
    'Ni': [dict(conditions)['W(Ni)']*100 for conditions in list_of_conditions],
    'Cr': [dict(conditions)['W(Cr)']*100 for conditions in list_of_conditions],
    'Mo': [dict(conditions)['W(Mo)']*100 for conditions in list_of_conditions],
    'Ti': [dict(conditions)['W(Ti)']*100 for conditions in list_of_conditions],
    'T': [dict(conditions)['T'] for conditions in list_of_conditions],
    'np(FCC_L12)': list_np_FCC_L12,
    'np(FCC_L12#1)': list_np_FCC_L12_1,
    'np(FCC_L12#2)': list_np_FCC_L12_2,
    'np(FCC_L12)_merge': list_np_FCC_L12_merge
})

df.to_excel(os.path.join(current_directory,
            "tc_full_df_check.xlsx"), index=False)

# --- data showing FCC possibilties
# Group by 'Ni' and 'Cr' and filter groups with any 'np(FCC_L12)' value > 0.99
filtered_groups_FCC_y = df.groupby(['Ni', 'Cr', 'Mo', 'Ti']).filter(
    lambda x: (x['np(FCC_L12)_merge'] > 0.99).any())
# Get unique 'Ni' and 'Cr' combinations
result_FCC_y = filtered_groups_FCC_y[['Ni', 'Cr', 'Mo', 'Ti']
                                     ].drop_duplicates().reset_index(drop=True)

# Calculate the 'Fe' column values
result_FCC_y['Fe'] = 100 - result_FCC_y['Ni'] - \
    result_FCC_y['Cr'] - result_FCC_y['Mo'] - result_FCC_y['Ti']
print(result_FCC_y.head(3))
# df_output_fname = output_fname+".xlsx"
df_output_fname = "MultiTaskModel_NiCrTiFe_Mo_TC_wt_pct.xlsx"

result_FCC_y.to_excel(
    os.path.join(current_directory, df_output_fname), index=False)

# --- data showing NO FCC possibilties
# Get unique combinations where 'np(FCC_L12)_merge' < 1 for each group
filtered_groups_FCC_n = df.groupby(['Ni', 'Cr', 'Mo', 'Ti']).filter(
    lambda group: group['np(FCC_L12)_merge'].max() < 0.99)
result_FCC_n = filtered_groups_FCC_n[['Ni', 'Cr', 'Mo', 'Ti']
                                     ].drop_duplicates().reset_index(drop=True)
result_FCC_n['Fe'] = 100 - result_FCC_n['Ni'] - \
    result_FCC_n['Cr'] - result_FCC_n['Mo'] - result_FCC_n['Ti']

# Plotting
plt.scatter(result_FCC_y['Ni'], result_FCC_y['Fe'],
            marker='o', color='blue', label='Y values')
plt.scatter(result_FCC_n['Ni'], result_FCC_n['Fe'],
            marker='o', color='red', label='N values')
plt.title('Scatter plot of Ni vs Fe')
plt.xlabel('W(Ni)')
plt.ylabel('W(Fe)')
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.axis('square')
plt_output_fname = output_fname+".png"
plt.savefig(os.path.join(current_directory, plt_output_fname))
