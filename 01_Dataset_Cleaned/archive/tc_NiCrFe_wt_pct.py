from tc_python import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))
cache_fname = os.path.basename(__file__) + "_cache"
output_fname = os.path.splitext(os.path.basename(__file__))[0]

print("Cache Filename: ", cache_fname)
print("Output Filename: ", output_fname)
print("Current Dir: ", current_directory)
 
# Setup thermodynamic calculation
with TCPython() as start:
    calculation = (
        start
        .set_cache_folder(os.path.join(current_directory, cache_fname))
        .set_ges_version(5)
        .select_database_and_elements("tchea4", ["Fe", "Ni", "Cr"])
        .get_system()
        .with_batch_equilibrium_calculation()
        .set_condition("W(Ni)", 1E-2)
        .set_condition("W(Cr)", 1E-2)
        .set_condition("T", 800+273.15)
        .disable_global_minimization()
    )

    # Generate condition combinations for equilibrium calculations
    k = 20
    list_of_conditions = [(("W(Ni)", w_ni), ("W(Cr)", w_cr), ("T", tk))
                          for w_ni in np.linspace(0.1e-2, 0.95, k)
                          for w_cr in np.linspace(0.1e-2, 0.95, k)
                          for tk in np.arange(900 + 273.15, 1300 + 273.15, 50)
                          if w_ni + w_cr <= 1]

    calculation.set_conditions_for_equilibria(list_of_conditions)
    results = calculation.calculate(["np(FCC_L12)"], 100)
    list_np_FCC_L12 = results.get_values_of('np(FCC_L12)')

# ====== postprocessing of tc calculation ======

df = pd.DataFrame(columns=['Ni', 'Cr', 'T', 'np(FCC_L12)'])

# Convert conditions and results to DataFrame
df = pd.DataFrame({
    'Ni': [dict(conditions)['W(Ni)']*100 for conditions in list_of_conditions],
    'Cr': [dict(conditions)['W(Cr)']*100 for conditions in list_of_conditions],
    'T': [dict(conditions)['T'] for conditions in list_of_conditions],
    'np(FCC_L12)': list_np_FCC_L12
})


# --- data showing FCC possibilties
# Group by 'Ni' and 'Cr' and filter groups with any 'np(FCC_L12)' value > 0.99
filtered_groups_FCC_y = df.groupby(['Ni', 'Cr']).filter(
    lambda x: (x['np(FCC_L12)'] > 0.99).any())
# Get unique 'Ni' and 'Cr' combinations
result_FCC_y = filtered_groups_FCC_y[['Ni', 'Cr']
                                     ].drop_duplicates().reset_index(drop=True)

# Calculate the 'Fe' column values
result_FCC_y['Fe'] = 100 - result_FCC_y['Ni'] - result_FCC_y['Cr']
print(result_FCC_y.head(3))
df_output_fname = output_fname+".xlsx"
result_FCC_y.to_excel(
    os.path.join(current_directory, df_output_fname), index=False)

# --- data showing no FCC possibilties
# Get unique combinations where 'np(FCC_L12)' < 1 for each group
filtered_groups_FCC_n = df.groupby(['Ni', 'Cr']).filter(
    lambda group: group['np(FCC_L12)'].max() < 1)
result_FCC_n = filtered_groups_FCC_n[['Ni', 'Cr']
                                     ].drop_duplicates().reset_index(drop=True)

# Plotting
plt.scatter(result_FCC_y['Ni'], result_FCC_y['Cr'],
            marker='o', color='blue', label='Y values')
plt.scatter(result_FCC_n['Ni'], result_FCC_n['Cr'],
            marker='o', color='red', label='N values')
plt.title('Scatter plot of Ni vs Cr')
plt.xlabel('W(Ni)')
plt.ylabel('W(Cr)')
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.axis('square')
plt_output_fname = output_fname+".png"
plt.savefig(os.path.join(current_directory, plt_output_fname))
