# %%
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))

from utils.FeatureCalculator import FeatureCalculator

compo_elem = ["Ni", "Cr", "Mo", "Ti", "Fe"]  # Your data
ele_frac = np.array([43.8, 38.3, 2.44, 1.04, 14.5])  # Your data

# Make a dictionary with keys of compo_elem and values of ele_frac, 
# excluding pairs where the fraction is 0, all in one line
ele_frac_dict = {elem: frac for elem, frac in zip(compo_elem, ele_frac) if frac != 0}

# # Normalize ele_frac values
# ele_frac_norm = np.array(list(ele_frac_dict.values()))
# ele_frac_norm /= ele_frac_norm.sum()

# compositions = [(list(ele_frac_dict.keys()), list(ele_frac_norm))]

compositions = [(list(ele_frac_dict.keys()), list(ele_frac_dict.values()))]


print(compositions)

# use FeatureCalculator and calculate features
calculator = FeatureCalculator(compositions)
features = calculator.calculate_features()

# print(features)




# # %%
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from utils.FeatureCalculator import FeatureCalculator
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count

# T_fname = "MultiTaskModel_NiCrMoTiFe_KW131_at_pct.xlsx"
# Features = [
#     "a",
#     "delta_a",
#     "Tm",
#     "sigma_Tm",
#     "Hmix",
#     "sigma_Hmix",
#     "ideal_S",
#     "elec_nega",
#     "sigma_elec_nega",
#     "VEC",
#     "sigma_VEC",
#     "bulk_modulus",
#     "sigma_bulk_modulus",
# ]

# # Load data
# T = pd.read_excel(T_fname)
# display(T.head())

# # Calculate features
# compo_elem = ["Ni", "Cr", "Mo", "Ti", "Fe"]
# ele_frac_all = T[compo_elem].values/100

 

# # create a comp_elem list for each composition
# compo_elem_all = []
# for i in range(len(ele_frac_all)):
#     compo_elem_all.append(compo_elem)

# # print(compo_elem_all[0:5])


# compositions = [(compo_elem, ele_frac) for compo_elem, ele_frac in zip(compo_elem_all, ele_frac_all)]   




 
# # print(compositions)
# # use FeatureCalculator and calculate features
# calculator = FeatureCalculator(compositions)
# features = calculator.calculate_features()

# # print(features)

# df_features = pd.DataFrame(features, columns=Features)
# display(df_features.head())

# # # Function to prepare the composition data
# # def prepare_compositions(compo_elem, ele_frac):
# #     compositions = [
# #         {
# #             "compo_elem": compo_elem,
# #             "ele_frac": ele_frac.tolist(),
# #         }
# #     ]
# #     return compositions

# # # Function to calculate features for a single composition
# # def calc_features(ele_frac):
# #     compositions = prepare_compositions(compo_elem, ele_frac)
# #     # Use FeatureCalculator and calculate features
# #     calculator = FeatureCalculator(compositions)
# #     features = calculator.calculate_features()[0]  # Extract single feature array as calculate_features returns a list of feature arrays
# #     return features

# # # Initialize a multiprocessing pool
# # with Pool(cpu_count()) as pool:
# #     # Calculate features using multiple processes
# #     all_features = list(tqdm(pool.imap(calc_features, ele_frac_all), total=len(ele_frac_all), desc="Calculating features"))

# # # Convert the list of features to a DataFrame for easier manipulation
# # all_features_df = pd.DataFrame(all_features, columns=Features_Sel)
# # display(all_features_df.head())


# # %%
# # formula = T["FORMULA"]
# # alloy_num = len(formula)

# # # Extract component_num, components, and fractions
# # components = []
# # fractions = []
# # component_num = []

# # for i in range(alloy_num):
# #     formula_temp = formula[i]
# #     formula_temp_split = re.split("\s+", formula_temp)
# #     component_num.append(len(formula_temp_split))
# #     components.append(" ".join([re.sub(r"\d+", "", s) for s in formula_temp_split]))
# #     fractions.append(" ".join([re.sub(r"\D", "", s) for s in formula_temp_split]))

# # # Calculate features
# # features = np.zeros((alloy_num, 13))
# # components_split = []
# # fractions_split = []

# # for i in range(alloy_num):
# #     components_split.append(re.split("\s+", components[i]))
# #     fractions_split.append([float(x) for x in re.split("\s+", fractions[i])])

# #     comp = [eval(s) for s in components_split[i]]
# #     ele_frac = fractions_split[i]

# #     calculator = FeatureCalculator(comp, ele_frac)
# #     features_temp = calculator.calculate_features()
# #     features[i, :] = features_temp

# # # Write to dataframe
# # T_features = pd.DataFrame(
# #     features,
# #     columns=[
# #         "a",
# #         "delta_a",
# #         "Tm",
# #         "sigma_Tm",
# #         "Hmix",
# #         "sigma_Hmix",
# #         "ideal_S",
# #         "elec_nega",
# #         "sigma_elec_nega",
# #         "VEC",
# #         "sigma_VEC",
# #         "bulk_modulus",
# #         "sigma_bulk_modulus",
# #     ],
# # )

# # T_features.to_excel(T_fname, index=False, columns=Features_Sel)

# # # Pearson's correlation
# # corr = T_features.corr()
# # plt.figure(figsize=(12, 12))
# # sns.heatmap(corr, annot=True)
# # plt.title("Correlation Matrix for hardness Features")
# # plt.show()



# %%
