# Import necessary libraries
import pandas as pd
import numpy as np
import math


class FeatureCalculator:
    # Define a dictionary of atomic numbers for different elements
    atomic_numbers = [
        ("Li", 3), ("Be", 4), ("B", 5), ("C", 6), ("N", 7), ("Na", 11),
        ("Mg", 12), ("Al", 13), ("Si", 14), ("P", 15), ("Ca", 20), ("Sc", 21),
        ("Ti", 22), ("V", 23), ("Cr", 24), ("Mn", 25), ("Fe", 26), ("Co", 27),
        ("Ni", 28), ("Cu", 29), ("Zn", 30), ("Ga", 31), ("Ge", 32), ("Sr", 38),
        ("Y", 39), ("Zr", 40), ("Nb", 41), ("Mo", 42), ("Tc", 43), ("Ru", 44),
        ("Rh", 45), ("Pd", 46), ("Ag", 47), ("Cd", 48), ("In", 49), ("Sn", 50),
        ("Sb", 51), ("La", 57), ("Ce", 58), ("Pr", 59), ("Nd", 60), ("Pm", 61),
        ("Sm", 62), ("Eu", 63), ("Gd", 64), ("Tb", 65), ("Dy", 66), ("Ho", 67),
        ("Er", 68), ("Tm", 69), ("Yb", 70), ("Lu", 71), ("Hf", 72), ("Ta", 73),
        ("W", 74), ("Re", 75), ("Os", 76), ("Pt", 78), ("Au", 79), ("Pb", 82),
        ("Bi", 83)
    ]
    atomic_numbers_dict = {
        element: number for element, number in atomic_numbers
    }

    def __init__(self, compositions):
        """
        This constructor receives a list of compositions. Each composition is a tuple with two elements:
        - a list of chemical elements, and 
        - a corresponding list of their fractions in the composition.

        For each composition, this constructor:
        - Normalizes the fractions, making them sum up to 1. This ensures comparability across different compositions.
        - Translates element names to atomic numbers using a pre-existing dictionary (self.atomic_numbers_dict). 
        This is for further calculations that work with atomic numbers instead of element names.
        - Filters out any elements with zero fraction from 'compo_elem', 'compo_num', and 'ele_frac'. 
        This is because elements with zero fraction do not contribute to the overall properties of the composition.

        The filtered and translated compositions are then stored in self.compositions for further use.
        """
        self.compositions = [
            {
                "compo_elem": [elem for idx, elem in enumerate(compo_elem) if ele_frac[idx] > 0],
                "compo_num": [self.atomic_numbers_dict[elem] for idx, elem in enumerate(compo_elem) if ele_frac[idx] > 0],
                "ele_frac": [frac for frac in ele_frac if frac > 0],
            }
            for compo_elem, ele_frac in compositions
        ]
        # print(self.compositions)

        self.load_data()  # Load the necessary data for further computations.

    def load_data(self):
        try:
            # Load the properties data and the mixing enthalpy data from Excel files
            # We take the specific columns and rows we need and convert them to numpy arrays
            self.data1_ld = pd.read_excel(
                "FeatureExaction_properties.xlsx",
                usecols="C:H",
                skiprows=1,
                header=None,
            ).iloc[:83].values

            self.data2_ld = pd.read_excel(
                "FeatureExaction_mixingenthalpy.xlsx",
                usecols="C:CG",
                skiprows=2,
                header=None,
            ).iloc[:84].values

        except Exception as e:
            raise IOError(f"Error reading data: {e}")

    def calculate_features(self):
        # This function calculates various features for each composition
        features_list = []
        for composition in self.compositions:
            compo_elem = composition["compo_elem"]
            compo_num = composition["compo_num"]
            ele_frac = composition["ele_frac"]
            ntotal = len(compo_num)

            # Extract the necessary data for each element in the composition
            radius, tm, elec_nega, vec, bulk = (
                self.data1_ld[:, 0],
                self.data1_ld[:, 1],
                self.data1_ld[:, 2],
                self.data1_ld[:, 3],
                self.data1_ld[:, 5],
            )
            enthalpy = self.data2_ld

            ele_size, ele_temp, ele_elec_nega, VEC = [
                np.zeros(ntotal) for _ in range(4)]
            for i in range(ntotal):
                ele_size[i] = radius[compo_num[i]-1]
                ele_temp[i] = tm[compo_num[i]-1]
                ele_elec_nega[i] = elec_nega[compo_num[i]-1]
                VEC[i] = vec[compo_num[i]-1]

            # print(ele_size)

            # --------------------------------------------------------------------------------
            # Then we perform various calculations on these data to form the features
            # The calculations involve operations such as averaging, computing deviations, enthalpy, entropy, etc.
            # --------------------------------------------------------------------------------
            # Calculate the mean radius and delta
            r_mean = sum(ele_size[i] * ele_frac[i] for i in range(ntotal))
            delta = math.sqrt(
                sum(
                    ele_frac[i] * (1 - ele_size[i] / r_mean) ** 2
                    for i in range(ntotal)
                )
            )

            # Calculate the average melting temperature and its deviation
            TM = sum(ele_frac[i] * ele_temp[i] for i in range(ntotal))
            DTM = math.sqrt(
                sum(ele_frac[i] * (ele_temp[i] - TM)
                    ** 2 for i in range(ntotal))
            )

            # Calculate the average electronic negativity and its deviation
            Mean_elecnega = sum(
                ele_elec_nega[i] * ele_frac[i] for i in range(ntotal))
            D_elecnega = math.sqrt(
                sum(
                    ele_frac[i] * (ele_elec_nega[i] - Mean_elecnega) ** 2
                    for i in range(ntotal)
                )
            )

            # Calculate the average VEC and its deviation
            MVEC = sum(ele_frac[i] * VEC[i] for i in range(ntotal))
            D_VEC = math.sqrt(
                sum(ele_frac[i] * (VEC[i] - MVEC) ** 2 for i in range(ntotal))
            )

            # Calculate the total mixing enthalpy
            ME = sum(
                4
                * ele_frac[i]
                * ele_frac[j]
                * enthalpy[compo_num[i]-1][compo_num[j]-1]
                for i in range(ntotal - 1)
                for j in range(i + 1, ntotal)
            )

            # Calculate the deviation of mixing enthalpy
            DME = math.sqrt(
                sum(
                    ele_frac[i]
                    * ele_frac[j]
                    * (enthalpy[compo_num[i]-1][compo_num[j]-1] - ME) ** 2
                    for i in range(ntotal - 1)
                    for j in range(i + 1, ntotal)
                )
            )

            # Calculate the ideal mixing entropy
            Sid = sum(-ele_frac[i] * math.log(ele_frac[i])
                      for i in range(ntotal))

            # Calculate average bulk modulus B and its standard deviation
            B = [self.data1_ld[compo_num[i]-1, 5] for i in range(ntotal)]
            B_ave = sum(ele_frac[i] * B[i] for i in range(ntotal))

            D_Bulk = math.sqrt(
                sum(ele_frac[i] * (B[i] - B_ave) ** 2 for i in range(ntotal))
            )

            # Collect all calculated features and add to the features list
            features = np.array([
                r_mean,
                delta,
                TM,
                DTM,
                ME,
                DME,
                Sid,
                Mean_elecnega,
                D_elecnega,
                MVEC,
                D_VEC,
                B_ave * 1e9,
                D_Bulk,
            ])
            features_list.append(features)

        # Return the list of features for each composition
        return features_list
