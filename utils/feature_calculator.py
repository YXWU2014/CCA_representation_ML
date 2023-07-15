# Import necessary libraries
import pandas as pd
import numpy as np
import math
import os


class FeatureCalculator:

    """
    A FeatureCalculator is designed to compute various chemical composition features, 
    necessary for material property predictions. It uses the elemental compositions of the
    material and the atomic weights and atomic numbers of the elements.

    Attributes:
        compositions (list of dict): Each dictionary represents a single composition with keys:
            'compo_elem' - the list of elements in the composition,
            'compo_num' - the corresponding atomic numbers,
            'compo_weight' - the atomic weights of the elements,
            'ele_wt_frac' - the weight fractions of the elements in the composition.
    """

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

    # Define a dictionary of atomic weight for different elements (in g/mol)
    atomic_weights = [
        ("Li", 6.941), ("Be", 9.012), ("B", 10.811), ("C",
                                                      12.011), ("N", 14.007), ("Na", 22.990),
        ("Mg", 24.305), ("Al", 26.982), ("Si", 28.086), ("P",
                                                         30.974), ("Ca", 40.078), ("Sc", 44.956),
        ("Ti", 47.867), ("V", 50.942), ("Cr", 51.996), ("Mn",
                                                        54.938), ("Fe", 55.845), ("Co", 58.933),
        ("Ni", 58.693), ("Cu", 63.546), ("Zn", 65.38), ("Ga",
                                                        69.723), ("Ge", 72.63), ("Sr", 87.62),
        ("Y", 88.906), ("Zr", 91.224), ("Nb",
                                        92.906), ("Mo", 95.95), ("Tc", 98), ("Ru", 101.07),
        ("Rh", 102.906), ("Pd", 106.42), ("Ag", 107.868), ("Cd",
                                                           112.414), ("In", 114.818), ("Sn", 118.71),
        ("Sb", 121.76), ("La", 138.905), ("Ce", 140.116), ("Pr",
                                                           140.908), ("Nd", 144.242), ("Pm", 145),
        ("Sm", 150.36), ("Eu", 151.964), ("Gd", 157.25), ("Tb",
                                                          158.925), ("Dy", 162.5), ("Ho", 164.93),
        ("Er", 167.259), ("Tm", 168.934), ("Yb", 173.045), ("Lu",
                                                            174.967), ("Hf", 178.49), ("Ta", 180.948),
        ("W", 183.84), ("Re", 186.207), ("Os", 190.23), ("Pt",
                                                         195.084), ("Au", 196.967), ("Pb", 207.2),
        ("Bi", 208.98)
    ]

    atomic_numbers_dict = {
        element: number for element, number in atomic_numbers
    }

    atomic_weights_dict = {
        element: weight for element, weight in atomic_weights
    }

    def __init__(self, compositions):
        """
        Constructs the FeatureCalculator with a list of elemental compositions.

        Args:
            compositions (list of tuples): Each tuple contains a list of elements 
            and a corresponding list of their weight fractions in the composition.
        """
        self.compositions = [
            {
                "compo_elem": [elem for idx, elem in enumerate(compo_elem) if ele_wt_frac[idx] > 0],
                "compo_num": [self.atomic_numbers_dict[elem] for idx, elem in enumerate(compo_elem) if ele_wt_frac[idx] > 0],
                "compo_weight": [self.atomic_weights_dict[elem] for idx, elem in enumerate(compo_elem) if ele_wt_frac[idx] > 0],
                "ele_wt_frac": [frac for frac in ele_wt_frac if frac > 0],
            }
            for compo_elem, ele_wt_frac in compositions
        ]
        # print(self.compositions)

        self.load_data()  # Load the necessary data for further computations.

    def load_data(self):
        """
        Loads the properties data and the mixing enthalpy data from Excel files,
        and converts them to numpy arrays. 
        """
        # Assuming that this script is located in the utils folder
        utils_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the paths to the Excel files
        excel_file_1_path = os.path.join(
            utils_dir, 'FeatureExaction_properties.xlsx')
        excel_file_2_path = os.path.join(
            utils_dir, 'FeatureExaction_mixingenthalpy.xlsx')

        try:
            self.data1_ld = pd.read_excel(
                excel_file_1_path,
                usecols="C:H",
                skiprows=1,
                header=None,
            ).iloc[:83].values

            self.data2_ld = pd.read_excel(
                excel_file_2_path,
                usecols="C:CG",
                skiprows=2,
                header=None,
            ).iloc[:84].values

        except Exception as e:
            raise IOError(f"Error reading data: {e}")

    def calculate_features(self):
        """
        Computes various features for each composition, such as mean radius, melting point, 
        atomic fractions, electronic negativity, enthalpy, entropy, and others.

        Returns:
            list of np.array: A list of feature vectors, one for each composition.
        """

        features_list = []
        for composition in self.compositions:

            compo_elem = composition["compo_elem"]
            compo_weight = composition["compo_weight"]
            compo_num = composition["compo_num"]
            ele_wt_frac = composition["ele_wt_frac"]
            ntotal = len(compo_num)

            # print(compo_elem)
            # print(compo_weight)
            # print(compo_num)
            # print(ele_wt_frac)

            # --------------------------------------------------------------------------------
            # Calculate the atomic fractions of each element in the composition
            # --------------------------------------------------------------------------------
            total_ele_at_frac = sum(
                np.array(ele_wt_frac)/np.array(compo_weight))

            # now let's convert the weight fractions to atomic fractions
            ele_at_frac = np.zeros(ntotal)
            for i in range(ntotal):
                ele_at_frac[i] = ele_wt_frac[i]/compo_weight[i] / \
                    total_ele_at_frac  # atomic fraction

            # print(ele_at_frac)

            # --------------------------------------------------------------------------------
            # Extract the necessary data for each element in the composition
            # --------------------------------------------------------------------------------
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
            r_mean = sum(ele_size[i] * ele_at_frac[i] for i in range(ntotal))
            delta = math.sqrt(
                sum(
                    ele_at_frac[i] * (1 - ele_size[i] / r_mean) ** 2
                    for i in range(ntotal)
                )
            )

            # Calculate the average melting temperature and its deviation
            TM = sum(ele_at_frac[i] * ele_temp[i] for i in range(ntotal))
            DTM = math.sqrt(
                sum(ele_at_frac[i] * (ele_temp[i] - TM)
                    ** 2 for i in range(ntotal))
            )

            # Calculate the average electronic negativity and its deviation
            Mean_elecnega = sum(
                ele_elec_nega[i] * ele_at_frac[i] for i in range(ntotal))
            D_elecnega = math.sqrt(
                sum(
                    ele_at_frac[i] * (ele_elec_nega[i] - Mean_elecnega) ** 2
                    for i in range(ntotal)
                )
            )

            # Calculate the average VEC and its deviation
            MVEC = sum(ele_at_frac[i] * VEC[i] for i in range(ntotal))
            D_VEC = math.sqrt(
                sum(ele_at_frac[i] * (VEC[i] - MVEC)
                    ** 2 for i in range(ntotal))
            )

            # Calculate the total mixing enthalpy
            ME = sum(
                4
                * ele_at_frac[i]
                * ele_at_frac[j]
                * enthalpy[compo_num[i]-1][compo_num[j]-1]
                for i in range(ntotal - 1)
                for j in range(i + 1, ntotal)
            )

            # Calculate the deviation of mixing enthalpy
            DME = math.sqrt(
                sum(
                    ele_at_frac[i]
                    * ele_at_frac[j]
                    * (enthalpy[compo_num[i]-1][compo_num[j]-1] - ME) ** 2
                    for i in range(ntotal - 1)
                    for j in range(i + 1, ntotal)
                )
            )

            # Calculate the ideal mixing entropy
            Sid = sum(-ele_at_frac[i] * math.log(ele_at_frac[i])
                      for i in range(ntotal))

            # Calculate average bulk modulus B and its standard deviation
            B = [self.data1_ld[compo_num[i]-1, 5] for i in range(ntotal)]
            B_ave = sum(ele_at_frac[i] * B[i] for i in range(ntotal))

            D_Bulk = math.sqrt(
                sum(ele_at_frac[i] * (B[i] - B_ave)
                    ** 2 for i in range(ntotal))
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
