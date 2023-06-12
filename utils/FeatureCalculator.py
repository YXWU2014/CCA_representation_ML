import pandas as pd
import numpy as np
import math


class FeatureCalculator:
    def __init__(self, compo_elem, ele_frac):
        self.compo_elem = compo_elem

        # create a dictionary of elements and their atomic numbers
        atomic_numbers = [
            ("Li", 3),
            ("Be", 4),
            ("B", 5),
            ("C", 6),
            ("N", 7),
            ("Na", 11),
            ("Mg", 12),
            ("Al", 13),
            ("Si", 14),
            ("P", 15),
            ("Ca", 20),
            ("Sc", 21),
            ("Ti", 22),
            ("V", 23),
            ("Cr", 24),
            ("Mn", 25),
            ("Fe", 26),
            ("Co", 27),
            ("Ni", 28),
            ("Cu", 29),
            ("Zn", 30),
            ("Ga", 31),
            ("Ge", 32),
            ("Sr", 38),
            ("Y", 39),
            ("Zr", 40),
            ("Nb", 41),
            ("Mo", 42),
            ("Tc", 43),
            ("Ru", 44),
            ("Rh", 45),
            ("Pd", 46),
            ("Ag", 47),
            ("Cd", 48),
            ("In", 49),
            ("Sn", 50),
            ("Sb", 51),
            ("La", 57),
            ("Ce", 58),
            ("Pr", 59),
            ("Nd", 60),
            ("Pm", 61),
            ("Sm", 62),
            ("Eu", 63),
            ("Gd", 64),
            ("Tb", 65),
            ("Dy", 66),
            ("Ho", 67),
            ("Er", 68),
            ("Tm", 69),
            ("Yb", 70),
            ("Lu", 71),
            ("Hf", 72),
            ("Ta", 73),
            ("W", 74),
            ("Re", 75),
            ("Os", 76),
            ("Pt", 78),
            ("Au", 79),
            ("Pb", 82),
            ("Bi", 83),
        ]
        atomic_numbers_dict = {element: number for element, number in atomic_numbers}

        self.compo_num = [atomic_numbers_dict[elem] for elem in self.compo_elem]

        self.ele_frac = np.array(ele_frac)
        self.ele_frac /= self.ele_frac.sum()

    def load_data(self):
        try:
            # Read the database
            self.data1_ld = pd.read_excel(
                "FeatureExaction_properties.xlsx",
                usecols="C:H",
                skiprows=1,
                header=None,
            ).values
            self.data2_ld = pd.read_excel(
                "FeatureExaction_mixingenthalpy.xlsx",
                usecols="C:CG",
                skiprows=2,
                header=None,
            ).values
        except Exception as e:
            print(f"Error reading data: {e}")
            return None

    def calculate_features(self):
        self.load_data()

        # Assign data columns
        radius, tm, elec_nega, vec, bulk = (
            self.data1_ld[:, 0],
            self.data1_ld[:, 1],
            self.data1_ld[:, 2],
            self.data1_ld[:, 3],
            self.data1_ld[:, 5],
        )
        enthalpy = self.data2_ld

        ntotal = len(self.compo_num)

        ele_size, ele_temp, ele_elec_nega, VEC = [np.zeros(ntotal) for _ in range(4)]

        for i in range(ntotal):
            ele_size[i] = radius[self.compo_num[i]]
            ele_temp[i] = tm[self.compo_num[i]]
            ele_elec_nega[i] = elec_nega[self.compo_num[i]]
            VEC[i] = vec[self.compo_num[i]]

        # Calculate the mean radius and delta
        r_mean = sum(ele_size[i] * self.ele_frac[i] for i in range(ntotal))
        delta = math.sqrt(
            sum(
                self.ele_frac[i] * (1 - ele_size[i] / r_mean) ** 2
                for i in range(ntotal)
            )
        )

        # Calculate the average melting temperature and its deviation
        TM = sum(self.ele_frac[i] * ele_temp[i] for i in range(ntotal))
        DTM = math.sqrt(
            sum(self.ele_frac[i] * (ele_temp[i] - TM) ** 2 for i in range(ntotal))
        )

        # Calculate the average electronic negativity and its deviation
        Mean_elecnega = sum(ele_elec_nega[i] * self.ele_frac[i] for i in range(ntotal))
        D_elecnega = math.sqrt(
            sum(
                self.ele_frac[i] * (ele_elec_nega[i] - Mean_elecnega) ** 2
                for i in range(ntotal)
            )
        )

        # Calculate the average VEC and its deviation
        MVEC = sum(self.ele_frac[i] * VEC[i] for i in range(ntotal))
        D_VEC = math.sqrt(
            sum(self.ele_frac[i] * (VEC[i] - MVEC) ** 2 for i in range(ntotal))
        )

        # Calculate the total mixing enthalpy
        ME = sum(
            4
            * self.ele_frac[i]
            * self.ele_frac[j]
            * enthalpy[self.compo_num[i]][self.compo_num[j]]
            for i in range(ntotal - 1)
            for j in range(i + 1, ntotal)
        )

        # Calculate the deviation of mixing enthalpy
        DME = math.sqrt(
            sum(
                self.ele_frac[i]
                * self.ele_frac[j]
                * (enthalpy[self.compo_num[i]][self.compo_num[j]] - ME) ** 2
                for i in range(ntotal - 1)
                for j in range(i + 1, ntotal)
            )
        )

        # Calculate the ideal mixing entropy
        Sid = sum(-self.ele_frac[i] * math.log(self.ele_frac[i]) for i in range(ntotal))

        # Calculate average bulk modulus B and its standard deviation
        B = [self.data1_ld[self.compo_num[i], 5] for i in range(ntotal)]
        B_ave = sum(self.ele_frac[i] * B[i] for i in range(ntotal))
        D_Bulk = math.sqrt(
            sum(self.ele_frac[i] * (B[i] - B_ave) ** 2 for i in range(ntotal))
        )

        # Collecting all the features for machine learning
        features = np.array(
            [
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
            ]
        )

        return features
