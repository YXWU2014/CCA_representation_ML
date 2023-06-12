# feature_calculator.py
import pandas as pd
import numpy as np
import math


def calculate_features(comp, ele_frac):
    """Calculates various features from the provided data.

    Parameters
    ----------
    comp : list of int
        The composition elements.
    ele_frac : list of float
        The element fractions.

    Returns
    -------
    np.array
        The calculated features.
    """
    try:
        # Read the database
        data1_ld = pd.read_excel(
            "FeatureExaction_properties.xlsx", usecols="C:H", skiprows=1, header=None
        ).values
        data2_ld = pd.read_excel(
            "FeatureExaction_mixingenthalpy.xlsx",
            usecols="C:CG",
            skiprows=2,
            header=None,
        ).values
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

    # Assign data columns
    radius, tm, elec_nega, vec, bulk = (
        data1_ld[:, 0],
        data1_ld[:, 1],
        data1_ld[:, 2],
        data1_ld[:, 3],
        data1_ld[:, 5],
    )
    enthalpy = data2_ld

    ntotal = len(comp)

    ele_size, ele_temp, ele_elec_nega, VEC = [np.zeros(ntotal) for _ in range(4)]

    for i in range(ntotal):
        ele_size[i] = radius[comp[i]]
        ele_temp[i] = tm[comp[i]]
        ele_elec_nega[i] = elec_nega[comp[i]]
        VEC[i] = vec[comp[i]]

    # Calculate the fractions of each element
    ele_frac /= sum(ele_frac)

    # Calculate the mean radius and delta
    r_mean = sum(ele_size[i] * ele_frac[i] for i in range(ntotal))
    delta = math.sqrt(
        sum(ele_frac[i] * (1 - ele_size[i] / r_mean) ** 2 for i in range(ntotal))
    )

    # Calculate the average melting temperature and its deviation
    TM = sum(ele_frac[i] * ele_temp[i] for i in range(ntotal))
    DTM = math.sqrt(sum(ele_frac[i] * (ele_temp[i] - TM) ** 2 for i in range(ntotal)))

    # Calculate the average electronic negativity and its deviation
    Mean_elecnega = sum(ele_elec_nega[i] * ele_frac[i] for i in range(ntotal))
    D_elecnega = math.sqrt(
        sum(
            ele_frac[i] * (ele_elec_nega[i] - Mean_elecnega) ** 2 for i in range(ntotal)
        )
    )

    # Calculate the average VEC and its deviation
    MVEC = sum(ele_frac[i] * VEC[i] for i in range(ntotal))
    D_VEC = math.sqrt(sum(ele_frac[i] * (VEC[i] - MVEC) ** 2 for i in range(ntotal)))

    # Calculate the total mixing enthalpy
    ME = sum(
        4 * ele_frac[i] * ele_frac[j] * enthalpy[comp[i]][comp[j]]
        for i in range(ntotal - 1)
        for j in range(i + 1, ntotal)
    )

    # Calculate the deviation of mixing enthalpy
    DME = math.sqrt(
        sum(
            ele_frac[i] * ele_frac[j] * (enthalpy[comp[i]][comp[j]] - ME) ** 2
            for i in range(ntotal - 1)
            for j in range(i + 1, ntotal)
        )
    )

    # Calculate the ideal mixing entropy
    Sid = sum(-ele_frac[i] * math.log(ele_frac[i]) for i in range(ntotal))

    # Calculate average bulk modulus B and its standard deviation
    B = [data1_ld[comp[i], 5] for i in range(ntotal)]
    B_ave = sum(ele_frac[i] * B[i] for i in range(ntotal))
    D_Bulk = math.sqrt(sum(ele_frac[i] * (B[i] - B_ave) ** 2 for i in range(ntotal)))

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
