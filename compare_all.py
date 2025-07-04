import numpy as np
import graph_operations as go
import pandas as pd
import matplotlib.pyplot as plt
import time
np.set_printoptions(linewidth=100)

# This script compares topological properties for two groups of subjects,
# loading them from a results folder. Image plots are generated for each property 
# and scale, and statistical tests are performed to compare the groups.

DRAW = True
save_figures = True
COMP = False
n_permutations = 100000
folder = "results/"
group1 = ['01_g4s01.txt', '02_g4s05.txt', '03_g4s11.txt', '04_g5s01.txt', '05_g5s02.txt', '06_g5s06.txt',
         '07_g5s08.txt', '08_g5s10.txt', '09_g5s12.txt', '10_g5s13.txt', '11_g6s01.txt', '12_g6s04.txt',
         '13_g6s06.txt', '14_g6s08.txt', '15_g6s09.txt', '16_g6s13.txt', '17_g6s14.txt', '18_g7c01.txt',
         '19_g7c04.txt']
group2 = ['20_g7s01.txt', '21_g7s02.txt', '22_g7s03.txt', '23_g7s05.txt', '24_g7s06.txt', '25_g7s08.txt',
         '26_g7s09.txt', '27_g7s10.txt', '28_g7s11.txt', '29_g7s12.txt', '30_g7s13.txt', '31_g7s14.txt',
         '32_g7s16.txt', '33_g7s17.txt', '34_g7s18.txt', '35_g7s19.txt', '36_g7s22.txt', '37_g7s23.txt', 
         '38_g7s24.txt', '39_g7s25.txt', '40_g7s26.txt', '41_g7s28.txt', '42_g7s29.txt', '43_g7s30.txt']

group = group1 + group2

# Diagnostic division
diagnostic = np.array(19*[0] + 24*[1])  # 19 from group1, 21 from group2

# Gender division
sex_list = [
    'm', 'm', 'f', 'f', 'f', 'm', 'm', 'f', 'm', 'm', 'f', 'm', 'f', 'f', 'm',
    'f', 'f', 'm', 'm', 'm', 'm', 'm', 'f', 'f', 'm', 'm', 'm', 'f', 'm', 'm',
    'm', 'm', 'm', 'm', 'f', 'm', 'm', 'm', 'f', 'm', 'm', 'm', 'f'
]
binary_sex = np.array([0 if s == 'm' else 1 for s in sex_list])

# Age division
age_list = [
    13, 13, 13, 14, 15, 14, 15, 15, 14, 14,
    16, 16, 17, 16, 17, 16, 17, 16, 17, 16,
    14, 17, 15, 16, 14, 13, 16, 13, 16, 16,
    16, 16, 17, 13, 17, 17, 13, 16, 14, 15,
    13, 14, 15
]
binary_age = np.array([1 if age >= 16 else 0 for age in age_list])

values = go.load_group_results(group, folder)
og_values = values.copy()

values_1 = values[diagnostic == 0, :, :]
values_2 = values[diagnostic == 1, :, :]

n_properties = values.shape[1]
n_levels = values.shape[2]

if DRAW:
    go.plot_properties(values_1[:, 0, :], values_2[:, 0, :], 
                       "Mean vertex degree", save_figures, "Mean degree")
    go.plot_properties(values_1[:, 1, :], values_2[:, 1, :], 
                       "Max vertex degree", save_figures, "Max degree")
    go.plot_properties(values_1[:, 2, :], values_2[:, 2, :], 
                       "Incidence matrix rank", save_figures, "Rank")
    go.plot_properties(values_1[:, 3, :], values_2[:, 3, :], 
                       "Number of connected components", save_figures, "B0")
    go.plot_properties(values_1[:, 4, :], values_2[:, 4, :], 
                       "Number of 1-dimensional holes", save_figures, "B1")
    go.plot_properties(values_1[:, 5, :], values_2[:, 5, :], 
                       "Logarithm of the largest elementary divisor", save_figures, "Log-BED")
    go.plot_properties(values_1[:, 6, :], values_2[:, 6, :], 
                       "Logarithm of the product of elementary divisors", save_figures, "Log-PED")
    go.plot_properties(values_1[:, 7, :], values_2[:, 7, :],
                       "Number of nontrivial elementary divisors", save_figures, "n-EDs")

if COMP:
    # exclude the properties with zero variance: log-BED, log-PED, n-EDs level 0
    values = go.flatten_data(values)
    # now values is a 2D array with shape (n_subjects, n_properties * n_levels)
    # keep all except the properties with zero variance: 5*4, 6*4, 7*4
    to_keep = [i for i in range(values.shape[1]) if i not in [5*4, 6*4, 7*4]]
    values = values[:, to_keep]
    
    values_1 = values[diagnostic == 0, :]
    values_2 = values[diagnostic == 1, :]
    
    # --------------------------------------------
    print('MANOVA TESTS ')
    print(30 * "-", " DIAG ", 30 * "-")
    maov = go.manova_test(values_1, values_2)
    print(maov)
    
    print(30 * "-", " AGE ", 30 * "-")
    v_1, v_2 = go.rearrange_groups(values, binary_age)
    print(go.manova_test(v_1, v_2))
    
    print(30 * "-", " SEX ", 30 * "-")
    v_1, v_2 = go.rearrange_groups(values, binary_sex)
    print(go.manova_test(v_1, v_2))

    # --------------------------------------------
    print('')
    print(30 * "-", " PERMUTATION TEST ", 30 * "-")
    print(f"Number of permutations: {n_permutations}")
    t0 = time.time()
    perm_o, perm_p = go.permutation_distance_test(values_1, values_2, 
                                                n_permutations=n_permutations)
    t_d = time.time() - t0
    print("Test results:")
    print("\tObserved distance:", perm_o)
    print("\tp-value:", perm_p)

    v_1, v_2 = go.rearrange_groups(values, binary_age)

    perm_o_age, perm_p_age = go.permutation_distance_test(v_1, v_2,
                                                        n_permutations=n_permutations)
    print("Age results:")
    print("\tObserved distance:", perm_o_age)
    print("\tp-value:", perm_p_age)

    v_1, v_2 = go.rearrange_groups(values, binary_sex)
    perm_o_sex, perm_p_sex = go.permutation_distance_test(v_1, v_2,
                                                            n_permutations=n_permutations)
    print("Gender results:")
    print("\tObserved distance:", perm_o_sex)
    print("\tp-value:", perm_p_sex)

    # --------------------------------------------
    print('')
    print(30 * "-", " OLS ", 30 * "-")
    ols = go.run_ols_model(og_values, age_list, sex_list, diagnostic)
    counter = {'age': 0, 'sex': 0, 'diagnostic': 0, 'total': 0}
    for key, value in ols.items():
        counter['total'] += 1
        #print(f"{key}: ")
        for k, v in value.items():
            if k == 'age' or k == 'sex' or k == 'diagnostic':
                if v < 0.05:
                    #print(f"  {k}: {v:.4f}")
                    counter[k] += 1
                    
    print(f"Total predictors exclude variance near 0 (actual total is { n_properties * n_levels}).")
    print("OLS significant (p<0.05) predictors by type:")
    for key, value in counter.items():
        print(f"  {key}: {value} ({value/counter['total']:.2%})")
