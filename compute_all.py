import numpy as np
import graph_operations as go
from joblib import Parallel, delayed
np.set_printoptions(linewidth=100)

# This script computes topological properties for a set of incidence matrices,
# which are stored in a folder. The results are saved in a results folder.

tol = 0.0031
correlation = 0.3
last_layer = 6

folder = "matrices/"
group1 = ['01_g4s01.txt', '02_g4s05.txt', '03_g4s11.txt', '04_g5s01.txt', '05_g5s02.txt', '06_g5s06.txt',
         '07_g5s08.txt', '08_g5s10.txt', '09_g5s12.txt', '10_g5s13.txt', '11_g6s01.txt', '12_g6s04.txt',
         '13_g6s06.txt', '14_g6s08.txt', '15_g6s09.txt', '16_g6s13.txt', '17_g6s14.txt', '18_g7c01.txt',
         '19_g7c04.txt']
group2 = ['20_g7s01.txt', '21_g7s02.txt', '22_g7s03.txt', '23_g7s05.txt', '24_g7s06.txt', '25_g7s08.txt',
         '26_g7s09.txt', '27_g7s10.txt', '28_g7s11.txt', '29_g7s12.txt', '30_g7s13.txt', '31_g7s14.txt',
         '32_g7s16.txt', '33_g7s17.txt', '34_g7s18.txt', '35_g7s19.txt', '36_g7s22.txt', '37_g7s23.txt', 
         '38_g7s24.txt', '39_g7s25.txt', '40_g7s26.txt', '41_g7s28.txt', '42_g7s29.txt', '43_g7s30.txt']
group = group1 + group2

def process_file(filename):
    print("Loading file ", filename)
    m = go.load_incidence_matrix(folder + filename, tol=tol,
                                 minimum_correlation=correlation)
    ending = filename.replace('.txt', '')
    result_folder = "results/" 
    df = go.obtain_topological_properties(m, label="TP_" + ending, 
            folder=result_folder, last_layer=6, verbose=False)
    return df

# Parallel processing of files in the group
results = Parallel(n_jobs=-1)(delayed(process_file)(filename) for filename in group)
print("All files processed.")
