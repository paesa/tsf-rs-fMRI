import os
import networkx as nx # for graph operations
import numpy as np  # for matrix operations
import matplotlib.pyplot as plt # for plotting
from sympy import Matrix, ZZ, GF # for smith normal form
from sympy.matrices.normalforms import smith_normal_form
from scipy.sparse import lil_matrix
import pandas as pd # dataframes
from scipy.spatial.distance import euclidean
from statsmodels.multivariate.manova import MANOVA
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------- GRAPHS ------------------------
def load_graph(filename, n_edges=1):
    """
    Load a graph from a file in the format:
    (vertex1, value1, vertex2, value2, ..., vertexN, valueN)
    where vertex1 is the first vertex and value2 is the value of the edge
    between vertex1 and vertex2, and so on.
    The function returns a NetworkX graph object.
    filename: path to the file containing the graph
    n_edges: number of edges to consider for each vertex, sorted by value.
    """
    edges = []
    with open(filename) as f:
        lines=f.readlines()
        first_line = lines[0].split(' ')
        n_vertices = len(first_line)
        lines = lines[1:]
        for line in lines:
            line = line.split(' ')
            first_element = line[0][1:-1]
            values = [(float(line[i]), i) for i in range(1, len(line))]
            # sort by the first element of the tuple
            values.sort(key=lambda x: -x[0])
            # get the n_edges first elements
            values = values[:n_edges]
            for value, second_element in values:
                edges.append((int(first_element), second_element))
    
    graph = nx.Graph()
    graph.add_nodes_from(range(1, n_vertices+1))
    graph.add_edges_from(edges) 
    return graph

def draw_graph(G, label=""):
    """
    Draw a graph using NetworkX and Matplotlib.
    G: NetworkX graph object
    label: optional label for saving the figure
    """
    pos = nx.spring_layout(G, k=0.25, iterations=50, seed=7)  # positions for all nodes - seed for reproducibility
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="black")
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color="gray")
    
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    if label:
        plt.savefig(f"imgs/graph_{label}.png")
    plt.show()

# ----------------------- INCIDENCE -----------------------
def load_incidence_matrix(filename, n_edges=1):
    """
    Load an incidence matrix from a file in the format:
    (vertex1, value1, vertex2, value2, ..., vertexN, valueN)
    
    where vertex1 is the first vertex and value2 is the value of the edge
    between vertex1 and vertex2, and so on.
    The function returns a sparse matrix in CSR format.
    filename: path to the file containing the incidence matrix
    n_edges: number of edges to consider for each vertex, sorted by value.
    """
    border_operator = []
    n_vertices = 0
    # for each line, we will add the border vertices by number
    with open(filename) as f:
        lines=f.readlines()
        first_line = lines[0].split(' ')
        n_vertices = len(first_line)
        lines = lines[1:]
        for line in lines:
            line = line.split(' ')
            first_element = int(line[0][1:-1])
            values = [(float(line[i]), i) for i in range(1, len(line))]
            # sort by the first element of the tuple
            values.sort(key=lambda x: -x[0])
            # get the n_edges first elements
            values = values[:n_edges]
            border = [first_element-1]
            for value, second_element in values:
                border.append(second_element-1)
            # sort the border
            border.sort()
            border_operator.append(border)
        
    # remove duplicate borders
    border_operator = np.unique(border_operator, axis=0)
    # create the incidence matrix
    n_edges = len(border_operator)
    incidence_matrix = lil_matrix((n_vertices, n_edges), dtype=int)
    print("Incidence matrix shape:", incidence_matrix.shape)
    for j in range(n_edges):
        print(j, "---", border_operator[j])
        for i in border_operator[j]:
            incidence_matrix[i, j] = 1
    incidence_matrix = incidence_matrix.tocsr()
    return incidence_matrix

def load_incidence_matrix_tol(filename, tol=0.01):
    """
    Load an incidence matrix from a file in the format:
    (vertex1, value1, vertex2, value2, ..., vertexN, valueN)
    
    where vertex1 is the first vertex and value2 is the value of the edge
    between vertex1 and vertex2, and so on.
    The function returns a sparse matrix in CSR format.
    filename: path to the file containing the incidence matrix
    tol: tolerance to consider adding more vertices to the border
         of the considered edge.
    """
    border_operator = []
    n_vertices = 0
    # for each line, we will add the border vertices by number
    with open(filename) as f:
        lines=f.readlines()
        first_line = lines[0].split(' ')
        n_vertices = len(first_line)
        lines = lines[1:]
        for line in lines:
            line = line.split(' ')
            first_element = int(line[0][1:-1])
            values = [(float(line[i]), i) for i in range(1, len(line))]
            # sort by the first element of the tuple
            values.sort(key=lambda x: -x[0])
            border = [first_element-1]
            first_value = values[0][0]
            for value, second_element in values:
                if abs(value - first_value) < tol:
                    border.append(second_element-1)
                else:
                    break
            # sort the border
            border.sort()
            border_operator.append(border)
        
    # remove duplicate borders
    # border_operator = np.unique(border_operator, axis=0)
    # unique does not work for different lengths, so we need to do it manually
    # group by length
    border_operator = sorted(border_operator, key=lambda x: len(x))
    b_o_grouped = []
    for i in range(len(border_operator)):
        if i == 0:
            b_o_grouped.append([border_operator[i]])
        else:
            if len(border_operator[i]) == len(border_operator[i-1]):
                b_o_grouped[-1].append(border_operator[i])
            else:
                b_o_grouped.append([border_operator[i]])
    # remove duplicates in each group
    for i in range(len(b_o_grouped)):
        #print("Group length:", len(b_o_grouped[i]))
        #print("Group:", b_o_grouped[i])
        b_o_grouped[i] = np.unique(b_o_grouped[i], axis=0)
        print("Group unique length:", len(b_o_grouped[i]))
    # append the unique groups
    border_operator = []
    for i in range(len(b_o_grouped)):
        border_operator.extend(b_o_grouped[i])
    # create the incidence matrix
    n_edges = len(border_operator)
    incidence_matrix = lil_matrix((n_vertices, n_edges), dtype=int)
    for j in range(n_edges):
        #print(j, "---", border_operator[j])
        for i in border_operator[j]:
            incidence_matrix[i, j] = 1
    incidence_matrix = incidence_matrix.tocsr()
    return incidence_matrix

# -------------------------- MAIN ------------------------
def obtain_topological_properties(matrix, label="", folder = "results", verbose=True):
    """
    Obtain topological properties of a given incidence matrix and
    its scale space model.
    matrix: incidence matrix in CSR format
    label: label for the output file
    folder: folder to save the output file
    verbose: whether to print the results to the console
    """
    B = matrix.tocsr()
    second_B = B.T @ B
    second_B.data = second_B.data % 2
    
    third_B = B @ second_B
    third_B.data = third_B.data % 2
    
    fourth_B = B.T @ third_B
    fourth_B.data = fourth_B.data % 2
    
    fifth_B = B @ fourth_B
    fifth_B.data = fifth_B.data % 2
    
    sixth_B = B.T @ fifth_B
    sixth_B.data = sixth_B.data % 2
    
    matrices_to_study = [B, second_B, third_B, fourth_B, fifth_B, sixth_B]
    
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(f"{folder}/{label}.txt", "w+")
    dicts = []
    
    for i, matrix in enumerate(matrices_to_study):
        mean_v_degree = np.mean(matrix.sum(axis=1))
        max_v_degree = np.max(matrix.sum(axis=1))
        if verbose:
            print(f"Matrix {i}: {matrix.shape}")
            print(f"\tMean v deg.: {mean_v_degree}")
            print(f"\tMax v deg.: {max_v_degree}")
        
        D = smith_normal_form(Matrix(matrix.todense()),
                              domain=ZZ)
        rank = D.rank()
        if rank > 0:
            elem_div = np.abs(D.diagonal()[:rank])
        else:
            elem_div = np.array([])
        elem_div = elem_div[elem_div > 1]
        
        
        betti_0 = matrix.shape[0] - rank
        betti_1 = matrix.shape[1] - rank
        
        if verbose:
            print(f"\tRank of matrix {i}: {rank}")
            print(f"\tElementary divisors: {elem_div}")
            print(f"\tBetti numbers: {betti_0}, {betti_1}")
            print("------------------------------------")
        
        dict = {"Mean deg.": mean_v_degree, "Max deg.": max_v_degree, "Rank": rank,
                "Elementary divisors": elem_div, "B0": betti_0, "B1": betti_1}
        dicts.append(dict)
        
    
    df = pd.DataFrame(columns=["Mean deg.", "Max deg.", "Rank", "Elementary divisors", "B0", "B1"],
                        data=dicts)
    f.write(df.to_csv())
    f.close()
    
    return df

# ------------------------ USEFUL ------------------------
def flatten_data(values):
    """
    Flatten a 3D array of shape (n1, d1, d2) to a 2D array of shape (n1, d1 * d2).
    values: 3D numpy array
    Returns: 2D numpy array
    """
    n1, d1, d2 = values.shape
    values = values.reshape((n1, d1 * d2))
    return values

# --------------------- GROUP TESTING ---------------------
def load_group_results(group, folder):
    """
    Load results from a group of files and return the values as a 3D numpy array.
    group: list of filenames in the group
    folder: folder where the files are located
    Returns: 3D numpy array of shape (n_files, n_properties, n_layers)
    """
    columns = ['Mean deg.', 'Max deg.', 'Rank', 'B0', 'B1', 
               'log-BED', 'log-PED', 'n-EDs']

    # read the first file to get the real number of layers
    f_ending = group[0].replace('.txt', '')
    file = folder + "TP_" + f_ending + ".txt"
    df = pd.read_csv(file, sep=",")
    df.drop(df.columns[0], axis=1)  # drop the first column
    layers = len(df)
    
    values = np.zeros((len(group), len(columns), layers))
    for i, filename in enumerate(group):
        print("Loading results ", filename)
        ending = filename.replace('.txt', '')
        file = folder + "TP_" + ending + ".txt"
        # load as dataframe
        df = pd.read_csv(file, sep=",")
        #drop the first column
        df = df.drop(df.columns[0], axis=1)
        #print(df)
        
        # get the values
        mean_deg = df['Mean deg.'].values
        max_deg = df['Max deg.'].values
        rank = df['Rank'].values
        b0 = df['B0'].values
        b1 = df['B1'].values
        bed = df['Elementary divisors'].values
        true_bed = [b.strip('[]').split(' ') for b in bed]
        true_bed = [[int(x) for x in b if x.isdigit()] for b in true_bed]
        bed = []
        ped = []
        ned = []
        for b in true_bed:
            if len(b) > 0:
                ped.append(abs(np.prod(b)))
                bed.append(np.max(b))
                ned.append(len(b))
            else:
                bed.append(1)
                ped.append(1)
                ned.append(0)
        # store the values
        values[i, 0, :] = mean_deg
        values[i, 1, :] = max_deg
        values[i, 2, :] = rank
        values[i, 3, :] = b0
        values[i, 4, :] = b1
        values[i, 5, :] = np.log2(bed)  # log-BED, adding 1 to avoid log(0)
        values[i, 6, :] = np.log2(ped)
        values[i, 7, :] = ned
    return values

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_properties(p_g1, p_g2, name, save=False, save_filename=None):
    """
    Plot all properties of two groups of graphs on each scale space level.
    p_g1: properties of group 1 (numpy array)
    p_g2: properties of group 2 (numpy array)
    name: y-axis label
    save: whether to save the plots as an image
    """
    # Color variables for easy customization
    point_color_g1 = 'blue'
    point_color_g2 = 'red'
    box_color_g1 = 'lightskyblue'
    box_color_g2 = 'orange'

    l_1 = len(p_g1)
    l_2 = len(p_g2)
    n_subplots = p_g1.shape[1]

    plt.figure(figsize=(10, 5))
    plt.suptitle(name, fontsize=16)

    for i in range(n_subplots):
        m_1 = p_g1.mean(axis=0)[i]
        m_2 = p_g2.mean(axis=0)[i]
        s_1 = p_g1.std(axis=0)[i]
        s_2 = p_g2.std(axis=0)[i]

        ax = plt.subplot(2, int(np.round(n_subplots / 2 + 0.4)), i + 1)

        # Draw boxes first (Group 1 and Group 2)
        ax.plot([1, l_1], [m_1, m_1], color=box_color_g1, linewidth=1)
        ax.plot([1, l_1], [m_1 - s_1, m_1 - s_1], linestyle='--', color=box_color_g1, linewidth=1)
        ax.plot([1, l_1], [m_1 + s_1, m_1 + s_1], linestyle='--', color=box_color_g1, linewidth=1)
        ax.fill_between([1, l_1], m_1 - s_1, m_1 + s_1, color=box_color_g1, alpha=0.15)

        ax.plot([l_1 + 1, l_1 + l_2], [m_2, m_2], color=box_color_g2, linewidth=1)
        ax.plot([l_1 + 1, l_1 + l_2], [m_2 - s_2, m_2 - s_2], linestyle='--', color=box_color_g2, linewidth=1)
        ax.plot([l_1 + 1, l_1 + l_2], [m_2 + s_2, m_2 + s_2], linestyle='--', color=box_color_g2, linewidth=1)
        ax.fill_between([l_1 + 1, l_1 + l_2], m_2 - s_2, m_2 + s_2, color=box_color_g2, alpha=0.15)

        # Draw points on top
        ax.plot(np.linspace(1, l_1, l_1), p_g1[:, i], '.', color=point_color_g1, label='Control', markersize=4)
        ax.plot(np.linspace(l_1 + 1, l_1 + l_2, l_2), p_g2[:, i], '.', color=point_color_g2, label='ISAD', markersize=4)

        ax.set_title("Level " + str(i + 1))

        # Custom x-axis ticks
        ax.set_xticks([l_1 / 2, l_1 + l_2 / 2])
        ax.set_xticklabels(["Controls", "ISAD"])

    plt.tight_layout()
    if save:
        if save_filename:
            name = save_filename
        plt.savefig(f"imgs/{name}.png", dpi=600)
    else:
        plt.show()


def violin_plot_properties(p_g1, p_g2, name, save=False):
    """
    Create a violin plot for the properties of two groups of graphs.
    p_g1: properties of group 1 (numpy array)
    p_g2: properties of group 2 (numpy array)
    name: name of the plot
    save: whether to save the plots as an image
    """
    import seaborn as sns
    
    l_1 = len(p_g1)
    l_2 = len(p_g2)
    
    plt.figure(figsize=(10, 5))
    plt.suptitle(name, fontsize=16)
    
    for i in range(p_g1.shape[1]):
        plt.subplot(2, int(np.round(p_g1.shape[1] / 2 + 0.4)), i + 1)
        sns.violinplot(data=[p_g1[:, i], p_g2[:, i]], palette=["blue", "red"])
        plt.title("Level " + str(i+1))
        plt.xticks([0, 1], ['Control', 'ISAD'])
    
    plt.tight_layout()
    if save:
        plt.savefig(f"imgs/{name}.png")
    else:
        plt.show()

def rearrange_groups(values, y):
    """
    Rearrange the values according to the binary labels y.
    values: 2D numpy array of shape (n_samples, n_features) 
    y: 1D numpy array of binary labels (0 or 1)
    Returns: two numpy arrays with values for each group (0 and 1)
    """
    y = np.array(y)
    v_1 = values[y == 0]
    v_2 = values[y == 1]
    return v_1, v_2

def manova_test(values_1, values_2):
    """Perform a MANOVA test on two sets of values, excluding low-variance features
    and applying PCA for dimensionality reduction if necessary.
    values_1: 2D or 3D numpy array for group 1
    values_2: 2D or 3D numpy array for group 2
    Returns: MANOVA test results
    """
    X1_flat = values_1
    X2_flat = values_2
    if values_1.ndim == 3:
        X1_flat = flatten_data(values_1)
    if values_2.ndim == 3:
        X2_flat = flatten_data(values_2)
    X = np.vstack((X1_flat, X2_flat))
    y = np.array([0] * len(X1_flat) + [1] * len(X2_flat))

    # Remove constant/near-constant features
    sel = VarianceThreshold(threshold=1e-5)
    X_sel = sel.fit_transform(X)
    
    # PCA to reduce dimensionality
    # since we have many features and few samples
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(15, X_sel.shape[1]))
    X_sel = pca.fit_transform(X_sel)

    df = pd.DataFrame(X_sel, columns=[f'v{i}' for i in range(X_sel.shape[1])])
    df['group'] = y
    formula = ' + '.join(df.columns[:-1]) + ' ~ group'
    maov = MANOVA.from_formula(formula, data=df)
    return maov.mv_test()

def permutation_distance_test(values_1, values_2, n_permutations=1000):
    """
    Perform a permutation test to compare the means of two groups based on the Euclidean distance
    between their means.
    values_1: 2D or 3D numpy array for group 1
    values_2: 2D or 3D numpy array for group 2
    n_permutations: number of permutations to perform
    Returns: observed distance and p-value
    """
    X1_flat = values_1
    X2_flat = values_2
    if values_1.ndim == 3:
        X1_flat = flatten_data(values_1)
    if values_2.ndim == 3:
        X2_flat = flatten_data(values_2)
    X = np.vstack((X1_flat, X2_flat))
    y = np.array([0] * len(X1_flat) + [1] * len(X2_flat))

    g1 = X[y == 0]
    g2 = X[y == 1]
    observed = euclidean(g1.mean(axis=0), g2.mean(axis=0))

    count = 1
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y) #np.random.randint(0, 2, size=len(y))
        g1_perm = X[y_perm == 0]
        g2_perm = X[y_perm == 1]
        dist = euclidean(g1_perm.mean(axis=0), g2_perm.mean(axis=0))
        if dist >= observed:
            count += 1

    p_value = count / n_permutations
    return observed, p_value

def run_ols_model(Y, age, sex, diagnostic):
    """
    Run an OLS regression for each feature in Y against the predictors:
    Y: 2D or 3D numpy array of shape (n_samples, n_features)
    age: 1D numpy array of ages
    sex: 1D numpy binary array indicating sex (0 or 1)
    diagnostic: 1D numpy binary array indicating diagnostic status
    Returns: dictionary for each feature with p-values of the predictors
    """
    results = {}
    if Y.ndim == 3:
        Y = flatten_data(Y)
    sel = VarianceThreshold(threshold=1e-5)
    #Y = sel.fit_transform(Y)
    df = pd.DataFrame(Y, columns=[f'v{i}' for i in range(Y.shape[1])])
    df['age'] = age
    df['sex'] = sex
    df['diagnostic'] = diagnostic

    for col in [f'v{i}' for i in range(Y.shape[1])]:
        formula = f"{col} ~ age + sex + diagnostic"
        model = smf.ols(formula, data=df).fit()
        results[col] = model.pvalues.to_dict()

    return results