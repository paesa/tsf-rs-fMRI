# tsf-rs-fMRI

This repository provides the code used in the master's thesis *Topological Scale Framework Applied to Resting-State fMRI Networks in Inhalant Substance Abuse*. It implements a novel analysis pipeline for subject-specific brain connectivity networks using algebraic diffusion on hypergraphs.

## 🔍 Overview

The main goal is to capture structural degradation in functional connectivity patterns of adolescent subjects with Inhaled Substance Abuse Disorder (ISAD), using a scale-space model built from sparse incidence matrices.

The method operates by:

* Constructing hypergraphs from correlation matrices.
* Building a multiscale topological model (\$s^2\$-model).
* Extracting homological and algebraic invariants at each scale.
* Performing statistical comparison across diagnostic, age, and sex groups.

## 📁 Repository Structure

```
tsf-rs-fMRI/
├── compute_all.py            # Script to compute topological features for all subjects
├── compare_all.py            # Statistical tests and plots across groups
├── graph_operations.py       # Core functions for loading, building, and analyzing hypergraphs
├── matrices/                 # Folder for incidence matrices (one per subject)
├── results/                  # Folder to store computed descriptor results
├── imgs/                     # Plots generated from comparisons
```

## 🛆 Requirements

* Python 3.8+
* NumPy
* pandas
* matplotlib
* statsmodels
* scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 📊 Running the Analysis

1. **Prepare the data:**

   This repository **does not** include the fMRI correlation matrices due to licensing. To obtain them:

   * Download from [Mijangos et al. (2024)](https://figshare.com/articles/dataset/Resting_state_correlation_matrices/21941681).
   * You can use functions from `graph_operations.py` to construct the incidence matrices.

2. **Compute topological descriptors:**

```bash
python compute_all.py
```

3. **Compare groups and generate plots:**

```bash
python compare_all.py
```

This will:

* Save descriptors to `results/`
* Generate comparative plots in `imgs/`
* Run MANOVA, permutation, and OLS statistical tests

## 📊 Output

* `results/`: Numerical descriptors (e.g., Betti numbers, degrees, matrix ranks)
* `imgs/`: Visualization of descriptor distributions across groups and scales

## 📬 Contact

For questions, issues, or collaboration ideas, feel free to contact:
📧 [carlospaesa@gmail.com](mailto:carlospaesa@gmail.com)

