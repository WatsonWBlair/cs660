# CS660 - Project 2

## Overview

This repository contains the codebase for **Project 2** in **CS660: Mathematical Foundations of Analytics (CRN# 71425), Fall 2024** at Pace University. The project involves implementing Python scripts to compute Eigenvalues, Eigenvectors, Principal Components, and Singular Values of a matrix, and comparing the execution time of our code with Python’s built-in libraries.

## Course Information
- Course: CS660/71425 Mathematical Foundations of Analytics
- Instructor: Prof. Tassos Sarbanes
- Group-1: Will Torres, Mike Griffin, Watson Blair, Syed Abdul Mubashir
- Semester: Fall 2024
- Project #: 2
- Due Date: 04-Nov-2024

## Project Requirements
1. **Eigenvalues and Eigenvectors Computation:**  
   - Compute the eigenvalues and corresponding eigenvectors of a matrix.
   - Compare the runtime with Python’s built-in function `numpy.linalg.eig`.
   - [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)

2. **Principal Components Computation:**  
   - Calculate the principal components of a matrix.
   - Compare runtime with `sklearn.decomposition.PCA`.
   - [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

3. **Singular Values Computation:**  
   - Perform Singular Value Decomposition (SVD) to find the singular values of a matrix.
   - Compare runtime with `numpy.linalg.svd`.
   - [Documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)

Each task includes creating independent computation modules and analyzing runtime performance.

## Approach
The core of our implementation consists of the following helper functions, designed to handle computations and execution timing:
- **`pca`**: Computes the principal components of a given matrix.
- **`eigen`**: Calculates eigenvalues and eigenvectors.
- **`svd`**: Performs Singular Value Decomposition.
- **`run_timer`**: Measures and records the execution time for each function.
- **`process_matrix`**: Organizes data preparation and initiates computations on matrices of various sizes.

These functions are combined within `presentation.ipynb` to execute each calculation and gather runtime data for comparison.

### Efficiency Analysis
To analyze computation efficiency, we test each function with randomly generated matrices of different sizes:

```python
# Define sample sizes for analysis
matrix_sizes = [10, 25, 50, 75, 100, 250, 400]
data_list = [np.random.rand(n, n) for n in matrix_sizes]
labels = [f'{n}x{n}' for n in matrix_sizes]

# Initialize BenchmarkUtility to iterate through matrices and benchmark
processor = BenchmarkUtility(labels=labels)
processor.batch_benchmark(data_list)
```

This setup allows us to evaluate how each implementation scales with matrix size and to benchmark our code against standard libraries.

## How to Run
1. Open `presentation.ipynb` in Jupyter Notebook.
2. Run the notebook to execute computations, collect runtime data, and view analysis results.
