import numpy as np
from sklearn.decomposition import PCA
from helpers.Timers.timer import run_timer

# from helpers.Eigne_Calculator.[filename] import [eig_func] as eigen_manual
from helpers.Principal_Component_Calculator.pca import pca as pca_manual
# from helpers.Singular_Value_Calculator.[filename] import [svd_func] as svd_manual 

def eigen_manual(data):
    # Placeholder for the actual manual eigen decomposition logic
    return np.array([]), np.array([])

def svd_manual(data):
    # Placeholder for the actual manual SVD logic
    return np.array([]), np.array([]), np.array([])

def process_matrix(i, data, labels, execution_times, results_summary):
    print(f"\nRunning computations on {labels[i]} matrix")
    
    # Manual Eigen
    results, exec_time = run_timer(eigen_manual, data)
    execution_times['manual_eig'].append(exec_time)
    results_summary['manual_eig'].append(results)

    # NumPy Eigen
    results, exec_time = run_timer(np.linalg.eig, data)
    execution_times['sklearn_eig'].append(exec_time)
    results_summary['sklearn_eig'].append(results)

    # Manual PCA
    results, exec_time = run_timer(pca_manual, 2, data)
    execution_times['manual_pca'].append(exec_time)
    results_summary['manual_pca'].append(results)

    # NumPy PCA
    sklearn_pca = PCA(n_components=2)
    results, exec_time = run_timer(sklearn_pca.fit_transform, data)
    execution_times['sklearn_pca'].append(exec_time)
    results_summary['sklearn_pca'].append(results)

    # Manual SVD
    results, exec_time = run_timer(svd_manual, data)
    execution_times['manual_svd'].append(exec_time)
    results_summary['manual_svd'].append(results)

    # Numy SVD
    results, exec_time = run_timer(np.linalg.svd, data)
    execution_times['sklearn_svd'].append(exec_time)
    results_summary['sklearn_svd'].append(results)