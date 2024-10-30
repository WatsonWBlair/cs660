import numpy as np
from sklearn.decomposition import PCA
from helpers.Timers.timer import run_timer
from typing import List

# from helpers.Eigne_Calculator.[filename] import [eig_func] as eigen_manual
from helpers.Principal_Component_Calculator.pca import pca as pca_manual
# from helpers.Singular_Value_Calculator.[filename] import [svd_func] as svd_manual 

def eigen_manual(data):
    # Placeholder for the actual manual eigen decomposition logic
    return np.array([]), np.array([])

def svd_manual(data):
    # Placeholder for the actual manual SVD logic
    return np.array([]), np.array([]), np.array([])

class MatrixProcessor:
    def __init__(self, labels: List[str], n_components: int = 2):
        self.labels = labels
        self.n_components = n_components
        self.execution_times = {
            'manual_eig': [], 'sklearn_eig': [],
            'manual_pca': [], 'sklearn_pca': [],
            'manual_svd': [], 'sklearn_svd': []
        }
        self.results_summary = {key: [] for key in self.execution_times.keys()}

    def process_matrix(self, i: int, data: np.ndarray):
        label = self.labels[i]
        
        # Eigen Decomposition
        results, exec_time = run_timer(eigen_manual, data)
        self.execution_times['manual_eig'].append(exec_time)
        self.results_summary['manual_eig'].append(results)

        results, exec_time = run_timer(np.linalg.eig, data)
        self.execution_times['sklearn_eig'].append(exec_time)
        self.results_summary['sklearn_eig'].append(results)
        
        # PCA Processing
        results, exec_time = run_timer(pca_manual, self.n_components, data)
        self.execution_times['manual_pca'].append(exec_time)
        self.results_summary['manual_pca'].append(results)

        sklearn_pca = PCA(n_components=self.n_components)
        results, exec_time = run_timer(sklearn_pca.fit_transform, data)
        self.execution_times['sklearn_pca'].append(exec_time)
        self.results_summary['sklearn_pca'].append(results)

        # Singular Value Decomposition (SVD)
        results, exec_time = run_timer(svd_manual, data)
        self.execution_times['manual_svd'].append(exec_time)
        self.results_summary['manual_svd'].append(results)

        results, exec_time = run_timer(np.linalg.svd, data)
        self.execution_times['sklearn_svd'].append(exec_time)
        self.results_summary['sklearn_svd'].append(results)
