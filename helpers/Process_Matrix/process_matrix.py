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