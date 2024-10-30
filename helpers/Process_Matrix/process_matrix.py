import numpy as np
from sklearn.decomposition import PCA
from helpers.Timers.timer import run_timer
from typing import List
import logging
from concurrent.futures import ThreadPoolExecutor

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
        
        computations = {
            'manual_eig': (run_timer, eigen_manual, data),
            'sklearn_eig': (run_timer, np.linalg.eig, data),
            'manual_pca': (run_timer, pca_manual, self.n_components, data),
            'sklearn_pca': (run_timer, self._sklearn_pca, data),
            'manual_svd': (run_timer, svd_manual, data),
            'sklearn_svd': (run_timer, np.linalg.svd, data),
        }

        for key, (timer_func, func, *args) in computations.items():
            try:
                results, exec_time = timer_func(func, *args)
                self.execution_times[key].append(exec_time)
                self.results_summary[key].append(results)
            except Exception as e:
                logging.error(f"Error in {key} for {label} matrix: {e}")

    def _sklearn_pca(self, data: np.ndarray) -> np.ndarray:
        sklearn_pca = PCA(n_components=self.n_components)
        return sklearn_pca.fit_transform(data)

    def execute_all(self, data_list: List[np.ndarray]):
        with ThreadPoolExecutor() as executor:
            for i, data in enumerate(data_list):
                executor.submit(self.process_matrix, i, data)