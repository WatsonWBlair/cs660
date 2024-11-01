
import numpy as np

"""eigne function calculates the eigenvalues and eigenvectors for a given covariance matrix"""
def eigne(covMatrix):

    result = recursiveDeterminate(covMatrix)
    
    return result


def recursiveDeterminate(matrix):
    size = matrix.shape[0]

    if(size == 2):
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for i in range(size):
        submatrix = np.delete(matrix, i, axis=1)[1:]
        det += ((-1) ** i) * matrix[0][i] * recursiveDeterminate(submatrix)

    return det

def identity(shape):
    result = np.zeros(shape)
    for i in range(shape[0]):
        result[i][i] = 1
    return result
