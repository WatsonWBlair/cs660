
import numpy as np
from sympy.matrices import Matrix, eye
from sympy import symbols
import pprint

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

"""
else:
    for i in range(size):
        mod_array = 
        multiplier = 1 if i%2 else -1
        determinate += multiplier*(matrix[0,i] * recursiveDeterminate(mod_array))

return determinate
"""

"""
 det = 0
    for i in range(len(matrix)):
        # Create submatrix by removing the first row and ith column
        submatrix = [row[:i] + row[i + 1:] for row in matrix[1:]]
        det += ((-1) ** i) * matrix[0][i] * determinant(submatrix)

    return det


"""


