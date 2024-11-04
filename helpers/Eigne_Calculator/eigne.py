
import numpy as np

"""eigne function calculates the eigenvalues and eigenvectors for a given covariance matrix"""
def eigne(covMatrix):

    result = recursiveDeterminate(covMatrix)
    
    return result

""" manual_eigen function authored by Syed Abdul Mubashir """
def manual_eigen(A):
    # Check if the matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    n = A.shape[0]  # size of the matrix
    
    # Step 1: Calculate the characteristic polynomial coefficients
    coeffs = np.poly(A)  # Get coefficients of the characteristic polynomial
    eigenvalues = np.roots(coeffs)  # Eigenvalues are the roots of the characteristic polynomial
    
    # Step 2: Eigenvector computation for each eigenvalue
    eigenvectors = []  # Initialize an empty list to store eigenvectors
    for lambda_val in eigenvalues:
        I = np.eye(n)  # Create an identity matrix of size n
        M = A - lambda_val * I  # Form matrix (A - lambda*I)
        
        # Step 3: Use Gaussian elimination to solve (A - lambda*I)x = 0
        for i in range(n):
            # Pivot: Swap rows if the diagonal element is zero
            if M[i][i] == 0:
                for j in range(i + 1, n):
                    if M[j][i] != 0:
                        M[[i, j]] = M[[j, i]]  # Swap rows
                        break
            
            # Normalize the row (make the leading entry 1) if possible
            M[i] = M[i] / M[i][i] if M[i][i] != 0 else M[i]
            
            # Eliminate entries below the current row to make it an upper triangular matrix
            for j in range(i + 1, n):
                M[j] = M[j] - M[j][i] * M[i]
        
        # TODO: Update 
        # Use the last row as an approximation for the eigenvector
        eigenvector = M[:, -1] if n > 1 else np.array([1])  # Handle single-row case
        eigenvectors.append(eigenvector)  # Add the computed eigenvector to the list

    # Return real values if possible for eigenvalues and eigenvectors
    return np.real_if_close(eigenvalues), [np.real_if_close(vec) for vec in eigenvectors]

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
