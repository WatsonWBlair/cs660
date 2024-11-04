import numpy as np
import pandas as pd
import matplotlib as plt
import sympy as smp
from sympy.solvers import solve
from sympy import Symbol

Matrix=Matrix*Matrix.mean()/np.std(Matrix)
shape=Matrix.shape
def identity(shape):
    result = np.zeros(shape)
    for i in range(shape[0]):
        result[i][i] = 1
    return result
#Build Arrays, and Transpose of those arrays
def SVDSetupU(Matrix):
    Matrix=np.array(Matrix).copy()

    Transpose=np.transpose(Matrix)
    MultipliedU=np.dot(Matrix, Transpose)

    return MultipliedU

def SVDSetupV(Matrix):
    Matrix=np.array(Matrix).copy()

    Transpose=np.transpose(Matrix)

    MultipliedV=np.dot(Transpose, Matrix)
    return MultipliedV

#Find Eigenvalues of Matrices U and V
def EigenvaluesU(MultipliedU):
  # Step 1: Calculate the characteristic polynomial coefficients
  coeffs = np.poly(MultipliedU)  # Get coefficients of the characteristic polynomial
  eigenvaluesU = np.roots(coeffs)
  valuesU=eigenvaluesU
  return valuesU  # Eigenvalues are the roots of the characteristic polynomial
def EigenvaluesV(MultipliedV):
  # Step 1: Calculate the characteristic polynomial coefficients
  coeffs = np.poly(MultipliedV)  # Get coefficients of the characteristic polynomial
  eigenvaluesV = np.roots(coeffs)
  valuesV=eigenvaluesV
  return valuesV  # Eigenvalues are the roots of the characteristic polynomial  

#Find Eigenvectors of Matrices U and V
def EigenvectorsU(MultipliedU):
    coeffs = np.poly(MultipliedU)  # Get coefficients of the characteristic polynomial
    eigenvalues = np.roots(coeffs)  # Eigenvalues are the roots of the characteristic polynomial  
    # Step 2: Eigenvector computation for each eigenvalue
    eigenvectors = []  # Initialize an empty list to store eigenvectors
    for lambda_val in eigenvalues:
        I = identity(shape)  # Create an identity matrix of size n
        M = MultipliedU - lambda_val * I  # Form matrix (A - lambda*I)
        
        # Step 3: Use Gaussian elimination to solve (A - lambda*I)x = 0
        for i in range(shape[0]):
            # Pivot: Swap rows if the diagonal element is zero
            if M[i][i] == 0:
                for j in range(i + 1, shape[0]):
                    if M[j][i] != 0:
                        M[[i, j]] = M[[j, i]]  # Swap rows
                        break
            
            # Normalize the row (make the leading entry 1) if possible
            M[i] = M[i] / M[i][i] if M[i][i] != 0 else M[i]
            
            # Eliminate entries below the current row to make it an upper triangular matrix
            for j in range(i + 1, shape[0]):
                M[j] = M[j] - M[j][i] * M[i]
        
        # TODO: Update 
        # Use the last row as an approximation for the eigenvector
        eigenvector = M[:, -1] if shape[0] > 1 else np.array([1])  # Handle single-row case
        eigenvectors.append(eigenvector)  # Add the computed eigenvector to the list

    # Return real values if possible for eigenvalues and eigenvectors
    return [np.real_if_close(vec) for vec in eigenvectors]

def EigenvectorsV(MultipliedV):
  coeffs = np.poly(MultipliedV)  # Get coefficients of the characteristic polynomial
  eigenvalues = np.roots(coeffs)  # Eigenvalues are the roots of the characteristic polynomial  
    # Step 2: Eigenvector computation for each eigenvalue
  eigenvectors = []  # Initialize an empty list to store eigenvectors
  for lambda_val in eigenvalues:
      I = identity(shape)  # Create an identity matrix of size n
      M = MultipliedV - lambda_val * I  # Form matrix (A - lambda*I)
        
        # Step 3: Use Gaussian elimination to solve (A - lambda*I)x = 0
      for i in range(shape[0]):
            # Pivot: Swap rows if the diagonal element is zero
          if M[i][i] == 0:
              for j in range(i + 1, shape[0]):
                  if M[j][i] != 0:
                      M[[i, j]] = M[[j, i]]  # Swap rows
                      break
            
            # Normalize the row (make the leading entry 1) if possible
          M[i] = M[i] / M[i][i] if M[i][i] != 0 else M[i]
            
            # Eliminate entries below the current row to make it an upper triangular matrix
          for j in range(i + 1, shape[0]):
                M[j] = M[j] - M[j][i] * M[i]
        
        # TODO: Update 
        # Use the last row as an approximation for the eigenvector
      eigenvector = M[:, -1] if shape[0] > 1 else np.array([1])  # Handle single-row case
      eigenvectors.append(eigenvector)  # Add the computed eigenvector to the list

    # Return real values if possible for eigenvalues and eigenvectors
  return [np.real_if_close(vec) for vec in eigenvectors]

#Compute Singular Values, the square roots of the eigenvalues of Matrix U.
def SingularValues(values_U):
    Sing_Values=np.sqrt(values_U)
    Sing_Values=np.sort(Sing_Values)[::-1]
    return Sing_Values

#create a Singular Value Array, which fills the diagonal of a 0 matrix of the same dimensions
# as our original matrix with the singular values we calculated.
def CreateSVArray(Sing_Values, Matrix):
    rows, columns=Matrix.shape
    length=min(rows,columns)
    def pad(Sing_Values):
      padded_zeros=Sing_Values
      for i in range(length-len(padded_zeros)):
          padded_zeros=np.append(padded_zeros,0)
      return padded_zeros
    SV_Array=pad(Sing_Values)
    return SV_Array


# #Build matrices U and V, which consist of the Eigenvectors of U and V
def MatrixU(vectors_U):
    matrix_U=np.array(vectors_U)
    return matrix_U


def MatrixV(vectors_V):
    matrix_V=np.array(vectors_V)
    return matrix_V

#Perform Singular Value Decomposition, the dot product of Matrices U and A, and the Singular Values Array
def SVDBuild(matrix_U,SV_Array,matrix_V):
    SVD=matrix_U@(SV_Array@matrix_V.T)
    return SVD

#Total Helper Function
def SVD(Matrix):
    MultipliedU=SVDSetupU(Matrix)
    MultipliedV=SVDSetupV(Matrix)
    values_U=EigenvaluesU(MultipliedU)
    values_V=EigenvaluesV(MultipliedV)
    vectors_U=EigenvectorsU(MultipliedU)
    vectors_V=EigenvectorsV(MultipliedV)
    Sing_Values=SingularValues(values_V)
    SV_Array=CreateSVArray(Sing_Values, Matrix)
    matrix_U=MatrixU(vectors_U)
    matrix_V=MatrixV(vectors_V)
    SVD=SVDBuild(matrix_U,SV_Array,matrix_V)
    return SVD