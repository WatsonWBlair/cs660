import numpy as np
import pandas as pd
import matplotlib as plt
import sympy as smp
from sympy.solvers import solve
from sympy import Symbol

def SVD_Setup(Matrix):
    #Build array, and transpose of that array
    Matrix=np.array(Matrix).copy()

    Transpose=np.transpose(Matrix)
    #Perform Matrix Multiplication to find AAt and AtA Matrices
    Multiplied_U=np.dot(Matrix, Transpose)

    Multiplied_V=np.dot(Transpose, Matrix)
    #Find Eigenvalues and Eigenvectors of AAt and AtA
    values_U,vectors_U=np.linalg.eig(Multiplied_U)

    values_V,vectors_V=np.linalg.eig(Multiplied_V)
    #Compute Singular Values, the square roots of the eignvalues of AtA.
    Sing_Values=np.sqrt(values_V)

    return Sing_Values

    def Create_SV_Array(Sing_Values, Matrix):
    #create a Singular Value Array, which fills the diagonal of a 0 matrix of the same dimensions 
    # as our original matrix with the singular values we calculated
        rows, columns=Matrix.shape
        
        SV_Array=np.zeros(rows,columns, int)
        
        np.fill_diagonal(SV_Array, Sing_Values)
    return SV_Array

    def Matrix_U(vectors_U):
    #Build matrix U, which consists of the Eigenvectors of AAt
        matrix_U=np.array(vectors_U)
        
    return matrix_U

    def Matrix_V(vectors_V):
    #Build Matrix V, which consists of the Eigenvectors of AtA   
        matrix_v=np.array(vectors_v)
        
    return matrix_v

    def SVD(matrix_U,SV_Array,matrix_V):
    #Perform Singular Value Decomposition, the dot product of Matrices U and A, and the Singular Values Array
        SVD=np.dot(matrix_, np.dot(SV_Array, matrix_v))
        
    return SVD

print(SVD_Setup(Matrix))
