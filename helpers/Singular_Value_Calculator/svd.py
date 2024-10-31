import numpy as np
import pandas as pd
import matplotlib as plt
import sympy as smp
from sympy.solvers import solve
from sympy import Symbol
Matrix=np.random.randint(1,10,size=(6,6))

def SVDSetup(Matrix):
    #Build array, and transpose of that array
    Matrix=np.array(Matrix).copy()

    Transpose=np.transpose(Matrix)
    #Perform Matrix Multiplication to find AAt and AtA Matrices
    MultipliedU=np.dot(Matrix, Transpose)

    MultipliedV=np.dot(Transpose, Matrix)
    return MultipliedU, MultipliedV

def Eigen(MultipliedU, MultipliedV):
    #Find Eigenvalues and Eigenvectors of AAt and AtA
    values_U,vectors_U=np.linalg.eig(MultipliedU)

    values_V,vectors_V=np.linalg.eig(MultipliedV)
    return values_U, vectors_U, values_V, vectors_V
def SingularValues(values_U, values_V):
    #Compute Singular Values, the square roots of the eignvalues of AtA.
    Sing_Values=np.sqrt(values_V)

    return Sing_Values

def CreateSVArray(Sing_Values, Matrix):
#create a Singular Value Array, which fills the diagonal of a 0 matrix of the same dimensions 
# as our original matrix with the singular values we calculated
    rows, columns=Matrix.shape
        
    SV_Array=np.zeros((rows,columns), dtype=int)
        
    np.fill_diagonal(SV_Array, Sing_Values)
    return SV_Array

def MatrixU(vectors_U):
#Build matrix U, which consists of the Eigenvectors of AAt
    matrix_U=np.array(vectors_U)
        
    return matrix_U

def MatrixV(vectors_V):
#Build Matrix V, which consists of the Eigenvectors of AtA   
    matrix_V=np.array(vectors_V)
        
    return matrix_V

def SVDBuild(matrix_U,SV_Array,matrix_V):
#Perform Singular Value Decomposition, the dot product of Matrices U and A, and the Singular Values Array
    SVD=np.dot(matrix_U, np.dot(SV_Array, matrix_V))
        
    return SVD

def SVD(Matrix):
  MultipliedU,MultipliedV=SVDSetup(Matrix)
  values_U,vectors_U,values_V,vectors_V=Eigen(MultipliedU,MultipliedV)
  Sing_Values=SingularValues(values_U,values_V)
  SV_Array=CreateSVArray(Sing_Values, Matrix)
  matrix_U=MatrixU(vectors_U)
  matrix_V=MatrixV(vectors_V)
  SVD=SVDBuild(matrix_U,SV_Array,matrix_V)
  return SVD
print(SVD(Matrix))