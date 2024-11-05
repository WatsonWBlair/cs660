import numpy as np
from helpers.Eigne_Calculator.eigne import manual_eigen
from helpers.Principal_Component_Calculator.pca import standardize_data

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
    values_U,_=manual_eigen(MultipliedU)

    return values_U

def EigenvaluesV(MultipliedV):
    values_V,_=manual_eigen(MultipliedV)
    return values_V 

#Find Eigenvectors of Matrices U and V
def EigenvectorsU(MultipliedU):
    _,vectors_U=manual_eigen(MultipliedU)
    return vectors_U

def EigenvectorsV(MultipliedV):
    _,vectors_V=manual_eigen(MultipliedV)
    return vectors_V

#Compute Singular Values, the square roots of the eigenvalues of Matrix U.
def SingularValues(values_V):  
    Sing_Values=np.sqrt(values_V)
    Sing_Values=np.sort(Sing_Values)[::-1]
    return Sing_Values

#create a Singular Value Array, which fills the diagonal of a 0 matrix of the same dimensions
# as our original matrix with the singular values we calculated.
def CreateSVArray(Sing_Values, Matrix):
    rows, columns=Matrix.shape
    length=min(rows,columns)
    padded_zero_values= np.pad(Sing_Values,(0,length-len(Sing_Values)),'constant')
    SV_Array=np.zeros((rows,columns), dtype=float)
    np.fill_diagonal(SV_Array, padded_zero_values)
    return SV_Array

#Build matrices U and V, which consist of the Eigenvectors of U and V
def MatrixU(vectors_U):
    matrix_U=np.array(vectors_U)
    return matrix_U


def MatrixV(vectors_V):
    matrix_V=np.array(vectors_V)
    return matrix_V

#Perform Singular Value Decomposition, the dot product of Matrices U and A, and the Singular Values Array
def SVDBuild(matrix_U,SV_Array,matrix_V):
    SVD=np.dot(matrix_U, np.dot(SV_Array, matrix_V.T))
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