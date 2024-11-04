import numpy as np
import seaborn as sns
# from ..Eigne_Calculator.eigne import manual_eigen
"""
Comps is the number of principal components to find
Data is the incoming data that we are finding the components of
"""
def pca(comps = 2, data = [[]]):
    # Standardize Data
    standardizedData = standardize_data(data)
    
    # Compute Covariance Matrix
    covMatrix = covariance(standardizedData)

    # Call to Eigne_Calculator helper function
    eigenvalues, eigenvectors = manual_eigen(covMatrix)
    # eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
    
    # Create feature vector, sorted by their importance
    order_of_importance = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[order_of_importance]

    sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns
    
    # Compute Explained Variance
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

    projectedData = standardizedData.dot(sorted_eigenvectors)
   
    return [sorted_eigenvalues, sorted_eigenvectors, explained_variance, projectedData[:,:comps], covMatrix]


def standardizeData(data):
    result=[]
    featureCount = len(data[0])
    for i in range(featureCount):
        row = data[:,i]
        mean = np.mean(row)
        std = np.std(row)
        standardized = (row - mean) / std
        result.append(standardized)
    return np.transpose(result)


def calculateCovariance(data):
    featureCount = len(data[0])
    observationCount = len(data)
    covMatrix = [ [0]*featureCount for i in range(featureCount)]

    # For each pair of variables
    for a in range(featureCount):
        for b in range(a, featureCount):
            # Subtract the mean of the first feature from values of the first feature
            featureA_Mean = np.sum(data[:,a]) / observationCount
            featureA = data[:,a] - featureA_Mean

            # subtract the mean of the second feature from values of the second feature
            featureB_Mean = np.sum(data[:,b]) / observationCount
            featureB = data[:,b] - featureB_Mean

            # Sum the products of features (A1*B1)+(A2*B2)+....+(An*Bn)
            productSumAB = sum(a*b for a, b in zip(featureA, featureB))
            # and divide by (# of observations - 1)
            covAB = productSumAB / (observationCount - 1)
            
            # Insert cov(A,B) values into covariance matrix. remember that cov(A,B) === cov(B,A)
            covMatrix[a][b] = covAB
            covMatrix[b][a] = covAB
    return covMatrix

def mean(x): # based on np.mean(X, axis = 0)  
    return sum(x)/len(x)  

def std(x): # based on np.std(X, axis = 0)
    return (sum((i - mean(x))**2 for i in x)/len(x))**0.5

def standardize_data(X):
    return (X - mean(X))/std(X)

def covariance(x): # based on: np.matmul(np.array.T, np.array)/(np.array.shape[0]-1)
    n = x.shape[0]

    # Centering the data by subtracting the mean of each column (feature)
    mean_centered = x - x.mean()
    
    # Initializing an empty matrix to accumulate the covariances
    cov_matrix = np.zeros((x.shape[1], x.shape[1])) # todo: reproduce 'zeros' method manually

    # Adding each outer product to the covariance matrix
    for i in range(n):
        cov_matrix += np.outer(mean_centered[i], mean_centered[i]) # todo: reproduce 'outer' method manually
    
    # Dividing by (n - 1) to finalize the covariance matrix
    cov_matrix /= (n - 1)

    return cov_matrix


# function coppied from peer module because relative import seems to be broken
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
        I = identity(n)  # Create an identity matrix of size n
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
    return np.array(np.real_if_close(eigenvalues)), np.array([np.real_if_close(vec) for vec in eigenvectors])

def identity(shape):
    result = np.zeros((shape, shape))
    for i in range(shape):
        result[i][i] = 1
    return result
