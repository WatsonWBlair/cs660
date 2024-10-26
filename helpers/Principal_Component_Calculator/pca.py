import numpy as np
import seaborn as sns
"""
Comps is the number of principal components to find
Data is the incoming data that we are finding the components of
"""
def pca(comps = 2, data = [[]]):
    # Standardize Data
    standardizedData = standardizeData(data)
    
    # Compute Covariance Matrix
    covMatrix = calculateCovariance(standardizedData)

    # Call to Eigne_Calculator helper function
    eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
    
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