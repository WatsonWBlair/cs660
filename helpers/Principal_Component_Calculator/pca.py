import numpy as np
"""
Comps is the number of principal components to find
Data is the incoming data that we are finding the components of
"""
def pca(comps = 2, data = [[]]):
    # Normalize Data
    # normalizedData = normalizeData(data)
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

    projectedData = data.dot(sorted_eigenvectors)
   
    return [sorted_eigenvalues, sorted_eigenvectors, explained_variance, projectedData[:,:comps]]



def normalizeData(data):
    result=[]
    featureCount = len(data[0])
    for i in range(featureCount):
        row = data[:,i]
        normalized = (row - np.min(row)) / np.max(row)
        result.append(normalized)
    return np.transpose(result)

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
    observationCount = len(data)
    featureCount = len(data[0])
    variances = list()
    covMatrix = [ [0]*featureCount for i in range(featureCount)]

    # for number of features:
    for i in range(featureCount):
        # find the mean of feature N
        featureMean = np.sum(data[:,i]) / observationCount
        # subtract mean from all observations
        lessFeatureMean = data[:,i] - featureMean
        # take sum of the squares of differences
        squaredLessMean = pow(lessFeatureMean, 2)
        summedMeans = sum(squaredLessMean)
        # devide by 1 less than the number to get the sample variance
        variance = summedMeans/(len(data[:,i]) - 1)
        variances.append(variance)
    # print(variances)
    # print(covMatrix)

    # For each pair of variables
    for a in range(featureCount):
        for b in range(a, featureCount):
            if a == b:
                covMatrix[a][b] = variances[a]
            else:
                featureA_Mean = np.sum(data[:,a]) / observationCount
                featureB_Mean = np.sum(data[:,b]) / observationCount
                # Subtract the mean of the first feature from values of the first feature
                featureA = data[:,a] - featureA_Mean
                # subtract the mean of the second feature from values of the second feature
                featureB = data[:,b] - featureB_Mean
                # Multiply the corresponding observations (X1*Y1),(X2*Y2),... ect...
                # Sum the resultant products and divide by (n-1)
                productSumAB = sum(a*b for a, b in zip(featureA, featureB))
                covAB = productSumAB / (observationCount - 1)
                # Arrange values into covariance matrix. remember that cov(X,Y) === cov(Y,X)
                covMatrix[a][b] = covAB
                covMatrix[b][a] = covAB
    return covMatrix