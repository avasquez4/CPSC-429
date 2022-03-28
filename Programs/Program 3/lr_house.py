#https://gist.github.com/marcelcaraciolo/1321585#file-multlin-py

from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel

#Evaluate the linear regression

def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and 
    the standard deviation is 1. 
    '''
    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r


def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta)

    sqErrors = (predictions - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J


def gradientDescent(X, y, theta, alpha, numIterations):

	m = y.size
	xTrans = X.transpose()
	J_history = zeros(shape=(numIterations, 1))
	
	for i in range(0, numIterations):

		#compute prediction and error
		prediction = np.dot(X, theta)
		error = prediction - y
		
		#compute gradient
		gradient = np.dot(xTrans, error) / m
		
		# update theta
		theta = theta - alpha * gradient
		
		# compute cost of updated theta
		cost = compute_cost(X, y, theta)
		J_history[i, 0] = cost
		
	return theta, J_history
    
    
def main():
	#Load the dataset
	data = loadtxt('ex1data2.txt', delimiter=',')

	# build X (descriptive feature data) and y (target data)
	X = data[:, :2]
	y = data[:, 2]


	#number of training samples
	m = y.size
	y.shape = (m, 1)

	#Scale features and set them to zero mean
	x, mean_r, std_r = feature_normalize(X)

	#Add a column of ones to X (interception data)
	it = ones(shape=(m, 3))
	it[:, 1:3] = x

	#Some gradient descent settings
	iterations = 100
	alpha = 0.01

	#Init Theta and Run Gradient Descent
	theta = zeros(shape=(3, 1))

	theta, J_history = gradientDescent(it, y, theta, alpha, iterations)
	
	plot(arange(iterations), J_history)
	xlabel('Iterations')
	ylabel('Cost Function')
	show()

	# Predict price of a 1650 sq-ft 3 br house
	price = array([1.0,   ((1650.0 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])]).dot(theta)
	print ('Predicted price of a 1650 sq-ft, 3 br house: %f' % (price))

#invoke the main method
main()
