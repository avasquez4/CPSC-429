# Decision Tree Implementation

import numpy as np

class dtree:
	""" A basic Decision Tree"""
	
	def __init__(self):
		""" Constructor """

	def read_data(self,filename):
		"""Read data in and save them into three variables
		   data: store all data for the features (class excluded)
		   classes: store all data for the class
		   featureNames: stores all feature names
		   featureValues: store each feature's unique values
		"""

		fid = open(filename,"r")
		data = []
		d = []
		for line in fid.readlines():
			d.append(line.strip())
		for d1 in d:
			data.append(d1.split(",")) # list of lists
		fid.close()

		self.featureNames = data[0] # first row as feature names
		self.targetName = self.featureNames[-1]
		self.featureNames = self.featureNames[:-1]

		data = data[1:] # remove the first row
		self.classData = []
		for d in range(len(data)):
			self.classData.append(data[d][-1]) # extract last column 
			data[d] = data[d][:-1]	# remove the last column in data

		# extract unique values values for each feature
		transposedData = np.transpose(np.copy(data))
		self.featureValues={}
		for i in range(len(self.featureNames)):
			self.featureValues[self.featureNames[i]] = np.unique(transposedData[i])
		print(self.featureValues)

		return data,self.classData,self.featureNames
       

	def ID3(self,data,classData,featureNames, parentMajority):

		""" The ID3 algorithm, which recursively constructs the tree"""
		
		nData = len(data)
		nClasses = len(classData)

		# base case 1: if D is empty, return the parentMajority class
		if nData==0 and nClasses==0:
			 return parentMajority
			 
		# get the number of features
		nFeatures = 0
		if nData != 0:
			nFeatures = len(data[0])
			
		# find the majority of target value
		majority = self.majority_class(classData)

		# base case 2: if d is empty (no features), return the majority class
		if nFeatures == 0 :
			return majority

		# base case 3: if all instances have the same target value, return the first target value
		elif classData.count(classData[0]) == nData:
			return classData[0]
		
		# general case to recursively build the tree
		else:

			# Choose the best feature based on information gain
			gain = np.zeros(nFeatures)
			for feature in range(nFeatures):
				gain[feature] = self.info_gain(data,classData,feature)
			bestFeature = np.argmax(gain)
			bestFeatureName = featureNames[bestFeature]
			
			tree = {bestFeatureName:{}}
			#print "The tree %s afer the best feature %s" % (tree, bestFeatureName)

			# Load the bestFeature's possible values into a list
			values = []
			for i in range(len(self.featureValues[bestFeatureName])):
				values.append(self.featureValues[bestFeatureName][i])
			#print "The best feature %s values %s" % (bestFeatureName, str(values))

			# Partition the original datapoints based on the best feature possible values
			# and then recursively invoke ID algorithm to build subtrees
			for value in values:
				newData = []
				newClassData = []
				index = 0
				newNames = []

				# partition the data
				for datapoint in data:
					if datapoint[bestFeature]==value:
						if bestFeature==0:
							newdatapoint = datapoint[1:]
							newNames = featureNames[1:]
						elif bestFeature==nFeatures:
							newdatapoint = datapoint[:-1]
							newNames = featureNames[:-1]
						else:
							newdatapoint = datapoint[:bestFeature]
							newdatapoint.extend(datapoint[bestFeature+1:])
							newNames = featureNames[:bestFeature]
							newNames.extend(featureNames[bestFeature+1:])

						newData.append(newdatapoint)
						newClassData.append(classData[index])
					index += 1

				# Now do recursive call to build the subtrees
				subtree = self.ID3(newData,newClassData,newNames, majority)

				# Add the subtree on to the tree
				#print "The subtree %s for the current tree %s" % ( subtree, tree,)
				tree[bestFeatureName][value] = subtree

			return tree

	def classify(self, tree, datapoint):
		""" classfication on a new datapoint using a tree"""

		if type(tree) == type("string"):
			return tree
		else:
			a = list(tree.keys())[0]
			for i in range(len(self.featureNames)):
				if self.featureNames[i]==a:
					break
			
			try:
				t = tree[a][datapoint[i]]
				return self.classify(t,datapoint)
			except:
				return None


	def classifyAll(self,tree,data):
		""" classfication on a set of data using a tree"""

		results = []
		for i in range(len(data)):
			results.append(self.classify(tree,data[i]))
		return results
	


	def printTree(self, tree, str):
		""" print out the decision tree"""

		if type(tree) == dict:
			for item in list(tree.values())[0].keys():
					print("%s %s = %s " % (str, list(tree.keys())[0], item))
					self.printTree(list(tree.values())[0][item], str + "\t")
		else:
			print("%s -> %s = %s" % (str, self.targetName, tree))


	def entropy(self,classData):
		""" calculate the entropy based on classData"""

		###### your implementation below ######

		(unique, counts) = np.unique(classData,return_counts=True)

		size = len(self.classData)

		totalEntropy= 0

		for count in counts:
			totalEntropy += -(count/size)*np.log2(count/size)
		
		return np.around(totalEntropy, decimals=4)


	def info_gain(self,data,classData,featureIndex):
		""" Calculate informatin information"""

		###### your implementation below ######

		#find all feature values for a given feature at feature index
		featureValues = [] #a list that holds all feature values of a given feature
		for column in data:
			featureValues.append(column[featureIndex])

		(unique, counts) = np.unique(featureValues,return_counts=True)

		#calculate partition entropy for each unique feature value of a given feature
		partitionEntropy = []
		for uniqueFeature in unique:
			index = 0
			targetValues = [] #a list that holds the target values for each feature value of a given feature
			for value in featureValues:
				if uniqueFeature == value:
					targetValues.append(classData[index])
				index+=1 
			partitionEntropy.append(self.entropy(targetValues))

		#calculate remainder using the list of partition entropy
		remainder = 0
		index = 0
		for entropy in partitionEntropy:
			remainder+=entropy*(counts[index]/sum(counts))
			index+=1
		
		#calculate information gain using remainder
		gain = self.entropy(classData) - remainder

		return np.around(gain, decimals=4)

	def majority_class (self, classData):
		""" find the majority of class"""
		
		###### your implementation below ######

		(unique, counts) = np.unique(classData,return_counts=True)

		currIndex, maxIndex = 0, 0	
		max = counts[0]
		for count in counts:
			if count > max:
				max = count
				maxIndex = currIndex
			currIndex += 1
 
		return unique[maxIndex]


	def predictionAccuracy(self, predicted, actual): 
		""" compute prediction accuracy on predicted data"""

		###### your implementation below ######

		correctCount = 0
		i = 0
		for x in predicted:
			if x == actual[i]:
				correctCount+=1
			i+=1

		accuracyCount = (correctCount / len(predicted)) * 100

		return np.around(accuracyCount, decimals=1)

