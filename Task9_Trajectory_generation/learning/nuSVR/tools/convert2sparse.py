import os
import yaml
import io
import os.path
import shutil
import numpy as np
from sklearn.datasets import dump_svmlight_file

with open('./../param.yaml',"r") as learningStream:
	paramLoaded = yaml.safe_load(learningStream)

savePath = paramLoaded["savePath"]
totalJoints = paramLoaded["totalJoints"]
learnErrorModel = paramLoaded["learnErrorModel"]
learntModelLoc = paramLoaded["learntModelLoc"]

testDataFileName = "testJointData"
meanDataFileName = "trainDataMean"
stdDataFileName = "trainDataStd"

# Get mean and std from leart model's location
with open('./../param.yaml',"r") as learningStream:
	paramLoaded = yaml.safe_load(learningStream)

testDataFileName = "testJointData"

meanDataFileName = "trainDataMean"
stdDataFileName = "trainDataStd"

print("> Loading the test data")

testDataFileName = os.path.join(savePath, testDataFileName + ".csv")
meanDataFileName = os.path.join(learntModelLoc, meanDataFileName + ".txt")
stdDataFileName = os.path.join(learntModelLoc, stdDataFileName + ".txt")

testData = np.genfromtxt(testDataFileName, dtype=float, delimiter=',') # np array of 7 columns with each column corresponding to that joint's torque

meanData = np.genfromtxt(meanDataFileName, dtype=float) # Contains the mean and std of the training data on which the model was trained
stdData = np.genfromtxt(stdDataFileName, dtype=float)   # Will be used to normalise the test data before making predictions

meanData = np.reshape(meanData, (-1,3*totalJoints))
stdData = np.reshape(stdData, (-1,3*totalJoints))

testDataSize = np.shape(testData)[0]

print("> Test data size: ", testDataSize)

print("> Normalising the test data")

test_input = testData[:testDataSize, 0:3*totalJoints]
test_input = (test_input - meanData)/stdData

print("> Saving in sparse format")
resultDirectory = os.path.join(savePath, 'sparseFormat') 
if not os.path.exists(resultDirectory):
	os.makedirs(resultDirectory)


dummyLabels = np.zeros((np.shape(testData)[0],))

for modelIndex in range(totalJoints):

	sparseTestDataFileName = "sparseTestData" + str(modelIndex)
	sparseTestDataFileName = os.path.join(resultDirectory, sparseTestDataFileName + ".dat")
	dump_svmlight_file(test_input, dummyLabels, sparseTestDataFileName, zero_based=False)

	sparseErrorTestDataFileName = "sparseErrorTestData" + str(modelIndex)
	sparseErrorTestDataFileName = os.path.join(resultDirectory, sparseErrorTestDataFileName + ".dat")
	dump_svmlight_file(test_input, dummyLabels, sparseErrorTestDataFileName, zero_based=False)