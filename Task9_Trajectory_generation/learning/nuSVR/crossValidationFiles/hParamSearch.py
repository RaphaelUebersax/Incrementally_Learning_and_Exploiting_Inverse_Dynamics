#!/usr/bin/env python
import csv
import os.path
import pandas
import math
import numpy as np
import yaml
import io
from thundersvm import *
import random

def getMean(inputList):
  return np.mean(np.array(inputList))

def getStd(inputList):
  return np.std(np.array(inputList))

# learning params
with open('./../param.yaml',"r") as learningStream:
	paramLoaded = yaml.safe_load(learningStream)

savePath = paramLoaded["savePath"]
totalJoints = paramLoaded["totalJoints"]

gridSearchCList = paramLoaded["gridSearchCList"]
gridSearchGammaList = paramLoaded["gridSearchGammaList"]

crossValFoldCount = paramLoaded["crossValFoldCount"]
testFoldSize = paramLoaded["testFoldSize"]
learnErrorModel = paramLoaded["learnErrorModel"]
nu = paramLoaded["nu"]
startModelIndex = paramLoaded["startModelIndex"]

endModelIndex = paramLoaded["endModelIndex"]
if endModelIndex > totalJoints:
	endModelIndex = totalJoints

cvScoreFileName = "hyperParam_CV_scoreReport" # Contains extended information about the performance on different folds of test and train data
cvScoreFileName = os.path.join(os.getcwd(), cvScoreFileName + ".txt")

scoreFileName = "hyperParam_overall_scoreReport" # Contains overall information about the performance on complete test and train data
scoreFileName = os.path.join(os.getcwd(), scoreFileName + ".txt")

# write the file headers manually

# loading the train and test data sets
if learnErrorModel:
	trainDataFileName = "normalisedErrorTrainData"
	testDataFileName = "normalisedErrorTestData"
else:
	trainDataFileName = "normalisedTrainData"
	testDataFileName = "normalisedTestData"

trainDataFileName = os.path.join(savePath, trainDataFileName + ".csv")
testDataFileName = os.path.join(savePath, testDataFileName + ".csv")

trainData = np.genfromtxt(trainDataFileName, dtype=float, delimiter=',')
testData = np.genfromtxt(testDataFileName, dtype=float, delimiter=',')

# trainData = trainData[:10000]
# testData = testData[:10000]

print("Loaded train & test data")

# Randomly splitting the train data into crossValFoldCount subsets for doing crossValFoldCount-fold cross-validation
trainIndexList = list(range(np.shape(trainData)[0]))
random.shuffle(trainIndexList)

trainFolds = []

foldSize = int(np.shape(trainData)[0]/crossValFoldCount)

for fold in range(crossValFoldCount):
	foldDataIndices = trainIndexList[fold*foldSize:(fold+1)*foldSize]
	foldData = trainData[foldDataIndices]
	trainFolds = trainFolds + [foldData]

print("Prepared train-folds")

# Randomly splitting the test data into smaller test sets of size 10,000
testIndexList = list(range(np.shape(testData)[0]))
random.shuffle(testIndexList)

testFolds = []
testFoldCount = int(np.shape(testData)[0]/testFoldSize)
for fold in range(testFoldCount):
	foldDataIndices = testIndexList[fold*testFoldSize:(fold+1)*testFoldSize]
	foldData = testData[foldDataIndices]
	testFolds = testFolds + [foldData]

print("Prepared test-folds")

# Doing gridSearch on possible values of C and gamma and documenting the train, cross val and test accuracies

cvScoreFile = open(cvScoreFileName, "a+")
cvScoreFile.writelines(
	"\n" + "----------------------------------------------------------" + "\n")

scoreFile = open(scoreFileName, "a+")
scoreFile.writelines(
	"\n" + "----------------------------------------------------------" + "\n")


foldList = list(range(crossValFoldCount))

for modelIndex in range(startModelIndex, endModelIndex):

	cvScoreFile.writelines("Model Index: " + str(modelIndex) + "\n")
	scoreFile.writelines("Model Index: " + str(modelIndex) + "\n")

	x_Train = trainData[:,:3*totalJoints]
	y_Train = trainData[:,3*totalJoints + modelIndex]
	
	x_Test = testData[:,:3*totalJoints]
	y_Test = testData[:,3*totalJoints + modelIndex]
			
	
	for C in gridSearchCList:
		cvScoreFile.writelines("  C: " + str(C) + "\n")
		for gamma in gridSearchGammaList:

			print("Model = " + str(modelIndex) + " C = " + str(C) + " Gamma = " + str(gamma))

			scoreFile.writelines("  C = " + str(C) + ", Gamma = " + str(gamma) + "\n")
			cvScoreFile.writelines("      Gamma: " + str(gamma) + "\n")
			
			crossValSvCount = []

			crossValTrainR2 = []
			crossValValR2 = []
			crossValTestR2 = []

			crossValTrainAIC = []
			crossValValAIC = []
			crossValTestAIC = []

			# Training, Validation and Test accuracies during n-fold cross-validation process
			for fold in range(crossValFoldCount):
				print("Fold count: " + str(fold))

				crossValTestData = trainFolds[fold] # data for validation

				crossValTrainFolds = foldList[:fold] + foldList[fold+1:]
				crossValTrainData = trainFolds[crossValTrainFolds[0]] # data for training

				for trainFoldCount in range(1,len(crossValTrainFolds)):
					crossValTrainData = np.concatenate((crossValTrainData, trainFolds[crossValTrainFolds[trainFoldCount]]), axis=0)

				SVRmodel = NuSVR(kernel='rbf', nu=nu, C=C, gamma=gamma, verbose=True, tol=0.001, max_iter=50000)

				x_cvTrain = crossValTrainData[:,:3*totalJoints]
				y_cvTrain = crossValTrainData[:,3*totalJoints + modelIndex]

				SVRmodel.fit(x_cvTrain,y_cvTrain)

				x_cvTest = crossValTestData[:,:3*totalJoints]
				y_cvTest = crossValTestData[:,3*totalJoints + modelIndex]

				nSV = SVRmodel.n_sv
				trainR2 = SVRmodel.score(x_cvTrain,y_cvTrain)
				valR2 = SVRmodel.score(x_cvTest,y_cvTest)
				testR2 = SVRmodel.score(x_Test,y_Test)

				k = nSV*3*totalJoints + 1

				if trainR2 <= 0: # very poor model
					trainAIC = float("inf")
				else:
					trainAIC = 0.002*k - 2*math.log(trainR2)
				
				if valR2 <= 0: # very poor model
					valAIC = float("inf")
				else:
					valAIC = 0.002*k - 2*math.log(valR2)

				if testR2 <= 0: # very poor model
					testAIC = float("inf")
				else:
					testAIC = 0.002*k - 2*math.log(testR2)  

				crossValSvCount = crossValSvCount + [nSV]

				crossValTrainR2 = crossValTrainR2 + [trainR2]
				crossValValR2 = crossValValR2 + [valR2]
				crossValTestR2 = crossValTestR2 + [testR2]

				crossValTrainAIC = crossValTrainAIC + [trainAIC]
				crossValValAIC = crossValValAIC + [valAIC]
				crossValTestAIC = crossValTestAIC + [testAIC]

			cvScoreFile.writelines("        Cross Validation Data" + "\n")

			cvScoreFile.writelines("        SV Count 		: " + str(crossValSvCount) + "\n")

			cvScoreFile.writelines("        Train R2  		: " + str(crossValTrainR2) + "\n")
			cvScoreFile.writelines("        Mean Train R2  	: " + str(getMean(crossValTrainR2)) + "\n")
			cvScoreFile.writelines("        Std Train R2  	: " + str(getStd(crossValTrainR2)) + "\n")
			scoreFile.writelines("        Mean Train R2  	: " + str(getMean(crossValTrainR2)) + "\n")
			scoreFile.writelines("        Std Train R2  	: " + str(getStd(crossValTrainR2)) + "\n")

			cvScoreFile.writelines("        Val R2    		: " + str(crossValValR2) + "\n")
			cvScoreFile.writelines("        Mean Val R2   : " + str(getMean(crossValValR2)) + "\n")
			cvScoreFile.writelines("        Std Val R2 	: " + str(getStd(crossValValR2)) + "\n")
			scoreFile.writelines("		Mean Val R2   : " + str(getMean(crossValValR2)) + "\n")
			scoreFile.writelines("		Std Val R2 	: " + str(getStd(crossValValR2)) + "\n")


			cvScoreFile.writelines("        Test R2   		: " + str(crossValTestR2) + "\n")
			cvScoreFile.writelines("        Mean Test R2   	: " + str(getMean(crossValTestR2)) + "\n")
			cvScoreFile.writelines("        Std Test R2   	: " + str(getStd(crossValTestR2)) + "\n")
			scoreFile.writelines("		Mean Test R2   	: " + str(getMean(crossValTestR2)) + "\n")
			scoreFile.writelines("		Std Test R2   	: " + str(getStd(crossValTestR2)) + "\n")


			cvScoreFile.writelines("        Train AIC 		: " + str(crossValTrainAIC) + "\n")
			cvScoreFile.writelines("        Val AIC   		: " + str(crossValValAIC) + "\n")
			cvScoreFile.writelines("        Test AIC  		: " + str(crossValTestAIC) + "\n")

			# Training and testing with the entire train data for the C, gamma combination
			SVRmodel = NuSVR(kernel='rbf', nu=nu, C=C, gamma=gamma, verbose=True, tol=0.001, max_iter=50000)
			SVRmodel.fit(x_Train,y_Train)
			
			nSV = SVRmodel.n_sv
			trainR2 = SVRmodel.score(x_Train,y_Train)
			testR2 = SVRmodel.score(x_Test,y_Test)

			k = nSV*3*totalJoints + 1

			if trainR2 <= 0: # very poor model
				trainAIC = float("inf")
			else:
				trainAIC = 0.002*k - 2*math.log(trainR2)
			
			if testR2 <= 0: # very poor model
				testAIC = float("inf")
			else:
				testAIC = 0.002*k - 2*math.log(trainR2)   

			scoreFile.writelines("      SV Count  : " + str(nSV) + "\n")
			scoreFile.writelines("      Train R2  : " + str(trainR2) + "\n")
			scoreFile.writelines("      Train AIC : " + str(trainAIC) + "\n")
			scoreFile.writelines("      Test R2   : " + str(testR2) + "\n")
			scoreFile.writelines("      Test AIC  : " + str(testAIC) + "\n")
			scoreFile.writelines("\n")
		 
			# Testing with subsets of the test data
			extendedR2Scores = []
			extendedAICScores = []

			for fold in range(testFoldCount):
				subTestData = testFolds[fold]
				x_subTest = subTestData[:,:3*totalJoints]
				y_subTest = subTestData[:,3*totalJoints + modelIndex]
				
				subTestR2 = SVRmodel.score(x_subTest,y_subTest)
				
				if subTestR2 <= 0: # very poor model
					subTestAIC = float("inf")
				else:
					subTestAIC = 0.002*k - 2*math.log(subTestR2)   

				extendedR2Scores = extendedR2Scores + [subTestR2]

				extendedAICScores = extendedAICScores + [subTestAIC]

			cvScoreFile.writelines("\n")
			cvScoreFile.writelines("        Testing on sub-sets" + "\n")
			cvScoreFile.writelines("        Test R2   : " + str(extendedR2Scores) + "\n")
			cvScoreFile.writelines("        Test AIC  : " + str(extendedAICScores) + "\n")
			cvScoreFile.writelines("\n")

cvScoreFile.close()
scoreFile.close()