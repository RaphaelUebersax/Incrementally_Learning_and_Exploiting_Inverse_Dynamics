#!/usr/bin/env python
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import tools
import yaml

with open('./../param.yaml',"r") as learningStream:
	paramLoaded = yaml.safe_load(learningStream)

savePath = paramLoaded["savePath"]
totalJoints = paramLoaded["totalJoints"]
learnErrorModel = paramLoaded["learnErrorModel"]

nb_folds = paramLoaded["crossValFoldCount"]
eta_grid = paramLoaded["etaGrid"]
nb_hidden_layers_grid = paramLoaded["hiddenLayerGrid"]
nb_hidden_neurons_grid = paramLoaded["hiddenNeuronGrid"]

startModelIndex = paramLoaded["startModelIndex"]
endModelIndex = paramLoaded["endModelIndex"]

if learnErrorModel:
	trainDataFileName = "normalisedErrorTrainData"
	testDataFileName = "normalisedErrorTestData"
else:
	trainDataFileName = "normalisedTrainData"
	testDataFileName = "normalisedTestData"

trainDataFileName = os.path.join(savePath, trainDataFileName + ".csv")
testDataFileName = os.path.join(savePath, testDataFileName + ".csv")

trainData = np.genfromtxt(trainDataFileName, dtype=float, delimiter=',') # np array of 21 columns (q, qdot, qddot)
testData = np.genfromtxt(testDataFileName, dtype=float, delimiter=',') # np array of 7 columns with each column corresponding to that joint's torque

np.random.shuffle(trainData)

train_data = torch.from_numpy(trainData)
test_data = torch.from_numpy(testData)

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')          

train_data = train_data.float()
test_data = test_data.float()

debug = False

# List of seed to use for randomization
seeds = [0]

# Train network
for jointIndex in range(startModelIndex,endModelIndex):
	print("Model Index: " + str(jointIndex))

	results = tools.grid_search_loop(train_data, test_data, device, seeds, 
										nb_folds, eta_grid, nb_hidden_layers_grid, 
										nb_hidden_neurons_grid, debug, jointIndex)

	print("Model Index: " + str(jointIndex) + " results")		

	print(results)

	# Save grid search scores
	header = "learning_rate,hidden_layers,hidden_neurons,train_mean,train_std,val_mean,val_std,test_mean,test_std,param_count"

	if learnErrorModel:
		np.savetxt('gridScores_errorModel'+ str(jointIndex) + '.csv',results.numpy(), delimiter=',',header=header,comments='')	
	else:
		np.savetxt('gridScores_torqueModel'+ str(jointIndex) + '.csv',results.numpy(), delimiter=',',header=header,comments='')
