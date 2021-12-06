import os
import yaml
import io
import os.path
import shutil

with open(r'./../param.yaml') as stream:
    paramLoaded = yaml.safe_load(stream)

learnErrorModel = paramLoaded["learnErrorModel"]
savePath = paramLoaded["savePath"]
learntModelLoc = paramLoaded["learntModelLoc"]
totalJoints = paramLoaded["totalJoints"]

# Making a directory in learnt model folder
learntModelLoc = os.path.join(learntModelLoc, os.path.basename(savePath))
if not os.path.isdir(learntModelLoc):
	os.mkdir(learntModelLoc)

if learnErrorModel:
	saveDirName = "errorModels" 	
	accuracies_base = "results_errorModel"
	gridScores_base = "gridScores_errorModel"
else:
	saveDirName = "torqueModels"
	accuracies_base = "results_torqueModel"
	gridScores_base = "gridScores_torqueModel"

modelSaveLocation = os.path.join(learntModelLoc,'ANN_Decoupled')
if not os.path.isdir(modelSaveLocation):
	os.mkdir(modelSaveLocation)

saveDirName = os.path.join(modelSaveLocation,saveDirName)
if not os.path.isdir(saveDirName):
	os.mkdir(saveDirName)

# Transfering data accompanying the model
trainDataMean = "trainDataMean"
trainDataStd = "trainDataStd"

trainDataMean = os.path.join(os.getcwd(), trainDataMean + ".txt")
trainDataStd = os.path.join(os.getcwd(), trainDataStd + ".txt")

shutil.copy(trainDataMean, saveDirName)
shutil.copy(trainDataStd, saveDirName)

for jointIndex in range(totalJoints):

	accuracies = os.path.join(os.getcwd(), accuracies_base + str(jointIndex) + ".csv")
	gridScores = os.path.join(os.getcwd(), gridScores_base + str(jointIndex) + ".csv")

	shutil.copy(accuracies, saveDirName)
	shutil.copy(gridScores, saveDirName)

	# Renaming and transferring the learnt models
	if learnErrorModel:
		old = os.path.join(os.getcwd(), 'errorModel' + str(jointIndex) + '.pt')
		new = os.path.join(os.getcwd(), 'error' + str(jointIndex) + '.pt')
		os.rename(old,new)
	else:
		old = os.path.join(os.getcwd(), 'torqueModel' + str(jointIndex) + '.pt')
		new = os.path.join(os.getcwd(), 'torque' + str(jointIndex)  + '.pt')
		os.rename(old,new)

	shutil.copy(new, saveDirName)