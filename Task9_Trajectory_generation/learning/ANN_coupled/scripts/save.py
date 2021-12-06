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
	accuracies = "results_errorModel"
	gridScores = "gridScores_errorModel"
else:
	saveDirName = "torqueModels"
	accuracies = "results_torqueModel"
	gridScores = "gridScores_torqueModel"

modelSaveLocation = os.path.join(learntModelLoc,'ANN_Coupled')
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
accuracies = os.path.join(os.getcwd(), accuracies + ".csv")
gridScores = os.path.join(os.getcwd(), gridScores + ".csv")

shutil.copy(trainDataMean, saveDirName)
shutil.copy(trainDataStd, saveDirName)
shutil.copy(accuracies, saveDirName)
shutil.copy(gridScores, saveDirName)

# Renaming and transferring the learnt models
if learnErrorModel:
	old = os.path.join(os.getcwd(), 'errorModel'+ '.pt')
	new = os.path.join(os.getcwd(), 'error'+ '.pt')
	os.rename(old,new)
else:
	old = os.path.join(os.getcwd(), 'torqueModel'+ '.pt')
	new = os.path.join(os.getcwd(), 'torque'+ '.pt')
	os.rename(old,new)

shutil.copy(new, saveDirName)