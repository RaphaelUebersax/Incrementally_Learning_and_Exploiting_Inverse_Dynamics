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
else:
	saveDirName = "torqueModels"

modelSaveLocation = os.path.join(learntModelLoc,'nuSVR')
if not os.path.isdir(modelSaveLocation):
	os.mkdir(modelSaveLocation)

saveDirName = os.path.join(modelSaveLocation,saveDirName)
if not os.path.isdir(saveDirName):
	os.mkdir(saveDirName)

# Transfering data accompanying the model
trainDataMean = "trainDataMean"
trainDataStd = "trainDataStd"
trainAcc = "trainAcc"
testAcc = "testAcc"
gridSearchScores_CV = "hyperParam_CV_scoreReport"
gridSearchScores_overall = "hyperParam_overall_scoreReport"

trainDataMean = os.path.join(os.getcwd(), trainDataMean + ".txt")
trainDataStd = os.path.join(os.getcwd(), trainDataStd + ".txt")
trainAcc = os.path.join(os.getcwd(), trainAcc + ".txt")
testAcc = os.path.join(os.getcwd(), testAcc + ".txt")
gridSearchScores_CV = os.path.join(os.getcwd(), gridSearchScores_CV + ".txt")
gridSearchScores_overall = os.path.join(os.getcwd(), gridSearchScores_overall + ".txt")

shutil.copy(trainDataMean, saveDirName)
shutil.copy(trainDataStd, saveDirName)
shutil.copy(trainAcc, saveDirName)
shutil.copy(testAcc, saveDirName)
shutil.copy(gridSearchScores_CV, saveDirName)
shutil.copy(gridSearchScores_overall, saveDirName)

# Renaming and transferring the learnt models
for i in range(totalJoints):
	if learnErrorModel:
	    old = os.path.join(os.getcwd(), 'sparseErrorTrainData'+ str(i) + '.dat.model')
	    new = os.path.join(os.getcwd(), 'error'+ str(i) + '.dat.model')
	    os.rename(old,new)
	else:
	    old = os.path.join(os.getcwd(), 'sparseTrainData'+ str(i) + '.dat.model')
	    new = os.path.join(os.getcwd(), 'torque'+ str(i) + '.dat.model')
	    os.rename(old,new)

	shutil.copy(new, saveDirName)