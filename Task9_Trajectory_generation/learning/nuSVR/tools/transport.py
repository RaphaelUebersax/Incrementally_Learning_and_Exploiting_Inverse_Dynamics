import yaml
import io
import os.path
import shutil

with open(r'./../param.yaml') as stream:
    paramLoaded = yaml.safe_load(stream)

# Transferring the files containing the mean and std of the recorded (processed) data 
meanStdDataPath = paramLoaded["savePath"]
meanStdDataPath = os.path.join(meanStdDataPath, 'sparseFormat')

meanFileName = "trainDataMean"
stdFileName = "trainDataStd"
meanFileName = os.path.join(meanStdDataPath, meanFileName + ".txt")
stdFileName = os.path.join(meanStdDataPath, stdFileName + ".txt")

newMeanPath = shutil.copy(meanFileName, os.getcwd())
newStdPath = shutil.copy(stdFileName, os.getcwd())
