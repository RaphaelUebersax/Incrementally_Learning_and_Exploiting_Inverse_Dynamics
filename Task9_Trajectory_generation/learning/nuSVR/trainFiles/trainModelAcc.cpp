#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdio.h>
#include <iterator>
#include "yaml-cpp/yaml.h"

using namespace std;
 
int main()
{   
    YAML::Node paramLoaded = YAML::LoadFile("./../param.yaml");

    int totalJoints = paramLoaded["totalJoints"].as<int>();
    string savePath = paramLoaded["savePath"].as<string>();
    bool learnErrorModel = paramLoaded["learnErrorModel"].as<bool>();

    int startModelIndex = paramLoaded["startModelIndex"].as<int>();
    int endModelIndex = paramLoaded["endModelIndex"].as<int>();

    string trainDataFileName;
    string dataName;
    string resultName;

    if(learnErrorModel){
        trainDataFileName = savePath + "/sparseFormat/sparseErrorTrainData";
        dataName = "sparseErrorTrainData";
        resultName = "PredError";
    }
    else{
        trainDataFileName = savePath + "/sparseFormat/sparseTrainData";
        dataName = "sparseTrainData";
        resultName = "PredTau";
    }

    string dataExtension = ".dat";
    string modelExtension = ".dat.model";

    int process = paramLoaded["process"].as<int>(); 
    int kernel = paramLoaded["kernel"].as<int>();
    double epsilon = paramLoaded["epsilon"].as<double>();

    string trainAccFileName = "trainAcc.txt";
    std::ofstream outfile;

    for(int modelIndex = startModelIndex; modelIndex < endModelIndex; modelIndex = modelIndex + 1){

        outfile.open(trainAccFileName, std::ios_base::app); // append instead of overwrite
        outfile << "---------------------";
        outfile << "\n";

        string currenTrainDataFile = trainDataFileName + to_string(modelIndex) + dataExtension;
        string currentTrainedModelFile = dataName + to_string(modelIndex) + modelExtension;
        string currentResultFileName = savePath + "/SVM_Regression_Results/model" + to_string(modelIndex) + "Train" + resultName + ".csv";

        char testCommand_init[75];
        sprintf (testCommand_init, "export OMP_NUM_THREADS=8");
        system(testCommand_init);

        char testCommand[75];
        sprintf (testCommand, "./thundersvmPackage/build/bin/thundersvm-predict %s %s %s | tee -a %s", currenTrainDataFile.c_str(), currentTrainedModelFile.c_str(), currentResultFileName.c_str(), trainAccFileName.c_str());
        system(testCommand);
        outfile.close();
    }
}