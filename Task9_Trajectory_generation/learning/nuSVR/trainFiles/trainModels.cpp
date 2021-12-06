#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdio.h>
#include <sstream>
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
 
    // Parameters for nuSVR
    int process = paramLoaded["process"].as<int>(); 
    int kernel = paramLoaded["kernel"].as<int>();
    double epsilon = paramLoaded["epsilon"].as<double>();
    double error = paramLoaded["error"].as<double>();
    double nu = paramLoaded["nu"].as<double>();

    // Tuned C and Gamma values for each joint model
    vector<double> CList = paramLoaded["tunedCList"].as<std::vector<double>>();
    vector<double> gammaList = paramLoaded["tunedGammaList"].as<std::vector<double>>();

    string trainDataFileName;

    if(learnErrorModel){
        trainDataFileName = savePath + "/sparseFormat/sparseErrorTrainData";
    }
    else{
        trainDataFileName = savePath + "/sparseFormat/sparseTrainData";
    }

    string dataExtension = ".dat";
    string modelExtension = ".dat.model";

    for(int modelIndex = startModelIndex; modelIndex < endModelIndex; modelIndex = modelIndex + 1){
    
        string currenTrainDataFile = trainDataFileName + to_string(modelIndex) + dataExtension;

        double currentGamma = gammaList[modelIndex];
        double currentC = CList[modelIndex];

        char trainCommand_init[80];
        sprintf (trainCommand_init, "export OMP_NUM_THREADS=8");
        system(trainCommand_init);

 		char trainCommand[80];
        sprintf (trainCommand, "./thundersvmPackage/build/bin/thundersvm-train -s %s -t %s -g %s -c %s -e %s -n %s -m %s %s", to_string(process).c_str(), to_string(kernel).c_str(), to_string(currentGamma).c_str(), to_string(currentC).c_str(), to_string(error).c_str(), to_string(nu).c_str(), to_string(5500).c_str(), currenTrainDataFile.c_str());   
        system(trainCommand);
    }
}