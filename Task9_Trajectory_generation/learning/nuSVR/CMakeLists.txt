find_package(yaml-cpp REQUIRED) 
set(PCL_BUILD_TYPE Release) 

add_executable (trainModels trainFiles/trainModels.cpp) 
add_executable (trainModelAcc trainFiles/trainModelAcc.cpp) 
add_executable (testModels testFiles/testModels.cpp) 
add_executable (testModelsOptional testFiles/testModelsOptional.cpp) 

target_link_libraries (trainModels yaml-cpp)
target_link_libraries (trainModelAcc yaml-cpp)
target_link_libraries (testModels yaml-cpp)
target_link_libraries (testModelsOptional yaml-cpp)