1) Configure the param.yaml in main 'learning' directory

2) Add the thundersvm package in 'thundersvmPackage' and build.

3) Build the target files here:

	mkdir build
	cd build
	cmake ..
	make

4) Learning the inverse dynamics:

	./1_GRID_SEARCH_CV.sh	- Searches the best C and Gamma combinations from C and 
							  Gamma list 
									> Survey the grid-search score file and select the best C and Gamma combinations and edit the tuned C and gamma lists in the param.yaml file

	./2_TRAIN_MODELS.sh 	- Trains the models corresponding to the tuned C and Gamma 
							values

	./3_TRAIN_ACC.sh		- Produces the train accuracy for the learned models
							- Also writes predicted torque (or error) values to PredTau (or PredError) csv files in the savePath directory

	./4_TEST_MODELS.sh		- Produces the test accuracy for the learned models (using the test set)
							- Also writes predicted torque (or error) values to PredTau (or PredError) csv files in the savePath directory

	./5_SAVE.sh 			- Renames the models and saves the relevant files to the 
							learnt models directory



5) OPTIONAL: If goal is only to test the model predictions on some novel test set, use 6_TEST.sh