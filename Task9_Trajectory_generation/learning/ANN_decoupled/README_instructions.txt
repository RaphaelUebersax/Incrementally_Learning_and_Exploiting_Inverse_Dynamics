1) Configure the param.yaml in main 'learning' directory

2) Learning the inverse dynamics:
	1_GRID_SEARCH_CV	- Searches the best (learning rate, hidden layer count, neurons per layer) combination
		> Survey the grid-search score file and select the best combination and edit the tuned values in:
			etaCoupled
			layersCoupled
			neuronsCoupled

	2_TRAIN_MODELS 		- Trains the models corresponding to the tuned C and Gamma values

	3_SAVE				- Renames and saves the models and relevant files to the learnt
						models directory