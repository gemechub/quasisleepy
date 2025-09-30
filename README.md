# quasisleepy
Sleep Wake Transition predictions from network features.
Utils script has functions that can be used to created windowed features and split the data into transductive and inductive train-test splits. Notebooks contain the xgboost transition prediction pipeline.
  -Transductive : train-test split is in block from all animals
  -Inductive : train-test split where some animals are left out for testing (not used in training)
  - Multi: NREM-to-Wake, Wake-to-NREM, Stay in NREM, Stay in Wake classes
  - Binary: Transition vs No-transition
    
