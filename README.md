# IL_solv_ML

Predicting solvation and partition thermodynamics with machine-learning estimators. 

The py script loads the trained LGBM models and predicts solvation free energies and water-ionic-liquids transfer free energies with features from cheap ab initio calculations at HF/6-31G* and chemical structures. 

An illustrative example given is ~40 molecules in the ionic solvent EMI_NTF, where EMI=1-Ethyl-3-methylimidazolium and NTF=Bis(trifluoromethylsulfonyl)imide. The file EMI_NTF contains features and experimental references of solvation and partition thermodynamics. 
