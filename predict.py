
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

import pickle
import joblib

name_list = ["hb_donor_cation","hb_acceptor_cation","nrot_cation","TPSA_cation","LogP_cation","MR_cation","n_rings_cation","fraction_sp3C_cation","weight_cation","n_heavy_cation","MPI_cation","Vmax_cation","Vmim_cation","dipole_cation","quadrupole_cation","hb_donor_anion","hb_acceptor_anion","nrot_anion","TPSA_anion","LogP_anion","MR_anion","n_rings_anion","fraction_sp3C_anion","weight_anion","n_heavy_anion","MPI_anion","Vmax_anion","Vmim_anion","dipole_anion","quadrupole_anion","hb_donor_solute","hb_acceptor_solute","nrot_solute","TPSA_solute","LogP_solute","MR_solute","n_rings_solute","fraction_sp3C_solute","weight_solute","n_heavy_solute","MPI_solute","Vmax_solute","Vmim_solute","dipole_solute","quadrupole_solute", "dG_solvation", "dG_partition"]

EMI_NTF_data=pd.read_csv('EMI_NTF',header=None,sep='\s+')
ref_partition=EMI_NTF_data.pop(EMI_NTF_data.shape[1]-1)
ref_solvation=EMI_NTF_data.pop(EMI_NTF_data.shape[1]-1)
EMI_NTF_feature=EMI_NTF_data

estimator_solvation=joblib.load('LGBM.solvation.joblib')
pred_solvation=estimator_solvation.predict(EMI_NTF_feature)
estimator_partition=joblib.load('LGBM.partition.joblib')
pred_partition=estimator_partition.predict(EMI_NTF_feature)

np.savetxt("ref_pred_solvation.dat", (pred_solvation,ref_solvation))
np.savetxt("ref_pred_partition.dat", (pred_partition,ref_partition))

