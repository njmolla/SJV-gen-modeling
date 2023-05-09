from run_gen_model import run_system
import numpy as np
#from mpi4py import MPI
import pickle
import pandas as pd
from pathlib import Path


num_samples = 300
from run_gen_model import run_system

parameterization = 'v4'
print(parameterization)
sensitivities_list = []
sensitivities_water_list = []
sensitivities_social_list = []
influences_list = []
influences_water_list = []
influences_social_list = []
seeds = []

i=0
seed = 0

while i < num_samples:
  print('%s out of %s'%(i, num_samples))
  np.random.seed(seed)
  stability_final, stability_1, stability_2, stability_3, converged, strategy_history, grad_history, strategy, J, eigvals, eigvectors, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus = run_system(parameterization = parameterization)
  
  seed += 1

  if stability_1 == True:
    right_eigvals = eigvals
    right_eigenvectors = eigvectors
    left_eigvals, left_eigenvectors = np.linalg.eig(np.transpose(J))
    sensitivities = np.log(-np.sum(np.abs(right_eigenvectors)/right_eigvals,axis=1))
    sensitivities_list.append(sensitivities)
    influences = np.log(-np.sum(np.abs(left_eigenvectors)/left_eigvals,axis=1))
    influences_list.append(influences)
    # sensitivities and influences broken down by contribution from resources and social interactions
    sensitivities_water = np.log(-np.sum(np.abs(right_eigenvectors[:,0:3])/right_eigvals[0:3],axis=1))
    sensitivities_water_list.append(sensitivities_water)
    sensitivities_social = np.log(-np.sum(np.abs(right_eigenvectors[:,3:])/right_eigvals[3:],axis=1))
    sensitivities_social_list.append(sensitivities_social)   
    
    influences_water = np.log(-np.sum(np.abs(left_eigenvectors[:,0:3])/left_eigvals[0:3],axis=1))
    influences_water_list.append(influences_water)
    influences_social = np.log(-np.sum(np.abs(left_eigenvectors[:,3:])/left_eigvals[3:],axis=1))
    influences_social_list.append(influences_social)
    
    seeds.append(seed) # keep track of which seeds lead to stable systems
    i += 1
  else:
    continue


with open('data\influences_sensitivities\%s\sensitivities_%s_water'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(sensitivities_water_list, f)
  
with open('data\influences_sensitivities\%s\sensitivities_%s_social'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(sensitivities_social_list, f)

with open('data\influences_sensitivities\%s\influences_%s_water'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(influences_water_list, f)
  
with open('data\influences_sensitivities\%s\influences_%s_social'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(influences_social_list, f)

with open('data\influences_sensitivities\%s\sensitivities_%s'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(sensitivities_list, f)

with open('data\influences_sensitivities\%s\influences_%s'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(influences_list, f)

