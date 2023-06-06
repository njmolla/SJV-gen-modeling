from run_gen_model import run_system
import numpy as np
#from mpi4py import MPI
import pickle
import pandas as pd
from pathlib import Path
import scipy


num_samples = 300
from run_gen_model import run_system

parameterization = 'base'
print(parameterization)
sensitivities_list = []
partial_impacts = []
influences_list = []

seeds = []

i=0
seed = 0

while i < num_samples:
  print('%s out of %s'%(i, num_samples))
  np.random.seed(seed)
  stability_final, stability_1, stability_2, stability_3, converged, strategy_history, grad_history, strategy, J, eigvals, eigvectors, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus = run_system(parameterization = parameterization)
  
  seed += 1

  if stability_1 == True:
    eigvals, left_eigenvectors, right_eigenvectors = scipy.linalg.eig(J, left=True)
    # get L1 norm of eigenvectors to make sensitivities and influences sum up nicely
    right_eigenvectors = right_eigenvectors/np.sum(abs(right_eigenvectors),axis=0)
    left_eigenvectors = left_eigenvectors/np.sum(abs(left_eigenvectors),axis=0)
    # get total sensitivities (can get rid of these eventually, for validation purposes)
    sensitivities = -np.sum(np.abs(right_eigenvectors)/eigvals,axis=1)
    sensitivities_list.append(sensitivities)
    influences = -np.sum(np.abs(left_eigenvectors)/eigvals,axis=1)
    influences_list.append(influences)
    # impacts broken down by each variable
    partial_impact = np.transpose(np.abs(right_eigenvectors)@np.transpose(np.abs(left_eigenvectors)/-np.broadcast_to(eigvals,np.shape(right_eigenvectors))))
    partial_impacts.append(partial_impact)
    
    seeds.append(seed) # keep track of which seeds lead to stable systems
    i += 1
  else:
    continue



with open('data\influences_sensitivities\%s\sensitivities_%s'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(sensitivities_list, f)

with open('data\influences_sensitivities\%s\influences_%s'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(influences_list, f)

with open('data\influences_sensitivities\%s\partial_impacts_%s'%(parameterization,parameterization), 'wb') as f:
  pickle.dump(partial_impacts, f)
