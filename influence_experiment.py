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
    right_eigvals = eigvals
    right_eigenvectors = eigvectors
    left_eigvals, left_eigenvectors = np.linalg.eig(np.transpose(J))
    sensitivities = np.log(-np.sum(np.abs(right_eigenvectors)/right_eigvals,axis=1))
    sensitivities_list.append(sensitivities)
    influences = np.log(-np.sum(np.abs(left_eigenvectors)/left_eigvals,axis=1))
    influences_list.append(influences)
    seeds.append(seed)
    i += 1
  else:
    continue


with open('sensitivities_%s'%(parameterization), 'wb') as f:
  pickle.dump(sensitivities_list, f)

with open('influences_%s'%(parameterization), 'wb') as f:
  pickle.dump(influences_list, f)
  
