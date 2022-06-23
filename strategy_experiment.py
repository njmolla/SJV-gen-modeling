from run_gen_model import run_system
import numpy as np
#from mpi4py import MPI
import pickle
import pandas as pd

num_samples = 200
strategies = np.zeros((num_samples,1064))
stabilities = np.zeros((num_samples,4))
resource_user_key = 'rural communities'
#resource_user_key = 'small growers'
#resource_user_key = 'investor growers'
#resource_user_key = 'investor growers (white area)'
entities = pd.read_excel('parameter_files\entity_list.xlsx',sheet_name=None, header=None)
N_list=entities['N'].values[:,0]
resource_user_num = np.nonzero(N_list==resource_user_key)[0][0]


for i in range(num_samples):
  np.random.seed(i)
  stability_final, stability_1, stability_2, stability_3, converged, strategy_history, grad_history, strategy, J, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus = run_system(user = resource_user_num)
  strategies[i] = strategy
  stabilities[i,0] = stability_1
  stabilities[i,1] = stability_2
  stabilities[i,2] = stability_3
  stabilities[i,3] = stability_final
  
seed = 'looping'
with open('strategies_%s_%s'%(resource_user_key,seed), 'wb') as f:
  pickle.dump(strategies, f)

with open('stabilities_%s_%s'%(resource_user_key,seed), 'wb') as f:
  pickle.dump(stabilities, f)
  

#effort_G = np.average(strategies[:,0:R*M*N],axis=0)


