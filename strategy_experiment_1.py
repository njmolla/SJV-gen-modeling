from run_gen_model import run_system
import numpy as np
#from mpi4py import MPI
import pickle
import pandas as pd
from pathlib import Path

num_samples = 300
#strategies = np.zeros((num_samples,1064))
strategies = []
stabilities = np.zeros((num_samples,4))
resource_user_key = 'rural communities'
#resource_user_key = 'small growers'
print(resource_user_key)
#resource_user_key = 'investor growers'
#resource_user_key = 'investor growers (white area)'
path = Path.cwd().joinpath('parameter_files', 'base', 'entity_list.xlsx')
entities_list = pd.read_excel(path,sheet_name=None, header=None)
N_list=entities_list['N'].values[:,0]
resource_user_num = np.nonzero(N_list==resource_user_key)[0][0]
converged_list = np.zeros(num_samples)


for i in range(num_samples):
  print('%s out of %s'%(i, num_samples))
  np.random.seed(i)
  stability_final, stability_1, stability_2, stability_3, converged, strategy_history, grad_history, strategy, J, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus = run_system(user = resource_user_num)
  if strategy == None:
    continue
  converged_list[i] = converged
  strategies.append(strategy)
  stabilities[i,0] = stability_1
  stabilities[i,1] = stability_2
  stabilities[i,2] = stability_3
  stabilities[i,3] = stability_final

strategies = np.array(strategies)  
with open('strategies_%s'%(resource_user_key), 'wb') as f:
  pickle.dump(strategies, f)

with open('stabilities_%s'%(resource_user_key), 'wb') as f:
  pickle.dump(stabilities, f)
  


