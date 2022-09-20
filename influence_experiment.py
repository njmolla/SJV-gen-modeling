from run_gen_model import run_system
import numpy as np
#from mpi4py import MPI
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
  stability_final, stability_1, stability_2, stability_3, converged, strategy_history, grad_history, strategy, J, eigvals, eigvectors, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus = run_system()
  
  seed += 1

  if stability_1 == True:
    right_eigenvectors = eigvectors
    eigvals, left_eigenvectors = np.linalg.eig(np.transpose(J))
    sensitivities = np.log(-np.sum(np.abs(right_eigenvectors)/eigvals,axis=1))
    sensitivities_list.append(sensitivities)
    influences = np.log(-np.sum(np.abs(left_eigenvectors)/eigvals,axis=1))
    influences_list.append(influences)
    seeds.append(seed)
    i += 1
  else:
    continue


with open('sensitivities_%s'%(parameterization), 'wb') as f:
  pickle.dump(sensitivities_list, f)

with open('influences_%s'%(parameterization), 'wb') as f:
  pickle.dump(influences_list, f)
  
## Graphing ################################################################
  
parameterization = 'v2'

with open('sensitivities_%s'%(parameterization), 'rb') as f:
  sensitivities_list = pickle.load(f)

with open('influences_%s'%(parameterization), 'rb') as f:
  influences_list = pickle.load(f)
  
influences_avg = np.mean(influences_list, axis=0)[3:3+N]
sensitivities_avg = np.mean(sensitivities_list, axis=0)[3:3+N]

with open('sensitivities_base', 'rb') as f:
  sensitivities_list_base = pickle.load(f)

with open('influences_base', 'rb') as f:
  influences_list_base = pickle.load(f)

influences_avg_base = np.mean(influences_list_base, axis=0)[3:3+N]
sensitivities_avg_base = np.mean(sensitivities_list_base, axis=0)[3:3+N]

entities = pd.read_excel('parameter_files\\%s\entity_list.xlsx'%(parameterization),sheet_name=None, header=None)
R_list = ['surface water', 'groundwater', 'groundwater quality']
N_list=entities['N'].values[:,0]
K_list=entities['K'].values[:,0]
M_list=entities['M'].values[:,0]

sns.set_style("darkgrid")
ax = sns.scatterplot(x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
sns.scatterplot(x = sensitivities_avg, y = influences_avg, hue = N_list, marker = 'x', s = 100)
plt.arrow(sensitivities_avg_base, influences_avg_base, sensitivities_avg, influences_avg)
plt.xlabel('Sensitivity', fontsize = 18)
plt.ylabel('Influence', fontsize = 18)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:N], labels[:N],bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig('comparison_%s.svg'%(parameterization))





# N = len(N_list)

# # RU influences
# sorted_RUs_indx = np.argsort(influences_avg[3:3+N])
# sorted_RUs = N_list[sorted_RUs_indx]

# entity_list = np.concatenate((R_list, N_list, K_list, M_list))

# plt.figure()
# #x = np.arange(len(entity_list))
# x = np.arange(N)
# plt.bar(x,influences_avg[3:3+N][sorted_RUs_indx], alpha=0.5)
# plt.xticks(x, sorted_RUs, rotation=30, ha='right')
# plt.ylabel('Influence')
# #plt.title('')
# plt.savefig('%s_influences_RUs.svg'%(parameterization),bbox_inches = 'tight')
# plt.show()

# # RU sensitivities
# sorted_RUs_indx = np.argsort(sensitivities_avg[3:3+N])
# sorted_RUs = N_list[sorted_RUs_indx]

# plt.figure()
# #x = np.arange(len(entity_list))
# x = np.arange(N)
# plt.bar(x,sensitivities_avg[3:3+N][sorted_RUs_indx], alpha=0.5)
# plt.xticks(x, sorted_RUs, rotation=30, ha='right')
# plt.ylabel('Sensitivity')
# #plt.title('')
# plt.savefig('%s_sensitivities_RUs.svg'%(parameterization),bbox_inches = 'tight')
# plt.show()