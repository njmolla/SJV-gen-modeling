from run_gen_model import run_system
import numpy as np
#from mpi4py import MPI
import pickle
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest
import matplotlib.pyplot as plt

num_samples = 1000

parameterizations = ['base','v1','v2','v3','v4']

data = pd.DataFrame(index=parameterizations, columns = ['proportion stable','CI low', 'CI high'])

for parameterization in parameterizations:
    print(parameterization)

    seed = 0
    stable = 0

    for i in range(num_samples):
      print('%s out of %s'%(i, num_samples))
      np.random.seed(seed)
      stability_final, stability_1, stability_2, stability_3, converged, strategy_history, grad_history, strategy, J, eigvals, eigvectors, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus = run_system(parameterization = parameterization)
      
      seed += 1
    
      if stability_final == True:
          stable += 1
    # save proportion stable in dataframe
    data.loc[parameterization, 'proportion stable'] = stable/num_samples
    #calculate confidence intervals
    result = binomtest(stable, num_samples).proportion_ci()
    data.loc[parameterization, ['CI low', 'CI high']] = [result[0], result[1]]


CI_low = data['proportion stable'].values - data['CI low'].values
CI_high = data['CI high'].values - data['proportion stable'].values
ax = data.plot.bar(y='proportion stable', yerr = np.stack([CI_low, CI_high]))
ax.figure.savefig('Proportion_Stable.svg')

# p1 = 203/500
# p2 = 193/500
# p = num_samples*(p1+p2)/(num_samples*2)
# z = (p1-p2)/np.sqrt(p*(1-p)*((1/num_samples)*2))

# #find p-value
# scipy.stats.norm.sf(abs(z))
