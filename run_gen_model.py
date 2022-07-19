import numpy as np
import networkx as nx
import pandas as pd
from parameterization_base import set_scale_params, set_fixed_exp_params
from compute_J import compute_Jacobian
from strategy_optimization import optimize_strategy


def run_system(user = None):
  '''
  Takes in system meta-parameters and produces a system parameterization and computes
  stability of that system
  
  Inputs:
    N1, N2, and N3: number of (exclusively) extractors, combined extractors and 
      accessors, and (exclusively) accessors, respectively
    K: number of non-resource user actors
    M: number of decision centers
    tot: total number of state variables (this is N + K + M + 1)
    C: density of decision center interventions
  Outputs:
    stability: a boolean, True indicates stable, False indicates unstable
    J: the Jacobian
    converged: a boolean, True indicates the optimization reached convergence, False
      indicates the optimization ran into the maximum number of iterations
    strategy_history: a list of all the strategies of all of the actors throughout the 
      optimization, used to debug the optimization
    grad: the gradient of the objective function for the chosen strategy. used to debug
      the optimization
    total_connectance: the proportion of all possible interactions that exist in the 
      final system
    The remaining outputs are all of the sampled or computed scale, exponent, and strategy parameters.
  '''
  print('a')
  entities = pd.read_excel('parameter_files\\base\entity_list.xlsx',sheet_name=None, header=None)
  N_list=entities['N'].values[:,0]
  N = len(N_list)

  K_list=entities['K'].values[:,0]
  K = len(K_list)

  M_list=entities['M'].values[:,0]
  M = len(M_list)

  tot = N+K+M
  
  R = 3

  phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P = set_scale_params(N,M,K,N_list,M_list,K_list,tot,R)
  
  ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus = set_fixed_exp_params(N,M,K,N_list,M_list,K_list,tot,R)
  
  # check stability before strategy optimization
  J = compute_Jacobian(N,K,M,tot,
      phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P,
      ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus)
        
  eigvals = np.linalg.eigvals(J)
  if np.all(eigvals.real < 0):  # stable if real part of eigenvalues is negative
    stability_1 = True
  else:
    stability_1 = False  # unstable if real part is positive, inconclusive if 0
  
  if user != None:
    max_iters = 1000 # change back to 100!! 
    strategy, stability_2, stability_3, converged, strategy_history, grad_history = optimize_strategy(max_iters, user, N, K, M, tot, R,
      phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus)
  else:
    strategy = None
    
  J = compute_Jacobian(N,K,M,tot,
      phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P,
      ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus)

  # check for positive entries
  if np.any(np.diagonal(J) > 0):
    print('Warning: positive diagonal elements of Jacobian')
        
  # --------------------------------------------------------------------------
  # Compute the eigenvalues of the Jacobian and check stability
  # --------------------------------------------------------------------------
  eigvals = np.linalg.eigvals(J)
  if np.all(eigvals.real < 0):  # stable if real part of eigenvalues is negative
    stability = True
  else:
    stability = False  # unstable if real part is positive, inconclusive if 0

  stability_final = stability 
  
  
  return (stability_final, stability_1, stability_2, stability_3, converged, strategy_history, grad_history, strategy, J, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus)


# if __name__ == "__main__":
  # (N1,N2,N3,K,M,tot,C,stability, J, converged, strategy_history, grad, total_connectance,
      # phi,psis,alphas,beta_tildes,sigma_tildes,betas,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dG,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dC,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      # G,H,W,K_p) = main()