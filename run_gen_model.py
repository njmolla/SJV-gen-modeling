import numpy as np
import networkx as nx
import pandas as pd
from compute_J import compute_Jacobian
from strategy_optimization import optimize_strategy




def set_scale_params(N,M,K,N_list,M_list,K_list,tot,R):
  '''
  Outputs:
    All of the scale parameters and strategy parameters
  '''
  phis = np.zeros(2) 
  phis[0] = 1 # sw
  phis[1] = np.random.uniform(0.001,0.004,(1,)) #gw
  psis = np.zeros(2)
  psis[0] = np.random.uniform(0.9,1,(1,)) #sw
  psis[1] = np.random.uniform(0.01,0.02,(1,)) #gw
  eq_R_ratio = np.random.uniform(0.1,0.5,(1,))
  psi_bars = np.zeros(2) 
  psi_bars[0] = 1-psis[0]# proportion of surface water transferred to groundwater
  psi_bars[1] = psi_bars[0]*(phis[0]/phis[1])*eq_R_ratio
  
  # 1: DACs
  # 2: small district farmers
  # 3: investor district farmers
  # 4: small white area growers
  # 5: investor white area growers
  # 6: municipalities
  # 7: other dischargers
  
  # set up logical arrays to make indexing cleaner
  
  sw_users = np.array([False]*(N+K))
  sw_users[1:3] = True
  small_growers = np.array([False]*(N+K))
  small_growers[[1,3]] = True
  
  psi_tildes = np.zeros((3,N)) # 
  psi_tildes[0,1:3] = np.random.dirichlet([0.3,0.7],1) # sw split
  psi_tildes[1] = np.random.dirichlet([0.000001,0.05,0.25,0.1,0.4,0.05,0.05],1) # gw split
  psi_tildes[2,1:N] = [0.07,0.4,0.07,0.4,0.03,0.03] # gw discharge split
  alphas = np.zeros((1,tot))
  alphas[0,0:2] = np.random.uniform(0.3,0.6,(2,))
  alphas[0,3] = np.random.uniform(0.3,0.6)
  alphas[0,[2,4,5,6]] = np.random.uniform(0.05,0.1,(4,))
  alphas[0,N:] = np.random.uniform(0.05,0.1,(K+M,))
  beta_tildes = np.zeros([1,tot]) # gain based on resource access in general
  beta_hats = np.zeros([1,tot]) # gain from govt support
  betas = np.zeros([1,tot]) # gain from collaboration/support from others
  beta_bars = np.zeros([1,tot]) # gain from "natural" gain

  sigmas_df = pd.read_csv('parameter_files\sigmas.csv')
  sigma_weights = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
  sigma_weights = np.array(sigma_weights, dtype=[('O', float)]).astype(float)
  total = np.sum(sigma_weights[:,:N],axis = 0)
  from_ngo = np.sum(sigma_weights[:N+K,:N],axis = 0)
  from_gov = np.sum(sigma_weights[N+K:,:N],axis = 0)
  # resource users have gain from extraction, collaboration, and recruitment/self-growth, respectively
  betas_1 = np.random.dirichlet([0.3,0.4,0.3],1).transpose()
  betas_2 = np.random.dirichlet([0.2,0.7,0.1],1).transpose()
  betas_3 = np.random.dirichlet([0.2,0.2,0.6],1).transpose()
  betas_4 = np.random.dirichlet([0.3,0.4,0.3],1).transpose()
  betas_5 = np.random.dirichlet([0.3,0.2,0.5],1).transpose()
  betas_6 = np.random.dirichlet([0.3,0.2,0.5],1).transpose()
  betas_7 = np.random.dirichlet([0.3,0.2,0.5],1).transpose()
  
  beta_params = np.stack([betas_1, betas_2, betas_3, betas_4, betas_5, betas_6, betas_7])
  beta_tildes[0,:N] = beta_params[:,0,0]
  fraction = np.where(total==0,0,from_ngo/total)
  betas[0,:N] = beta_params[:,1,0]*fraction
  beta_hats[0,:N] = beta_params[:,1,0]*fraction
  beta_bars[0,:N] = beta_params[:,2,0]

  sigma_tildes = np.zeros([3,N+K]) # gain based on each resource state variable
  sigma_tildes[1,~sw_users] = 1 # white area growers rely entirely on groundwater
  sigma_tildes[1,0] = np.random.uniform(0.4,0.6) # salience of gw availability to communities
  sigma_tildes[2,0] = 1 - sigma_tildes[1,0] # salience of gw quality to communities
  sigma_tildes[0,sw_users] = np.random.uniform(0.1,0.5,(2,)) # reliance of growers w/ sw access on sw
  sigma_tildes[1,sw_users] = 1-sigma_tildes[0,sw_users] # reliance of growers w/ sw access on gw 

  sigmas = np.zeros((N+K,tot)) # sigma_k,n is kxn $
  sigma_hats = np.zeros((M,tot))

  for i in range(tot-1): # loop through to fill in each column
    sigmas[:,i][sigma_weights[:N+K,:][:,i]>0] = np.random.dirichlet(sigma_weights[:N+K,:][:,i][sigma_weights[:N+K,:][:,i]>0])
    sigma_hats[:,i][sigma_weights[-M:,:][:,i]>0] = np.random.dirichlet(sigma_weights[N+K:,:][:,i][sigma_weights[N+K:,:][:,i]>0])
    
  # non-govt and govt orgs actors have natural gain and gain from collaboration from other actors (betas) and govt (beta_hats)
  total = np.sum(sigma_weights[:,N:N+K+M],axis = 0)
  from_ngo = np.sum(sigma_weights[:N+K,N:],axis = 0)
  from_gov = np.sum(sigma_weights[N+K:,N:],axis = 0)
  beta_bars[0,N:] = np.random.uniform(np.ones(K+M)*0.2, np.ones(K+M)*0.3)
  fraction = np.where(total==0,0,from_ngo/total)
  betas[0,N:] = np.random.uniform(fraction-0.1*fraction,fraction+0.1*(fraction)) * (1-beta_bars[0,N:])
  beta_hats[0,N:] = 1 - beta_bars[0,N:] - betas[0,N:]
  
  lambdas = np.zeros((N+K,tot))  # lambda_k,n is kxn $
  lambda_hats = np.zeros((M,tot))
  lambdas_df = pd.read_excel('parameter_files\lambdas.xlsx')
  lambdas_weights = lambdas_df.fillna(0).values[:,1:] # array of weights for sampling
  lambdas_weights = np.array(lambdas_weights, dtype=[('O', float)]).astype(float)
  for i in range(tot): # loop through to fill in each (each column sums to 1)
    lambdas[:,i][lambdas_weights[:N+K,:][:,i]>0] = np.random.dirichlet(lambdas_weights[:N+K,:][:,i][lambdas_weights[:N+K,:][:,i]>0])
    lambda_hats[:,i][lambdas_weights[N+K:,:][:,i]>0] = np.random.dirichlet(lambdas_weights[N+K:,:][:,i][lambdas_weights[N+K:,:][:,i]>0])
 
  # losses
  etas = np.zeros((1,tot))
  eta_hats = np.zeros((1,tot))
  total = np.sum(lambdas_weights,axis = 0)
  from_ngo = np.sum(lambdas_weights[:N+K],axis = 0)
  from_gov = np.sum(lambdas_weights[N+K:],axis = 0)
  eta_bars = np.random.uniform(beta_bars[0],0.9,(tot))
  fraction = np.where(total==0,0,from_ngo/total)
  etas[0] = np.random.uniform(fraction-0.1*(fraction),fraction+0.1*(fraction))*(1-eta_bars)
  eta_hats[0] = 1 - eta_bars[0] - etas[0]
  
  # effort allocation parameters 
  G = np.zeros((N+K,R,M,N))  # F_i,m,n is ixmxn positive effort for influencing resource extraction governance $
    # get indices
  EJ_groups = np.nonzero(K_list=='EJ groups')
  DACs_idx = np.nonzero(N_list == 'rural communities')
  growers = np.nonzero(np.any([N_list == 'small growers', N_list =='investor growers', N_list == 'small growers (white area)', N_list =='investor growers (white area)'],axis=0))[0]
  # EJ groups help DACs receive funding for water supply and water
  # treatment infrastructure from the state
  G[N+EJ_groups[0],[1,2],np.nonzero(M_list=='Financial Assistance (SWRCB)')[0],DACs_idx] = np.random.uniform(1,2,(1,2))
  G[N+EJ_groups[0],[1,2],np.nonzero(M_list=='Local Water Boards')[0],DACs_idx] = np.random.uniform(1,2,(1,2))
  #G[[1,2]][:,N+EJ_groups[0],np.nonzero(M_list=='Local Water Boards')[0],DACs_idx] = np.random.uniform(0.5,1, (2,1,1))
  # UCCE helps growers get grants from NRCS grants
  G[N+np.nonzero(K_list=='UC Extension/research community')[0],2,np.nonzero(M_list=='NRCS')[0],growers] = np.random.uniform(0.5,1.5, (1,1,1,4))
  G = np.divide(G,np.sum(G,axis=0))
  G = np.nan_to_num(G)
  
  E = np.zeros((N+K,3,N))
  E[N+EJ_groups[0],[1,2],DACs_idx] = np.random.uniform(0.5,1, (1,2))
  E = np.divide(E,np.sum(E,axis=0))
  E = np.nan_to_num(E)
  T = np.zeros(N+K)
  T[N+np.nonzero(K_list == 'Sustainable conservation')[0]] = np.random.uniform(0.5,1)
  T = T/np.sum(T)
  H = np.zeros((N+K,M))  # effort for influencing recharge policy 
  H[N + np.nonzero(K_list=='Flood-MAR network')[0], np.nonzero(M_list=='Water Rights Division (SWRCB)')[0]] = np.random.uniform(0.3,0.5)
  H = np.divide(H,np.sum(H,axis=0))
  H = np.nan_to_num(H)
  C = sigma_weights[:N+K]  # effort for collaboration. C_i,n is ixn 
  C[lambdas_weights[:N+K]>0] = -1*lambdas_weights[:N+K][lambdas_weights[:N+K]>0]
  C = np.divide(C,np.sum(np.abs(C),axis=0))
  C = np.nan_to_num(C)
  
  
  P = np.zeros((N+K,M,tot))
  P[EJ_groups,np.nonzero(M_list=='Local Water Boards')[0],DACs_idx] = np.random.uniform(0.5,1)
  P = np.divide(P,np.sum(P,axis=0))
  P = np.nan_to_num(P)
  
  return phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P

def set_fixed_exp_params(N, M, K,N_list,M_list,K_list,tot,R):
  '''
  Takes in system meta-parameters and samples the exponent parameters
  (used for the correlation experiments)
  
  Inputs:
  Inputs:
    N1: number of resource users
    N2: number of non-govt orgs
    M: number of decision centers
    tot: total number of state variables (this is N + K + M + 1)
    C: density of decision center interventions
  Outputs:
    All of the exponent parameters 
  '''
  # ------------------------------------------------------------------------
  # Initialize exponent parameters
  # ------------------------------------------------------------------------
  
  # 1: DACs
  # 2: small district farmers
  # 3: investor district farmers
  # 4: small white area growers
  # 5: investor white area growers 
  
  # TO DO: fix parameterization for water quality!!!
  sw_users = np.array([False]*(N+K))
  sw_users[1:3] = True
  ds_dr = np.zeros((2))
  de_dr = np.zeros((3,N+K))
  de_dr[0,sw_users] = 1
  de_dr[1,0] = np.random.uniform(1,2)
  de_dr[1,1] = np.random.uniform(0.5,1.5)
  de_dr[1,2] = np.random.uniform(0,0.5)
  de_dr[1,3] = np.random.uniform(1,2)
  de_dr[1,4] = np.random.uniform(0,0.5)
  de_dr[2,0] = np.random.uniform(1,2)*-1
  dt_dr = 0.5 
  de2_de1 = -1
  de_dg = np.zeros((3,M,N))  ###### $
  de_dE = np.zeros((3,N+K,N))
  
  # de/dg for surface water
  df = pd.read_excel('parameter_files\de_dg_sw_lower.xlsx') #lower bounds for de_dg for sw
  sw_lower = df.fillna(0).values[1:,1:]
  sw_lower = np.array(sw_lower, dtype=[('O', float)]).astype(float)
  df = pd.read_excel('parameter_files\de_dg_sw_upper.xlsx')
  sw_upper = df.fillna(0).values[1:,1:]
  sw_upper = np.array(sw_upper, dtype=[('O', float)]).astype(float) 
  de_dg[0,:,:] = np.random.uniform(sw_lower[-M:], sw_upper[-M:])
  de_dE[0,N:,:] = np.random.uniform(sw_lower[N:N+K], sw_upper[N:N+K])
  # de/dg for groundwater
  df = pd.read_excel('parameter_files\de_dg_gw_lower.xlsx') #lower bounds for de_dg for sw
  gw_lower = df.fillna(0).values[1:,1:]
  gw_lower = np.array(gw_lower, dtype=[('O', float)]).astype(float)
  df = pd.read_excel('parameter_files\de_dg_gw_upper.xlsx')
  gw_upper = df.fillna(0).values[1:,1:]
  gw_upper = np.array(gw_upper, dtype=[('O', float)]).astype(float) 
  de_dg[1,:,:] = np.random.uniform(gw_lower[-M:], gw_upper[-M:])
  de_dE[1,N:,:] = np.random.uniform(gw_lower[N:N+K], gw_upper[N:N+K]) 
  # de/dg for groundwater quality
  df = pd.read_excel('parameter_files\de_dg_gwq_lower.xlsx') #lower bounds for de_dg for sw
  gwq_lower = df.fillna(0).values[1:,1:]
  gwq_lower = np.array(gwq_lower, dtype=[('O', float)]).astype(float)
  df = pd.read_excel('parameter_files\de_dg_gwq_upper.xlsx')
  gwq_upper = df.fillna(0).values[1:,1:]
  gwq_upper = np.array(gwq_upper, dtype=[('O', float)]).astype(float) 
  de_dg[2,:,:] = np.random.uniform(gwq_lower[-M:], gwq_upper[-M:])
  de_dE[2,N:,:] = np.random.uniform(gwq_lower[N:N+K], gwq_upper[N:N+K])
  # make sure RUs cannot use E as a strategy
  de_dE[:,:N,:] = np.zeros((3,N,N))
  
  # dg/dG doesn't depend on the resource, so treated as NxMxN and then broadcasted
  dg_dG = np.random.uniform(0.5,1,(N+K,M,N))  # dg_m,n/(dF_i,m,n * x_i) is ixmxn $
  # get indices for some exceptions
  big_growers_idx = np.nonzero(N_list=='investor growers')
  IDs_idx = np.nonzero(M_list=='Irrigation/water districts')
  growers = np.nonzero(np.any([N_list == 'small growers', N_list =='investor growers', N_list == 'small growers (white area)', N_list =='investor growers (white area)'],axis=0))[0]
  grower_groups = np.nonzero(np.any([K_list == 'Grower advocacy groups', K_list == 'UC Extension/research community', K_list == 'Sustainable conservation', K_list == 'MPEP', K_list == 'PCAs/CCAs'],axis=0))[0]
  EJ_groups = np.nonzero(K_list=='EJ groups')
  DACs_idx = np.nonzero(N_list == 'rural communities')
  
  dg_dG[big_growers_idx,IDs_idx,:] = np.random.uniform(1,2,(N)) # big growers have outsized influence on IDs/WDs
  dg_dG[DACs_idx,IDs_idx,:] = np.random.uniform(0,0.1,(N)) # DACs have essentially no representation on IDs/WD boards
  dg_dG[big_growers_idx,np.nonzero(M_list=='Drinking Water Division (SWRCB)'),:] = 0
  dg_dG[big_growers_idx,np.nonzero(M_list=='Local Water Boards'),:] = 0
  dg_dG[big_growers_idx,np.nonzero(M_list=='County Board of Supervisors'),:] = 0
  dg_dG[:,np.nonzero(M_list=='Friant-Kern Canal'[0][0]),:] = 0 # cannot affect how Friant-kern canal delivers water to individuals
  # dg_dG[np.meshgrid(growers,grower_groups)] = np.random.uniform(1,2,(len(grower_groups),len(growers),N))
  # dg_dG[DACs_idx, EJ_groups] = np.random.uniform(1,2,(1,1,N)
  # dg_dG[growers, EJ_groups] = np.random.uniform(0,0.2,(len(growers),1,N))
  dg_dG = np.broadcast_to(dg_dG, (3,N+K,M,N))
  
  dh_dH = dg_dG[0,:,:,0]/2 #np.zeros((N+K,M))
  dh_dH[np.nonzero(N_list=='small growers (white area)')] = np.random.uniform(0,0.05,(1,M))
  dh_dH[np.nonzero(N_list=='investor growers (white area)')] = np.random.uniform(0,0.1,(1,M))
  
  
  dg_dy = np.random.uniform(0.5,1,(3,M,N)) # 
  dh_dy = np.random.uniform(0.5,1,(M))
  
  data = pd.read_excel('parameter_files\dt_dh.xlsx', sheet_name=None) #lower bounds for de_dg for sw
  lower = data['lower'].fillna(0).values[:,1:]
  lower = np.array(lower, dtype=[('O', float)]).astype(float)
  upper  = data['lower'].fillna(0).values[:,1:]
  upper = np.array(upper, dtype=[('O', float)]).astype(float)
  dts = np.random.uniform(lower,upper) 
  dt_dh = dts[-M:]  
  dt_dT = np.zeros(N+K)
  dt_dT[N:] = dts[:K][:,0]
  
  db_de = np.zeros((3,N+K))
  db_de[0,sw_users] = np.random.uniform(0.5,1)
  db_de[1, ~sw_users] = np.random.uniform(1,2)
  db_de[1,sw_users] = np.random.uniform(0.5,1)
  db_de[2, 0] = np.random.uniform(1,2)

  dc_dC = np.random.uniform(1,1.5,(N+K,tot)) #dc_dw_p_i,n is ixn $
  dc_dC[:N,:] = np.random.uniform(0.5,1,(N,tot))
  indices = np.arange(0,N+K)
  dc_dC[indices,indices] = 0
  dc_dC[:N+K,N+K+np.nonzero(M_list=='Friant-Kern Canal')[0][0]] = 0
  
  dp_dP = np.zeros((N+K,M,tot))
  # assume that ability to influence g corresponds to ability to influence p as well
  dp_dP[big_growers_idx,IDs_idx,:] = np.random.uniform(1,2,(tot))
  dp_dP[DACs_idx,IDs_idx,:] = np.random.uniform(0,0.5,(tot))
  dp_dy = np.random.uniform(0.5,1,(M,tot))
  du_dx_plus = np.random.uniform(0,1,(tot))
  du_dx_minus = np.random.uniform(1,2,(tot))

  return ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus

##########################################################################################

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
  print('b')
  entities = pd.read_excel('parameter_files\entity_list.xlsx',sheet_name=None, header=None)
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
    max_iters = 100 # change back to 100!! 
    strategy, stability_2, stability_3 = optimize_strategy(max_iters, user, N, K, M, tot, R,
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
  
  return (stability_final, stability_1, stability_2, stability_3, strategy, J, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus)


# if __name__ == "__main__":
  # (N1,N2,N3,K,M,tot,C,stability, J, converged, strategy_history, grad, total_connectance,
      # phi,psis,alphas,beta_tildes,sigma_tildes,betas,beta_bars,sigmas,etas,lambdas,eta_bars,mus,ds_dr,de_dr,de_dg,dg_dG,dg_dy,dp_dy,db_de,da_dr,dq_da,da_dp,dp_dH,dc_dC,dc_dw_n,dl_dx,du_dx,di_dK_p,di_dK_n,di_dy_p,di_dy_n,
      # G,H,W,K_p) = main()