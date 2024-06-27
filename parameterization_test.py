import numpy as np
import networkx as nx
import pandas as pd
from compute_J import compute_Jacobian
from strategy_optimization import optimize_strategy
from pathlib import Path


''' 
Summary of changes from base parameterization:
 - reduce size/number of large growers (reduce amount of water they use so the split is more even among users) x
 - increased interaction of small growers and communities with local governance, remove outsized influence of large growers x
 - reduced groundwater and surface water extraction x
 - stronger enforcement of gw quality regulations x
 - increased non-ag extraction x
 - increased interaction b/w small growers and rural communities
'''

def set_scale_params(N,M,K,N_list,M_list,K_list,tot,R):
  '''
  Outputs:
    All of the scale parameters and strategy parameters
  '''
  phis = np.zeros(2) 
  phis[0] = np.random.uniform(0.62,0.68) # sw
  phis[1] = np.random.uniform(0.07,0.11,(1,)) #gw
  
  psis = np.zeros(2)
  psis[0] = np.random.uniform(0.78,0.84,(1,)) #sw
  psis[1] = np.random.uniform(0.77,0.83,(1,)) #gw
  psi_bars = np.zeros(2)
  psi_bars[0] = 1-psis[0]# proportion of surface water transferred to groundwater
  psi_bars[1] = 1-psis[1]
  #eq_R_ratio = np.random.uniform(0.005,0.007,(1,))
  eq_R_ratio = (psi_bars[1]/psi_bars[0])*(phis[1]/phis[0])

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
  small_growers = np.array([False]*(N))
  small_growers[[1,3]] = True

  psi_tildes = np.zeros((3,N)) # 
  #psi_tildes[0,0:3] = np.array([1/3,1/3,1/3])
  psi_tildes[0,1:3] = np.array([0.5,0.5]) # sw split
  weights = np.array([10,15,15,15,15,15,15]) # gw split
  psi_tildes[1] = weights/(np.sum(weights)) 
  weights = np.array([2,2,2,2,1,1])
  psi_tildes[2,1:N] = weights/(np.sum(weights)) # gw discharge split
  
  de2_de1 = -0.9*((phis[0]*psis[0]*psi_tildes[0,:])/(phis[1]*psi_tildes[1,:]))*eq_R_ratio
  de2_de1 = np.nan_to_num(de2_de1)
  
  alphas = np.zeros((1,tot))
  alphas[0,1:N] = np.random.uniform(0.05,0.1,(6,))
  alphas[0,0] = np.random.uniform(0.3,0.6,(1,))
  # alphas[0,3] = np.random.uniform(0.3,0.6)
  # alphas[0,[2,4,5,6]] = np.random.uniform(0.05,0.1,(4,))
  alphas[0,N:] = np.random.uniform(0.05,0.1,(K+M,))
  beta_tildes = np.zeros([1,tot]) # gain based on resource access in general
  beta_hats = np.zeros([1,tot]) # gain from govt support
  betas = np.zeros([1,tot]) # gain from collaboration/support from others
  beta_bars = np.zeros([1,tot]) # gain from "natural" gain

  # rural communities gain from growers, small growers gain from rural communities
  path = Path.cwd().joinpath('parameter_files', 'test', 'sigmas.xlsx')
  sigmas_df = pd.read_excel(path)
  sigma_weights = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
  sigma_weights = np.array(sigma_weights, dtype=[('O', float)]).astype(float)
  total = np.sum(sigma_weights[:,:N],axis = 0)
  from_ngo = np.sum(sigma_weights[:N+K,:N],axis = 0)
  from_gov = np.sum(sigma_weights[-M:,:N],axis = 0)
  # resource users have gain from extraction, collaboration, and recruitment/self-growth, respectively
  betas_1 = np.random.dirichlet([30,40,30],1).transpose()
  betas_2 = np.random.dirichlet([30,20,50],1).transpose()
  betas_3 = np.random.dirichlet([30,20,50],1).transpose()
  betas_4 = np.random.dirichlet([30,20,50],1).transpose()
  betas_5 = np.random.dirichlet([30,20,50],1).transpose()
  betas_6 = np.random.dirichlet([30,20,50],1).transpose()
  betas_7 = np.random.dirichlet([30,20,50],1).transpose()
  
  beta_params = np.stack([betas_1, betas_2, betas_3, betas_4, betas_5, betas_6, betas_7])
  beta_tildes[0,:N] = beta_params[:,0,0]
  fraction = np.where(total==0,0,from_ngo/total)
  betas[0,:N] = beta_params[:,1,0]*fraction
  beta_hats[0,:N] = beta_params[:,1,0]*fraction
  beta_bars[0,:N] = beta_params[:,2,0]

  sigma_tildes = np.zeros([3,N]) # gain based on each resource state variable
  sigma_tildes[1,0] = np.random.uniform(0.4,0.6) # salience of gw availability to communities
  sigma_tildes[2,0] = 1 - sigma_tildes[1,0] # salience of gw quality to communities
  sigma_tildes[0,1:] = -de2_de1[1:]/(1-de2_de1[1:]) #np.random.uniform(0.1,0.5,(2,)) # reliance of growers w/ sw access on sw
  sigma_tildes[1,1:] = 1-sigma_tildes[0,1:] # reliance of growers on gw 

  sigmas = np.zeros((N+K,tot)) # sigma_k,n is kxn $
  sigma_hats = np.zeros((M,tot))

  for i in range(tot-1): # loop through to fill in each column
    sigmas[:,i][sigma_weights[:N+K,:][:,i]>0] = np.random.dirichlet((sigma_weights[:N+K,:][:,i][sigma_weights[:N+K,:][:,i]>0])*10)
    sigma_hats[:,i][sigma_weights[-M:,:][:,i]>0] = np.random.dirichlet((sigma_weights[-M:,:][:,i][sigma_weights[-M:,:][:,i]>0])*10)
    
  # non-govt and govt orgs actors have natural gain and gain from collaboration from other actors (betas) and govt (beta_hats)
  total = np.sum(sigma_weights[:,N:N+K+M],axis = 0)
  from_ngo = np.sum(sigma_weights[:N+K,N:],axis = 0)
  from_gov = np.sum(sigma_weights[-M:,N:],axis = 0)
  beta_bars[0,N:] = np.random.uniform(np.ones(K+M)*0.2, np.ones(K+M)*0.3)
  fraction = np.where(total==0,0,from_ngo/total)
  betas[0,N:] = np.random.uniform(fraction-0.1*fraction,fraction+0.1*(fraction)) * (1-beta_bars[0,N:])
  beta_hats[0,N:] = 1 - beta_bars[0,N:] - betas[0,N:]
  
  # reduced effect of regulations on grower capacity to reflect greater support for growers
  lambdas = np.zeros((N+K,tot))  # lambda_k,n is kxn $
  lambda_hats = np.zeros((M,tot))
  path = Path.cwd().joinpath('parameter_files', 'test', 'lambdas.xlsx')
  lambdas_df = pd.read_excel(path)
  lambdas_weights = lambdas_df.fillna(0).values[:,1:] # array of weights for sampling
  lambdas_weights = np.array(lambdas_weights, dtype=[('O', float)]).astype(float)
  for i in range(tot-1): # loop through to fill in each (each column sums to 1)
   lambdas[:,i][lambdas_weights[:N+K,:][:,i]>0] = np.random.dirichlet((lambdas_weights[:N+K,:][:,i][lambdas_weights[:N+K,:][:,i]>0])*10)
   lambda_hats[:,i][lambdas_weights[-M:,:][:,i]>0] = np.random.dirichlet((lambdas_weights[-M:,:][:,i][lambdas_weights[-M:,:][:,i]>0])*10)
 
  # losses
  etas = np.zeros((1,tot))
  eta_hats = np.zeros((1,tot))
  total = np.sum(lambdas_weights,axis = 0)
  from_ngo = np.sum(lambdas_weights[:N+K],axis = 0)
  from_gov = np.sum(lambdas_weights[-M:],axis = 0)
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
  G[DACs_idx,[1,2],np.nonzero(M_list=='Local Water Boards')[0],DACs_idx] = np.random.uniform(1,2)
  G[DACs_idx,[1,2],np.nonzero(M_list=='County Board of Supervisors')[0],DACs_idx] = np.random.uniform(0.5,1)
  #G[DACs_idx,[1,2],np.nonzero(M_list=='Drinking Water Division (SWRCB)')[0],DACs_idx] = np.random.uniform(1,2)
  #G[DACs_idx,[1,2],np.nonzero(M_list=='Groundwater Management (SWRCB)')[0],DACs_idx] = np.random.uniform(1,2)
  #G[DACs_idx,[2],np.nonzero(M_list=='Division of Water Quality (SWRCB)')[0],DACs_idx] = np.random.uniform(1,2)
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
  
  return phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, de2_de1, G, E, T, H, C, P

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
  
  sw_users = np.array([False]*(N+K))
  sw_users[1:3] = True
  ds_dr = np.zeros((2))
  ds_dr[0] = -1
  ds_dr[1] = 0 
  de_dr = np.zeros((3,N+K))
  de_dr[0,sw_users] = 1
  de_dr[0,sw_users] = 1  
  de_dr[1,0] = np.random.uniform(0,0.5) # DACs
  de_dr[1,1] = np.random.uniform(0,0.5) # small growers
  #de_dr[1,1] = np.random.uniform(0.25,0.75) # small growers
  de_dr[1,2] = np.random.uniform(0,0.5) # investor growers
  de_dr[1,3] = np.random.uniform(0,0.5)  # white area small growers
  #de_dr[1,3] = np.random.uniform(0.5,1)  # white area small growers
  de_dr[1,4] = np.random.uniform(0,0.5) # white area investor growers
  de_dr[1,5:] = np.random.uniform(0,1)
  de_dr[2,0] = np.random.uniform(0,0.5)*-1
  dt_dr = 0.5 
  de_dg = np.zeros((3,M,N))  ###### $
  de_dE = np.zeros((3,N+K,N))
  
  # de/dg for surface water
  path = Path.cwd().joinpath('parameter_files', 'test', 'de_dg_sw_lower.xlsx')
  df = pd.read_excel(path) #lower bounds for de_dg for sw
  sw_lower = df.fillna(0).values[:,1:]
  sw_lower = np.array(sw_lower, dtype=[('O', float)]).astype(float)
  path = Path.cwd().joinpath('parameter_files', 'test', 'de_dg_sw_upper.xlsx')
  df = pd.read_excel(path)
  sw_upper = df.fillna(0).values[:,1:]
  sw_upper = np.array(sw_upper, dtype=[('O', float)]).astype(float) 
  de_dg[0,:,:] = np.random.uniform(sw_lower[-M:], sw_upper[-M:])
  de_dE[0,N:,:] = np.random.uniform(sw_lower[:K], sw_upper[:K])
  # de/dg for groundwater
  path = Path.cwd().joinpath('parameter_files', 'test', 'de_dg_gw_lower.xlsx')
  df = pd.read_excel(path) #lower bounds for de_dg for sw
  gw_lower = df.fillna(0).values[:,1:]
  gw_lower = np.array(gw_lower, dtype=[('O', float)]).astype(float)
  path = Path.cwd().joinpath('parameter_files', 'test', 'de_dg_gw_upper.xlsx')
  gw_upper = df.fillna(0).values[:,1:]
  gw_upper = np.array(gw_upper, dtype=[('O', float)]).astype(float) 
  de_dg[1,:,:] = np.random.uniform(gw_lower[-M:], gw_upper[-M:])
  de_dE[1,N:,:] = np.random.uniform(gw_lower[:K], gw_upper[:K]) 
  # de/dg for groundwater quality
  path = Path.cwd().joinpath('parameter_files', 'test', 'de_dg_gwq_lower.xlsx')
  df = pd.read_excel(path) #lower bounds for de_dg for sw
  gwq_lower = df.fillna(0).values[:,1:]
  gwq_lower = np.array(gwq_lower, dtype=[('O', float)]).astype(float)
  path = Path.cwd().joinpath('parameter_files', 'test', 'de_dg_gwq_upper.xlsx')
  df = pd.read_excel(path)
  gwq_upper = df.fillna(0).values[:,1:]
  gwq_upper = np.array(gwq_upper, dtype=[('O', float)]).astype(float) 
  de_dg[2,:,:] = np.random.uniform(gwq_lower[-M:], gwq_upper[-M:])
  de_dE[2,N:,:] = np.random.uniform(gwq_lower[:K], gwq_upper[:K])
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
  
  #dg_dG[big_growers_idx,np.nonzero(M_list=='Drinking Water Division (SWRCB)'),:] = 0
  #dg_dG[big_growers_idx,np.nonzero(M_list=='Local Water Boards'),:] = 0
  #dg_dG[big_growers_idx,np.nonzero(M_list=='County Board of Supervisors'),:] = 0
  dg_dG[:,np.nonzero(M_list=='Friant-Kern Canal'),:] = 0 # cannot affect how Friant-kern canal delivers water to individuals

  dg_dG = np.broadcast_to(dg_dG, (3,N+K,M,N))
  
  dh_dH = dg_dG[0,:,:,0]/2 #np.zeros((N+K,M))
  dh_dH[np.nonzero(N_list=='small growers (white area)')] = np.random.uniform(0,0.1,(1,M))
  dh_dH[np.nonzero(N_list=='investor growers (white area)')] = np.random.uniform(0,0.1,(1,M))
  
  
  dg_dy = np.random.uniform(0.5,1,(3,M,N)) # 
  dh_dy = np.random.uniform(0.5,1,(M))
  path = Path.cwd().joinpath('parameter_files', 'test', 'dt_dh.xlsx')
  data = pd.read_excel(path, sheet_name=None) #lower bounds for de_dg for sw
  lower = data['lower'].fillna(0).values[:,1:]
  lower = np.array(lower, dtype=[('O', float)]).astype(float)
  upper  = data['upper'].fillna(0).values[:,1:]
  upper = np.array(upper, dtype=[('O', float)]).astype(float)
  dts = np.random.uniform(lower,upper) 
  dt_dh = dts[-M:]  
  dt_dT = np.zeros(N+K)
  dt_dT[N:] = dts[:K][:,0]
  
  db_de = np.zeros((3,N+K)) # changed this to cap at 1
  db_de[0,sw_users] = np.random.uniform(0.5,1)
  db_de[1, ~sw_users] = np.random.uniform(0.5,1)
  db_de[1,sw_users] = np.random.uniform(0.5,1)
  db_de[2, 0] = np.random.uniform(0.5,1)

  dc_dC = np.random.uniform(1,1.5,(N+K,tot)) #dc_dw_p_i,n is ixn $
  dc_dC[:N,:] = np.random.uniform(0.5,1,(N,tot))
  indices = np.arange(0,N+K)
  dc_dC[indices,indices] = 0
  dc_dC[:N+K,N+K+np.nonzero(M_list=='Friant-Kern Canal')[0][0]] = 0
  
  dp_dP = np.random.uniform(0.5,1,(N+K,M,tot))
  dp_dy = np.random.uniform(0.5,1,(M,tot))
  du_dx_plus = np.random.uniform(0,1,(tot))
  du_dx_minus = np.random.uniform(1,2,(tot))

  return ds_dr, de_dr, dt_dr, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus
