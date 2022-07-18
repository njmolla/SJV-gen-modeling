import numpy as np
import pandas as pd

entities = pd.read_excel('parameter_files\entity_list.xlsx',sheet_name=None, header=None)
N_list=entities['N'].values[:,0]
N = len(N_list)

K_list=entities['K'].values[:,0]
K = len(K_list)

M_list=entities['M'].values[:,0]
M = len(M_list)

tot = N+K+M

R = 3

#drdot/dG

drdot_dG_test = np.zeros((R,N+K,R,M,N))
for a in range(N+K):
  for m in range(M):
    for n in range(N):
      drdot_dG_test[0,a,0,m,n] = -phis[0]*psis[0]*psi_tildes[0,n]*de_dg[0,m,n]*dg_dG[0,a,m,n]
      
for a in range(N+K):
  for m in range(M):
    for n in range(N):
      drdot_dG_test[1,a,1,m,n] = -phis[1]*psi_tildes[1,n]*de_dg[1,m,n]*dg_dG[1,a,m,n]
      
for a in range(N+K):
  for m in range(M):
    for n in range(N):
      drdot_dG_test[1,a,0,m,n] = -phis[1]*psi_tildes[1,n]*de2_de1*de_dg[0,m,n]*dg_dG[0,a,m,n]
      
for a in range(N+K):
  for m in range(M):
    for n in range(N):
      drdot_dG_test[2,a,2,m,n] = phis[1]*psi_tildes[2,n]*de_dg[2,m,n]*dg_dG[2,a,m,n]
            
dxdot_dG_test = np.zeros((tot,N+K,R,M,N))

for a in range(N+K):
  for m in range(M):
    for n in range(N):
      dxdot_dG_test[n,a,0,m,n] = alphas[0,n]*beta_tildes[0,n]*(sigma_tildes[0,n]*db_de[0,n] + sigma_tildes[1,n]*db_de[1,n]*de2_de1)*de_dg[0,m,n]*dg_dG[0,a,m,n]
      
for a in range(N+K):
  for m in range(M):
    for n in range(N):
      dxdot_dG_test[n,a,1,m,n] = alphas[0,n]*beta_tildes[0,n]*sigma_tildes[1,n]*db_de[1,n]*de_dg[1,m,n]*dg_dG[1,a,m,n]
      
for a in range(N+K):
  for m in range(M):
    for n in range(N):
      dxdot_dG_test[n,a,2,m,n] = alphas[0,n]*beta_tildes[0,n]*sigma_tildes[2,n]*db_de[2,n]*de_dg[2,m,n]*dg_dG[2,a,m,n]
      
# drdot/dE

drdot_dE_test = np.zeros((R,N+K,R,N))

for a in range(N+K):
  for n in range(N):
    drdot_dE_test[0,a,0,n] = -phis[0]*psis[0]*psi_tildes[0,n]*de_dE[0,a,n]

for a in range(N+K):
  for n in range(N):
    drdot_dE_test[1,a,1,n] = -phis[1]*psi_tildes[1,n]*de_dE[1,a,n]
    
for a in range(N+K):
  for n in range(N):
    drdot_dE_test[1,a,0,n] = -phis[1]*psi_tildes[1,n]*de2_de1*de_dE[0,a,n]
    
for a in range(N+K):
  for n in range(N):
    drdot_dE_test[2,a,2,n] = phis[1]*psi_tildes[2,n]*de_dE[2,a,n]
 
# dxdot/dE 
dxdot_dE_test = np.zeros((tot,N+K,R,N))

for a in range(N+K):
  for n in range(N):
    dxdot_dE_test[n,a,0,n] =  alphas[0,n]*beta_tildes[0,n]*(sigma_tildes[0,n]*db_de[0,n] + sigma_tildes[1,n]*db_de[1,n]*de2_de1)*de_dE[0,a,n]

for a in range(N+K):
  for n in range(N):
    dxdot_dE_test[n,a,1,n] =  alphas[0,n]*beta_tildes[0,n]*sigma_tildes[1,n]*db_de[1,n]*de_dE[1,a,n]
    
for a in range(N+K):
  for n in range(N):
    dxdot_dE_test[n,a,2,n] = alphas[0,n]*beta_tildes[0,n]*sigma_tildes[2,n]*db_de[2,n]*de_dE[2,a,n]
    
# drdot_dH
drdot_dH_test = np.zeros((R,N+K,M))
for a in range(N+K):
  for m in range(M):
      drdot_dH_test[0,a,m] = -phis[0]*psi_bars[0]*dt_dh[m]*dh_dH[a,m]
      
for a in range(N+K):
  for m in range(M):
      drdot_dH_test[1,a,m] = phis[1]*psi_bars[1]*dt_dh[m]*dh_dH[a,m]      
   
for a in range(N+K):
  for m in range(M):
      drdot_dH_test[2,a,m] = -phis[1]*psi_bars[1]*dt_dh[m]*dh_dH[a,m]
      
# drdot_dT
drdot_dT_test = np.zeros((R,N+K))
for a in range(N+K):
  drdot_dT_test[0,a] = -phis[0]*psi_bars[0]*dt_dT[a]
      
for a in range(N+K):
  drdot_dT_test[1,a] = phis[1]*psi_bars[1]*dt_dT[a]
   
for a in range(N+K):
  drdot_dT_test[2,a] = -phis[1]*psi_bars[1]*dt_dT[a]
  

dxdot_dC_plus_test = np.zeros((tot,N+K,tot))

for n in range(tot):
  for a in range(N+K):
      dxdot_dC_plus_test[n,a,n] = alphas[0,n]*betas[0,n]*sigmas[a,n]*dc_dC[a,n]
      
dxdot_dC_minus_test = np.zeros((tot,N+K,tot))

for n in range(tot):
  for a in range(N+K):
      dxdot_dC_minus_test[n,a,n] = alphas[0,n]*etas[0,n]*lambdas[a,n]*dc_dC[a,n]
      
dxdot_dP_plus_test = np.zeros((tot,N+K,M,tot))
      
for a in range(N+K):
  for m in range(M):
    for t in range(tot):
      dxdot_dP_plus_test[t,a,m,t] = alphas[0,t]*beta_hats[0,t]*sigma_hats[m,t]*dp_dP[a,m,t]      
      
dxdot_dP_minus_test = np.zeros((tot,N+K,M,tot))
      
for a in range(N+K):
  for m in range(M):
    for t in range(tot):
      dxdot_dP_minus_test[t,a,m,t] = alphas[0,t]*eta_hats[0,t]*lambda_hats[m,t]*dp_dP[a,m,t]      
      
#### Calculating how objectives change with each parameter

grad_G_test = np.zeros((R,R,M,N))

G_copy = G
G_copy[:,2,:,1:] = 0

E_copy = E
E_copy[:,2,1:] = 0  
#dR_dG is R,N+K,R,M,N. dX_dG is tot,N+K,R,M,N, dg_dG 3,N+K,M,N, G is N+K,R,M,N

p=0 # the actor whose strategy is being optimized
for m in range(M):
  for n in range(N):
    for r in range(R):
      grad_G_test[0,r,m,n] = de_dr[0,p]*dR_dG[0,p,r,m,n] + np.sum(de_dg[0,:,p]*dg_dy[0,:,p]*dX_dG[-M:,p,r,m,n] + np.sum(de_dg[0,:,p]*dg_dG[0,:,:,p]*dX_dG[:N+K,p,r,m,n][:,np.newaxis]*G_copy[:,0,:,p], axis=0)) + np.sum(de_dE[0,N:,p]*dX_dG[N:N+K,p,r,m,n]*E_copy[N:,0,p])
      
# for n=p
if p==0:
  for m in range(M):
    grad_G_test[0,0,m,p] += de_dg[0,m,p]*dg_dG[0,p,m,p]
      
for m in range(M):
  for n in range(N):
    for r in range(R):
      grad_G_test[1,r,m,n] = de_dr[1,p]*dR_dG[1,p,r,m,n] + de2_de1*grad_G_test[0,r,m,n] + np.sum(de_dg[1,:,p]*dg_dy[1,:,p]*dX_dG[-M:,p,r,m,n] + np.sum(de_dg[1,:,p]*dg_dG[1,:,:,p]*dX_dG[:N+K,p,r,m,n][:,np.newaxis]*G_copy[:,1,:,p],axis=0)) + np.sum(de_dE[1,N:,p]*dX_dG[N:N+K,p,r,m,n]*E_copy[N:,1,p])
      
      
          grad_G[i] = de_dr[i,0] * dR_dG[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,0]*dg_dy[i,:,0], (M,1,1,1)), dX_dG[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,0],dg_dG[i,:,:,0]*G_copy[:,i,:,0]), (N+K,M,1,1,1)),
                                               # 1m            km          km
                    np.reshape(dX_dG[:N+K,l], (N+K,1,R,M,N))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,:,0]*E_copy[:,i,0],(N+K,1,1,1)),dX_dG[:N+K,l]),axis=0)

# for n=p
if p==0:
  for m in range(M):
    grad_G_test[1,1,m,p] += de_dg[1,m,p]*dg_dG[1,p,m,p]
  
for m in range(M):
  for n in range(N):
    for r in range(R):
      grad_G_test[0,r,m,n] = de_dr[0,p]*dR_dG[0,p,r,m,n] + np.sum(de_dg[0,:,p]*dg_dy[0,:,p]*dX_dG[-M:,p,r,m,n] + np.sum(de_dg[0,:,p]*dg_dG[0,:,:,p]*dX_dG[:N+K,p,r,m,n][:,np.newaxis]*G_copy[:,0,:,p], axis=0)) + np.sum(de_dE[0,N:,p]*dX_dG[N:N+K,p,r,m,n]*E_copy[N:,0,p])
    
for m in range(M):
  for n in range(N):
    for r in range(R):  
      grad_G_test[2,r,m,n] = de_dr[2,p]*dR_dG[2,p,r,m,n] + np.sum(de_dg[2,:,p]*dg_dy[2,:,p]*dX_dG[-M:,p,r,m,n] + np.sum(de_dg[2,:,p]*dg_dG[2,:,:,p]*dX_dG[:N+K,p,r,m,n][:,np.newaxis]*G_copy[:,2,:,p],axis=0)) + np.sum(de_dE[2,N:,p]*dX_dG[N:N+K,p,r,m,n]*E_copy[N:,2,p])
         
if p==0: 
  for m in range(M):
    grad_G_test[2,2,m,p] += de_dg[2,m,p]*dg_dG[2,p,m,p]
    
###########################################
    
grad_E_test = np.zeros((R,R,N))

#dR_dE is R,N+K,R,N. dX_dG is tot,N+K,R,N, de_dE 3,N+K,N, E is N+K,R,N

for n in range(N):
  for r in range(R):
    grad_E_test[0,r,n] = de_dr[0,p]*dR_dE[0,p,r,n] + np.sum(de_dg[0,:,p]*dg_dy[0,:,p]*dX_dE[-M:,p,r,n] + np.sum(de_dg[0,:,p]*dg_dG[0,:,:,p]*dX_dE[:N+K,p,r,n][:,np.newaxis]*G_copy[:,0,:,p],axis=0)) + np.sum(de_dE[0,N:,p]*dX_dE[N:N+K,p,r,n]*E_copy[N:,0,p])

# for n=p
grad_E_test[0,0,p] += de_dE[0,p,p]
    
for n in range(N):
  for r in range(R):
    grad_E_test[1,r,n] = de_dr[1,p]*dR_dE[1,p,r,n] + np.sum(de_dg[0,:,p]*dg_dy[0,:,p]*dX_dE[-M:,p,r,n] + np.sum(de_dg[0,:,p]*dg_dG[0,:,:,p]*dX_dE[:N+K,p,r,n][:,np.newaxis]*G_copy[:,0,:,p],axis=0)) + np.sum(de_dE[0,N:,p]*dX_dE[N:N+K,p,r,n]*E_copy[N:,0,p]) + de2_de1*grad_E_test[0,r,n]

if p==0:
# for n=p
  grad_E_test[1,1,p] += de_dE[1,p,p]
    
for n in range(N):
  for r in range(R):
    grad_E_test[2,r,n] = de_dr[2,p]*dR_dE[2,p,r,n] + np.sum(de_dg[2,:,p]*dg_dy[2,:,p]*dX_dE[-M:,p,r,n] + np.sum(de_dg[2,:,p]*dg_dG[2,:,:,p]*dX_dE[:N+K,p,r,n][:,np.newaxis]*G_copy[:,2,:,p],axis=0)) + np.sum(de_dE[2,N:,p]*dX_dE[N:N+K,p,r,n]*E_copy[N:,2,p])
    
if p==0: 
  grad_E_test[2,2,p] += de_dE[2,p,p]
  
###########################################
grad_T_test = np.zeros((R))

#dR_dT is R,N+K. dX_dT is tot,N+K, de_dE 3,N+K,N, E is N+K,R,N

grad_T_test[0] = de_dr[0,p]*dR_dT[0,p] + np.sum(de_dg[0,:,p]*dg_dy[0,:,p]*dX_dT[-M:,p] + np.sum(de_dg[0,:,p]*dg_dG[0,:,:,p]*dX_dT[:N+K,p][:,np.newaxis]*G_copy[:,0,:,p],axis=0)) + np.sum(de_dE[0,N:,p]*dX_dT[N:N+K,p]*E_copy[N:,0,p])


grad_T_test[1] = de_dr[1,p]*dR_dT[1,p] + np.sum(de_dg[1,:,p]*dg_dy[1,:,p]*dX_dT[-M:,p] + np.sum(de_dg[1,:,p]*dg_dG[1,:,:,p]*dX_dT[:N+K,p][:,np.newaxis]*G_copy[:,1,:,p],axis=0)) + np.sum(de_dE[1,N:,p]*dX_dT[N:N+K,p]*E_copy[N:,1,p]) + de2_de1*grad_T_test[0]


grad_T_test[2] = de_dr[2,p]*dR_dT[2,p] + np.sum(de_dg[2,:,p]*dg_dy[2,:,p]*dX_dT[-M:,p] + np.sum(de_dg[2,:,p]*dg_dG[2,:,:,p]*dX_dT[:N+K,p][:,np.newaxis]*G_copy[:,2,:,p],axis=0)) + np.sum(de_dE[2,N:,p]*dX_dT[N:N+K,p]*E_copy[N:,2,p])

############################################
grad_H_test = np.zeros((R,M))

#dR_dH is R,N+K,M. dX_dH is tot,N+K,M

for m in range(M):
  grad_H_test[0,m] = de_dr[0,p]*dR_dH[0,p,m] + np.sum(de_dg[0,:,p]*dg_dy[0,:,p]*dX_dH[-M:,p,m] + np.sum(de_dg[0,:,p]*dg_dG[0,:,:,p]*dX_dH[:N+K,p,m][:,np.newaxis]*G_copy[:,0,:,p],axis=0)) + np.sum(de_dE[0,N:,p]*dX_dH[N:N+K,p,m]*E_copy[N:,0,p])
  
for m in range(M):
  grad_H_test[1,m] = de_dr[1,p]*dR_dH[1,p,m] + np.sum(de_dg[1,:,p]*dg_dy[1,:,p]*dX_dH[-M:,p,m] + np.sum(de_dg[1,:,p]*dg_dG[1,:,:,p]*dX_dH[:N+K,p,m][:,np.newaxis]*G_copy[:,1,:,p],axis=0)) + np.sum(de_dE[1,N:,p]*dX_dH[N:N+K,p,m]*E_copy[N:,1,p]) + de2_de1*grad_H_test[0,m]
  
for m in range(M):
  grad_H_test[2,m] = de_dr[2,p]*dR_dH[2,p,m] + np.sum(de_dg[2,:,p]*dg_dy[2,:,p]*dX_dH[-M:,p,m] + np.sum(de_dg[2,:,p]*dg_dG[2,:,:,p]*dX_dH[:N+K,p,m][:,np.newaxis]*G_copy[:,2,:,p],axis=0)) + np.sum(de_dE[2,N:,p]*dX_dH[N:N+K,p,m]*E_copy[N:,2,p])
  
###############################################
grad_C_test = np.zeros((R, tot))

#dR_dC is R,N+K,tot. dX_dH is tot,N+K,M

for k in range(tot):
  grad_C_test[0,k] = de_dr[0,p]*dR_dC_plus[0,p,k] + np.sum(de_dg[0,:,p]*dg_dy[0,:,p]*dX_dC_plus[-M:,p,k] + np.sum(de_dg[0,:,p]*dg_dG[0,:,:,p]*dX_dC_plus[:N+K,p,k][:,np.newaxis]*G_copy[:,0,:,p],axis=0)) + np.sum(de_dE[0,N:,p]*dX_dC_plus[N:N+K,p,k]*E_copy[N:,0,p])
  
for k in range(tot):
  grad_C_test[1,k] = de_dr[1,p]*dR_dC_plus[1,p,k] + np.sum(de_dg[1,:,p]*dg_dy[1,:,p]*dX_dC_plus[-M:,p,k] + np.sum(de_dg[1,:,p]*dg_dG[1,:,:,p]*dX_dC_plus[:N+K,p,k][:,np.newaxis]*G_copy[:,1,:,p],axis=0)) + np.sum(de_dE[1,N:,p]*dX_dC_plus[N:N+K,p,k]*E_copy[N:,1,p]) + de2_de1*grad_C_test[0,k]
  
for k in range(tot):
  grad_C_test[2,k] = de_dr[2,p]*dR_dC_plus[2,p,k] + np.sum(de_dg[2,:,p]*dg_dy[2,:,p]*dX_dC_plus[-M:,p,k] + np.sum(de_dg[2,:,p]*dg_dG[2,:,:,p]*dX_dC_plus[:N+K,p,k][:,np.newaxis]*G_copy[:,2,:,p],axis=0)) + np.sum(de_dE[2,N:,p]*dX_dC_plus[N:N+K,p,k]*E_copy[N:,2,p])
  
###################################################

grad_P_test = np.zeros((3,M,tot))

#dR_dP is R,N+K,M,tot. dX_dP is tot,N+K,M,tot

for m in range(M):
  for k in range(tot):
    grad_P_test[0,m,k] = de_dr[0,p]*dR_dP_plus[0,p,m,k] + np.sum(de_dg[0,:,p]*dg_dy[0,:,p]*dX_dP_plus[-M:,p,m,k] + np.sum(de_dg[0,:,p]*dg_dG[0,:,:,p]*dX_dP_plus[:N+K,p,m,k][:,np.newaxis]*G_copy[:,0,:,p],axis=0)) + np.sum(de_dE[0,N:,p]*dX_dP_plus[N:N+K,p,m,k]*E_copy[N:,0,p])
 
for m in range(M): 
  for k in range(tot):
    grad_P_test[1,m,k] = de_dr[1,p]*dR_dP_plus[1,p,m,k] + np.sum(de_dg[1,:,p]*dg_dy[1,:,p]*dX_dP_plus[-M:,p,m,k] + np.sum(de_dg[1,:,p]*dg_dG[1,:,:,p]*dX_dP_plus[:N+K,p,m,k][:,np.newaxis]*G_copy[:,1,:,p],axis=0)) + np.sum(de_dE[1,N:,p]*dX_dP_plus[N:N+K,p,m,k]*E_copy[N:,1,p]) + de2_de1*grad_C_test[0,k]
for m in range(M):  
  for k in range(tot):
    grad_P_test[2,m,k] = de_dr[2,p]*dR_dP_plus[2,p,m,k] + np.sum(de_dg[2,:,p]*dg_dy[2,:,p]*dX_dP_plus[-M:,p,m,k] + np.sum(de_dg[2,:,p]*dg_dG[2,:,:,p]*dX_dP_plus[:N+K,p,m,k][:,np.newaxis]*G_copy[:,2,:,p],axis=0)) + np.sum(de_dE[2,N:,p]*dX_dP_plus[N:N+K,p,m,k]*E_copy[N:,2,p])