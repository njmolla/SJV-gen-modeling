import numpy as np
from numba import jit


#@jit(nopython=True)
def compute_Jacobian(N,K,M,tot,
      phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P,
      ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus):

  # --------------------------------------------------------------------------
  # Compute Jacobian (vectorized)
  # --------------------------------------------------------------------------
  J = np.zeros((tot+3,tot+3))
  # dr•/dr
  # Jacobian elements calculated separately for each resource
  J[0,0] = phis[0]*(ds_dr[0] - psis[0]*np.sum(psi_tildes[0,:]*de_dr[0,:N])-psi_bars[0]*dt_dr)
  J[1,0] = phis[1]*(psi_bars[1]*dt_dr - np.sum(psi_tildes[1,:]*de2_de1*de_dr[0,:N]))
  J[1,1] = phis[1]*(psis[1]*ds_dr[1]-np.sum(psi_tildes[1,:]*de_dr[1,:N]))
  J[2,0] = phis[1]*psi_bars[1]*dt_dr
  J[2,1] = phis[1]*psis[1]*ds_dr[1]
  J[2,2] = -phis[1]
  

  # dr•/dx (1x(N+K))
  # For the NxMxN stuff: i = axis 0, m = axis 1, n = axis 2
  J[0,3:N+K+3] = phis[0] * (-psis[0]*
        np.sum(np.expand_dims(psi_tildes[0,:],axis=0)*(de_dE[0]*E[:,0] + np.sum(np.multiply(de_dg[0], dg_dG[0] * G[:,0]),axis=1)),axis = 1) 
        - psi_bars[0] * (dt_dT * T + np.sum(np.multiply(np.transpose(dt_dh), dh_dH * H), axis = 1)))
        
  J[1,3:N+K+3] = phis[1]*(np.multiply(psi_bars[1], dt_dT * T + np.sum(np.multiply(np.transpose(dt_dh), dh_dH * H), axis = 1)) - np.sum(
  np.expand_dims(psi_tildes[1,:],axis=0) * de2_de1 * (de_dE[0]*E[:,0] + np.sum(np.multiply(de_dg[0], dg_dG[0] * G[:,0]),
  axis=1) + de_dE[1]*E[:,1] + np.sum(np.multiply(de_dg[1], dg_dG[1] * G[:,1]),axis=1)), axis=1))
  
  J[2,3:N+K+3] = phis[1]*(np.sum(np.expand_dims(psi_tildes[2,:],axis=0)*(de_dE[2]*E[:,2] + np.sum(np.multiply(de_dg[2], dg_dG[2] * G[:,2]),axis=1)), axis=1) - psi_bars[1] * (dt_dT * T + np.sum(np.multiply(np.transpose(dt_dh), dh_dH * H), axis = 1)))
  
 

  # dr•/dy
  J[0,N+K+3:] = phis[0] * (-psis[0] * np.sum(
        np.multiply(np.expand_dims(psi_tildes[0,:N],axis=0), de_dg[0] * dg_dy[0]),                  # 1xn             1xmxn     mxn
       axis = 1) - (psi_bars[0]*dt_dh*np.expand_dims(dh_dy, axis=1))[:,0])
       
  J[1,N+K+3:] = phis[1] * ((psi_bars[1]*dt_dh*np.expand_dims(dh_dy, axis=1))[:,0] - np.sum(
        np.multiply(np.expand_dims(psi_tildes[1,:N],axis=0), np.multiply(de2_de1, de_dg[0] * dg_dy[0])  + de_dg[1]*dg_dy[1]),                 # 1xn             1xmxn     mxn
       axis = 1))
       
  J[2,N+K+3:] = phis[1] * (np.sum(
        np.multiply(np.expand_dims(psi_tildes[2,:N],axis=0),de_dg[2] * dg_dy[2]), axis = 1) - (psi_bars[1]*dt_dh*np.expand_dims(dh_dy, axis=1))[:,0])                 # 1xn             1xmxn     mxn

  # dx•/dr
  J[3:N+3,0] = alphas[0,:N] * (beta_tildes[0,:N]*sigma_tildes[0]*db_de[0,:N]*de_dr[0,:N])+ beta_tildes[0,:N]*sigma_tildes[1]*db_de[1,:N]*de2_de1*de_dr[0,:N]
  J[3:N+3,1] = alphas[0,:N] * (beta_tildes[0,:N]*sigma_tildes[1]*db_de[1,:N]*de_dr[1,:N])
  J[3:N+3,2] = alphas[0,:N] * (beta_tildes[0,:N]*sigma_tildes[2]*db_de[2,:N]*de_dr[2,:N])
                                           # 1xn
  # dx•/dx for n != i (NxN+K)

  J[3:N+3,3:N+K+3] = np.transpose(np.multiply(alphas[0,np.newaxis,:N],
        np.multiply(np.expand_dims(beta_tildes[0,:N]*sigma_tildes[0,:N]*db_de[0,:N], axis = 0),       de_dE[0] * E[:,0] + np.sum(np.multiply(de_dg[0], dg_dG[0]*G[:,0]), axis = 1))
                     #  1xn                                ixmxn
        + np.multiply(np.expand_dims(beta_tildes[0,:N]*sigma_tildes[1,:N]*db_de[1,:N], axis=0),       de_dE[1] * E[:,1] + np.sum(np.multiply(de_dg[1], dg_dG[1]*G[:,1]), axis = 1) + np.multiply(de2_de1,de_dE[0] * E[:,0] + np.sum(np.multiply(de_dg[0],dg_dG[0]*G[:,0]),axis=1)))
        
        + np.multiply(np.expand_dims(beta_tildes[0,:N]*sigma_tildes[2,:N]*db_de[2,:N], axis=0), de_dE[2] * E[:,2] + np.sum(np.multiply(de_dg[2], dg_dG[2]*G[:,2]), axis = 1))

      ))
      
  # dx•/dx for n != i, adding in terms that apply to non-govt orgs and dec centers (TxN+K)
  P_plus = np.zeros(np.shape(P))
  assign_when(P_plus, P, P>=0)
  P_minus = np.zeros(np.shape(P))
  assign_when(P_minus, P, P<0)
  C_plus = np.zeros(np.shape(C))
  assign_when(C_plus, C, C>=0)
  C_minus = np.zeros(np.shape(C))
  assign_when(C_minus, C, C<0)
  
  J[3:tot+3,3:N+K+3] += np.transpose(np.multiply(alphas,
                                    # 1xT
         np.multiply(beta_hats,np.sum(np.multiply(sigma_hats,dp_dP*P_plus),axis = 1)) - np.multiply(eta_hats,np.sum(np.multiply(lambda_hats,dp_dP*P_minus),axis=1))
                                          #(N+K)xT
        + np.multiply(betas,sigmas*dc_dC*C_plus) - np.multiply(etas,lambdas*dc_dC*C_minus)
                       # 1xn            ixn
      ))

 # dx•/dx for n = i (overwrite the diagonal) (unvectorized)
 # for RUs
  for i in range(N):

    J[i+3,i+3] = alphas[0,i] * (
                     beta_tildes[0,i]* sigma_tildes[0,i] * db_de[0,i] * np.sum(de_dg[0,:,i] * dg_dG[0,i,:,i] * G[i,0,:,i])
                   + beta_tildes[0,i]* sigma_tildes[1,i] * db_de[1,i] * np.sum(de_dg[1,:,i] * dg_dG[1,i,:,i] * G[i,1,:,i])
                   + de2_de1[i] * np.sum(de_dg[0,:,i] * dg_dG[0,i,:,i] * G[i,0,:,i])
                   + beta_tildes[0,i]* sigma_tildes[2,i] * db_de[2,i] * np.sum(de_dg[2,:,i] * dg_dG[2,i,:,i] * G[i,2,:,i])
                   + beta_hats[0,i] * np.sum(sigma_hats[:,i]*dp_dP[i,:,i]*P_plus[i,:,i]) - eta_hats[0,i] * np.sum(lambda_hats[:,i]*dp_dP[i,:,i]*P_minus[i,:,i])
                   + beta_bars[0,i] * du_dx_plus[i]
                   - eta_bars[i] * du_dx_minus[i]
                 )
                 
#  indices = np.arange(1,N+1)  # Access the diagonal of the actor part.
#  J[indices,indices] = alphas[0] * (
#        (beta_tildes*db_de)[0]*np.sum(de_dg[0]*dg_dG[np.arange(N),:,np.arange(N)].transpose()*G[np.arange(N),:,np.arange(N)].transpose(),axis=0)
#        #                                                          mxn                                 mxn
#        + (sigma_tildes*dq_da)[0]*np.sum(da_dp[0]*dp_dH[np.arange(N),:,np.arange(N)].transpose()*H[np.arange(N),:,np.arange(N)].transpose(),axis=0)
#        - eta_bars*dl_dx
#      )

  # dx•/dy (NxM)
  J[3:N+3,N+K+3:] = np.transpose(alphas[0,:N] * (
        np.multiply(beta_tildes[0,:N]*sigma_tildes[0,:N]*db_de[0,:N], de_dg[0]*dg_dy[0])
                              # N                              
        + np.multiply(beta_tildes[0,:N]*sigma_tildes[1,:N]*db_de[1,:N], de_dg[1]*dg_dy[1] + np.multiply(de2_de1,de_dg[0]*dg_dy[0]))
        + np.multiply(beta_tildes[0,:N]*sigma_tildes[2,:N]*db_de[2,:N], de_dg[2]*dg_dy[2])
        + np.multiply(beta_hats[0,:N], sigma_hats[:,:N]*dp_dy[:,:N]) - np.multiply(eta_hats[0,:N], lambda_hats[:,:N]*dp_dy[:,:N])
        
      ))
  # dx•/dy (K+MxM)   
  J[N+3:,N+K+3:] = np.transpose(alphas[0,N:] * (
           np.multiply(beta_hats[0,N:], sigma_hats[:,N:]*dp_dy[:,N:]) - np.multiply(eta_hats[0,N:], lambda_hats[:,N:]*dp_dy[:,N:])
      ))
  
  # dx•/dx for n = i (non-RUs) 
  for i in range(K):

    J[i+N+3,i+N+3] = alphas[0,N+i] * (
                   beta_hats[0,N+i] * np.sum(sigma_hats[:,N+i]*dp_dP[N+i,:,N+i]*P_plus[N+i,:,N+i]) - eta_hats[0,N+i] * np.sum(lambda_hats[:,N+i]*dp_dP[N+i,:,N+i]*P_minus[N+i,:,N+i])
                   + beta_bars[0,N+i] * du_dx_plus[N+i]
                   - eta_bars[N+i] * du_dx_minus[N+i]
                 )  
      
  # dy•/dy
  for i in range(M):

    J[i+N+K+3,i+N+K+3] = alphas[0,N+K+i] * (
                   beta_bars[0,N+K+i] * du_dx_plus[N+K+i]
                   - eta_bars[N+K+i] * du_dx_minus[N+K+i]
                 )

  return J


#@jit(nopython=True)
def assign_when(lhs, rhs, conditions):
  """
  This does
     lhs[conditions] = rhs[conditions]
  (this is done because Numba doesn't like logical indexing)
  lhs and rhs are any numpy arrays with the same shape
  conditions is a boolean numpy array with the same shape
  For example, to do
     x[x > 0] = y[x > 0]
  use
      assign_when(x, y, x > 0)
  """
  for nd_index in np.ndindex(lhs.shape):
    if conditions[nd_index]:
      lhs[nd_index] = rhs[nd_index]