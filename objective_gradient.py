import numpy as np
from compute_J import compute_Jacobian
from compute_J import assign_when
from numba import jit

def steady_state_gradient(strategy, n, l, N, K, M, R, tot,
          phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus, drdot_dG, dxdot_dG, drdot_dE, dxdot_dE, drdot_dH, dxdot_dH, drdot_dT, dxdot_dT, drdot_dC_plus, dxdot_dC_plus, drdot_dC_minus,dxdot_dC_minus, drdot_dP_plus, dxdot_dP_plus, drdot_dP_minus, dxdot_dP_minus):
  R = 3
  # Unpack strategy parameters.
  G[l] = strategy[0:R*M*N].reshape((R,M,N))
  E[l] = strategy[R*M*N:R*M*N + R*N].reshape((R,N))
  T[l] = strategy[R*M*N + R*N:R*M*N + R*N + 1]
  H[l] = strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M]
  C[l] = strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]
  P[l] = strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot].reshape((M,tot))

  # Compute Jacobian
  J = compute_Jacobian(N,K,M,tot,
      phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P,
      ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus)
  eigvals = np.linalg.eigvals(J)
  if np.all(eigvals.real < 0):  # stable if real part of eigenvalues is negative
    stability = True
  else:
    stability = False  # unstable if real part is positive, inconclusive if 0
    

  #print('Jacobian:',str(J))
  # Compute inverse Jacobian
  J_inv = np.linalg.inv(J)


  ## Compute how the steady state of the system changes with respect to each strategy parameter
  dR_dG, dX_dG = multiply_by_inverse_jacobian(drdot_dG, dxdot_dG, J_inv, tot, N, M, R)
  dR_dE, dX_dE = multiply_by_inverse_jacobian(drdot_dE, dxdot_dE, J_inv, tot, N, M, R)
  dR_dT, dX_dT = multiply_by_inverse_jacobian(drdot_dT, dxdot_dT, J_inv, tot, N, M, R)
  dR_dH, dX_dH = multiply_by_inverse_jacobian(drdot_dH, dxdot_dH, J_inv, tot, N, M, R)
  #print('dR*/dH:', dR_dH[1,0,3])
  dR_dC_plus, dX_dC_plus = multiply_by_inverse_jacobian(drdot_dC_plus, dxdot_dC_plus, J_inv, tot, N, M, R)
  dR_dC_minus, dX_dC_minus = multiply_by_inverse_jacobian(drdot_dC_minus, dxdot_dC_minus, J_inv, tot, N, M, R)
  dR_dP_plus, dX_dP_plus = multiply_by_inverse_jacobian(drdot_dP_plus, dxdot_dP_plus, J_inv, tot, N, M, R)
  dR_dP_minus, dX_dP_minus = multiply_by_inverse_jacobian(drdot_dP_minus, dxdot_dP_minus, J_inv, tot, N, M, R)
  return dR_dG, dX_dG, dR_dE, dX_dE, dR_dT, dX_dT, dR_dH, dX_dH, dR_dC_plus, dX_dC_plus, dR_dC_minus, dX_dC_minus, dR_dP_plus, dX_dP_plus, dR_dP_minus, dX_dP_minus, stability

#@jit(nopython=True)
def objective_grad(strategy, n, l, N, K, M, R, tot,
          psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus, dR_dG, dX_dG, dR_dE, dX_dE, dR_dT, dX_dT, dR_dH, dX_dH, dR_dC_plus, dX_dC_plus, dR_dC_minus, dX_dC_minus, dR_dP_plus, dX_dP_plus, dR_dP_minus, dX_dP_minus):
  '''
  inputs:
    strategy for a single actor (flattened)
    n is the actor whose objective we want to optimize
    l is the actor whose strategy it is
    J is the Jacobian (is it calculated or passed in?????)  TODO: remove this parameter
    N,K,M,T meta parameters
    scale parameters
    exponent parameters
    strategy parameters
  modified variables:
    strategy parameters F,H,W,K_p,D_jm will be modified to match input strategy for l
  return the gradient of the objective function at that point for that actor
  '''
  # calculate gradients of objective function for one actor
  # for extraction
  # n's objective, l's strategy (same for resource users) n,l used to be i,j

                            
  # grad G should be R,R,M,N, before being aggregated into one objective. dR_dG is R,N+K,R,M,N. dX_dG is tot,N+K,R,M,N
  grad_G = np.zeros((R,R,M,N))
  
  G_copy = G
  G_copy[:,2,:,1:] = 0
  
  E_copy = E
  E_copy[:,2,1:] = 0  
  
  for i in range(R):
    grad_G[i] = de_dr[i,n] * dR_dG[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,n]*dg_dy[i,:,n], (M,1,1,1)), dX_dG[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,n],dg_dG[i,:,:,n]*G_copy[:,i,:,n]), (N+K,M,1,1,1)),
                                               # 1m            km          km
                    np.reshape(dX_dG[:N+K,l], (N+K,1,R,M,N))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,N:,n]*E_copy[N:,i,n],(K,1,1,1)),dX_dG[N:N+K,l]),axis=0)

  # special case for i=n     
  grad_G[np.arange(R),np.arange(R),:,n] += de_dg[:,:,n]*dg_dG[:,n,:,n]
  
  # subtract off added term in diagonal for dischargers (for whom G is technically inapplicable)
  if n != 0:
    grad_G[2,:,:,n] += -de_dg[2,:,n]*dg_dG[2,n,:,n]
  
  #special case for e2
  grad_G[1] += de2_de1*grad_G[0]
  
  
  # grad wrt E should be R,R,N before being aggregated into one objective
  grad_E = np.zeros((R,R,N))
  
  for i in range(R):
    grad_E[i] = de_dr[i,n] * dR_dE[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,n]*dg_dy[i,:,n], (M,1,1)), dX_dE[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,n],dg_dG[i,:,:,n]*G_copy[:,i,:,n]), (N+K,M,1,1)),
                                               # 1m            km          km
                    np.reshape(dX_dE[:N+K,l], (N+K,1,R,N))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,N:,n]*E[N:,i,n],(K,1,1)),dX_dE[N:N+K,l]),axis=0)
        
  # special case for i=n     
  grad_E[np.arange(R),np.arange(R),n] += de_dE[:,n,n]
  
    # subtract off added term in diagonal for dischargers (for whom E is technically inapplicable)
  if n != 0:
    grad_E[2,2,n] += -de_dE[2,n,n]
  
  #special case for e2
  grad_E[1] += de2_de1*grad_E[0]
  
  # grad wrt T
  
  grad_T = np.zeros((3))

  for i in range(R):
    grad_T[i] = de_dr[i,n] * dR_dT[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,n]*dg_dy[i,:,n], (M)), dX_dT[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,n],dg_dG[i,:,:,n]*G_copy[:,i,:,n]), (N+K,M)),
                                               # 1m            km          km
                    np.reshape(dX_dT[:N+K,l], (N+K,1))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,N:,n]*E_copy[N:,i,n],(K)),dX_dT[N:N+K,l]),axis=0)
          
  #special case for e2
  grad_T[1] += de2_de1*grad_T[0]
  
  # grad wrt H
  

  grad_H = np.zeros((3,M))

  for i in range(R):
    grad_H[i] = de_dr[i,n] * dR_dH[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,n]*dg_dy[i,:,n], (M,1)), dX_dH[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,n],dg_dG[i,:,:,n]*G_copy[:,i,:,n]), (N+K,M,1)),
                                               # 1m            km          km
                    np.reshape(dX_dH[:N+K,l], (N+K,1,M))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,N:,n]*E_copy[N:,i,n],(K,1)),dX_dH[N:N+K,l]),axis=0)
          
  #special case for e2
  grad_H[1] += de2_de1*grad_H[0]
  
  # grad wrt C #########################
  
  grad_C_plus = np.zeros((3,tot))
  grad_C_minus = np.zeros((3,tot))

  for i in range(R):
    grad_C_plus[i] = de_dr[i,n] * dR_dC_plus[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,n]*dg_dy[i,:,n], (M,1)), dX_dC_plus[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,n],dg_dG[i,:,:,n]*G_copy[:,i,:,n]), (N+K,M,1)),
                                               # 1m            km          km
                    np.reshape(dX_dC_plus[:N+K,l], (N+K,1,tot))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,N:,n]*E_copy[N:,i,n],(K,1)),dX_dC_plus[N:N+K,l]),axis=0)
        
    grad_C_minus[i] = de_dr[i,n] * dR_dC_minus[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,n]*dg_dy[i,:,n], (M,1)), dX_dC_minus[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,n],dg_dG[i,:,:,n]*G_copy[:,i,:,n]), (N+K,M,1)),
                                               # 1m            km          km
                    np.reshape(dX_dC_minus[:N+K,l], (N+K,1,tot))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,:,n]*E[:,i,n],(N+K,1)),dX_dC_minus[:N+K,l]),axis=0)
          
  #special case for e2
  grad_C_minus[1] += de2_de1*grad_C_minus[0]
  grad_C_plus[1] += de2_de1*grad_C_plus[0]
  
  # grad wrt P #########################
  
  grad_P_plus = np.zeros((3,M,tot))
  grad_P_minus = np.zeros((3,M,tot))

  for i in range(R):
    grad_P_plus[i] = de_dr[i,n] * dR_dP_plus[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,n]*dg_dy[i,:,n], (M,1,1)), dX_dP_plus[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,n],dg_dG[i,:,:,n]*G_copy[:,i,:,n]), (N+K,M,1,1)),
                                               # 1m            km          km
                    np.reshape(dX_dP_plus[:N+K,l], (N+K,1,M,tot))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,:,n]*E_copy[:,i,n],(N+K,1,1)),dX_dP_plus[:N+K,l]),axis=0)
        
    grad_P_minus[i] = de_dr[i,n] * dR_dP_minus[i,l] + np.sum(np.multiply(np.reshape(de_dg[i,:,n]*dg_dy[i,:,n], (M,1,1)), dX_dP_minus[N+K:,l])
                   # scalar                               jxi         # m                             mji
            + np.sum(
                np.multiply(  # Both factors need to be kmji
                    np.reshape(np.multiply(de_dg[i,:,n],dg_dG[i,:,:,n]*G_copy[:,i,:,n]), (N+K,M,1,1)),
                                               # 1m            km          km
                    np.reshape(dX_dP_minus[:N+K,l], (N+K,1,M,tot))
                       # k1ji
                )
            ,axis=0)  # Sum over k
        ,axis=0) + np.sum(np.multiply(np.reshape(de_dE[i,:,n]*E_copy[:,i,n],(N+K,1,1)),dX_dP_minus[:N+K,l]),axis=0)
          
  #special case for e2
  grad_P_minus[1] += de2_de1*grad_P_minus[0]
  grad_P_plus[1] += de2_de1*grad_P_plus[0]
  
  # average objectives across resources to get an aggregated objective. average is weighted by sigma_tildes, an indication of #importance of each resource to the user
  
  # TO TEST - CHANGE BACK AFTER
  grad_G_avg = grad_G[2]
  grad_E_avg = grad_E[2]
  grad_T_avg = grad_T[2]
  grad_H_avg = grad_H[2]
  grad_C_plus_avg = grad_C_plus[2]
  grad_C_minus_avg = grad_C_minus[2]
  grad_P_plus_avg = grad_P_plus[2]  
  grad_P_minus_avg = grad_P_minus[2]
  
  # grad_G_avg = np.sum(sigma_tildes[:,l][:,np.newaxis,np.newaxis]*grad_G, axis=0)
  # grad_E_avg = np.sum(sigma_tildes[:,l][:,np.newaxis]*grad_E, axis=0)
  # grad_T_avg = np.sum(sigma_tildes[:,l]*grad_T, axis=0)
  # grad_H_avg = np.sum(sigma_tildes[:,l][:,np.newaxis]*grad_H, axis=0)
  # grad_C_plus_avg = np.sum(sigma_tildes[:,l][:,np.newaxis]*grad_C_plus, axis=0)
  # grad_C_minus_avg = np.sum(sigma_tildes[:,l][:,np.newaxis]*grad_C_minus, axis=0)
  # grad_P_plus_avg = np.sum(sigma_tildes[:,l][:,np.newaxis,np.newaxis]*grad_P_plus, axis=0)  
  # grad_P_minus_avg = np.sum(sigma_tildes[:,l][:,np.newaxis,np.newaxis]*grad_P_minus, axis=0)

  grad_C_avg = np.zeros((tot))
  assign_when(grad_C_avg, grad_C_plus_avg, C[l]>=0)
  assign_when(grad_C_avg, -grad_C_minus_avg, C[l]<0)
  
  grad_P_avg = np.zeros((M,tot))
  assign_when(grad_P_avg, grad_P_plus_avg, P[l]>=0)
  assign_when(grad_P_avg, -grad_P_minus_avg, P[l]<0)

  # objective function gradient for extractors
  return np.concatenate((grad_G_avg.flatten(),
                         grad_E_avg.flatten(),
                         grad_T_avg.flatten(),
                         grad_H_avg.flatten(),
                         grad_C_avg.flatten(),
                         grad_P_avg.flatten()))


@jit(nopython=True)
# def multiply_by_inverse_jacobian(drdot_dp, dxdot_dp, J_inv, tot, N, M, R):
  # # shape is the shape of strategy parameter p. 
  # shape = drdot_dp[0].shape
  # # dSdot_dp == how steady state changes wrt p, packed into one variable

  # dSdot_dp = np.concatenate((
                 # drdot_dp,
                 # dxdot_dp),
             # axis=0)

  # size = drdot_dp[0].size
  # strategy_length = np.shape(dSdot_dp)[0] # different for diff strategies, the number of entities that are affected by strategy
  # dSdot_dp = dSdot_dp.reshape(strategy_length, size)  # this should already be true
  

  # # do the actual computation
  # dSS_dp = -J_inv[:,:strategy_length] @ dSdot_dp

  # # unpack
  # dSS_dp = dSS_dp.reshape((tot+3, *shape))
  # dR_dp = dSS_dp[:R]
  # dX_dp = dSS_dp[R:]

  # return dR_dp, dX_dp
  
def multiply_by_inverse_jacobian(drdot_dp, dxdot_dp, J_inv, tot, N, M, R):
  # shape is the shape of strategy parameter p. For example, D_jm is (N,M,M).
  shape = drdot_dp[0].shape
  # dSdot_dp == how steady state changes wrt p, packed into one variable

  dSdot_dp = np.concatenate(
                 (drdot_dp,
                 dxdot_dp),
             axis=0)

  size = drdot_dp[0].size
  dSdot_dp = dSdot_dp.reshape(tot+R, size)  # this should already be true

  # do the actual computation
  dSS_dp = -J_inv @ dSdot_dp

  # unpack
  dSS_dp = dSS_dp.reshape((tot+R, *shape))
  dR_dp = dSS_dp[:R]
  dX_dp = dSS_dp[R:]
  return dR_dp, dX_dp
  
"""
  dSdot_dW_n = np.concatenate((np.broadcast_to(drdot_dW_n,(1,N,N)),dxdot_dW_n,dydot_dW_n), axis=0)
  dSdot_dW_n = dSdot_dW_n.reshape(T,(N)**2)
  dSS_dW_n = -J_inv @ dSdot_dW_n
  dSS_dW_n = dSS_dW_n.reshape(T,N,N)
  dR_dW_n = dSS_dW_n.reshape(T,N,N)[0]
  dX_dW_n = dSS_dW_n.reshape(T,N,N)[1:N+1]
  dY_dW_n = dSS_dW_n.reshape(T,N,N)[N+1:N+1+M]
"""