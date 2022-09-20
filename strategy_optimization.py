import numpy as np
from scipy import optimize
from objective_gradient import objective_grad
from objective_gradient import steady_state_gradient

def correct_scale_params(sigmas, lambdas, sigma_hats, lambda_hats, alloc_params, i, N, K, betas, beta_hats, beta_tildes, beta_bars, du_dx_plus, du_dx_minus, etas, eta_hats, eta_bars):
  '''
  Corrects scale parameters (sigmas, sigma_hats, or lambdas) to be consisent with optimization
  results. Takes in scale parameters (2d) and strategy parameters for a particular actor i (1d),
  and sets scale parameters to 0 if the corresponding strategy parameters are 0, ensuring
  that the scale parameters still add to 1.
  '''
  new_zeros_sigmas = (alloc_params <= 0)
  sigmas[i, new_zeros_sigmas] = 0
  all_zeros_sigmas = (np.sum(sigmas, axis=0) == 0)

  # for all columns that now have new zeros (but are not all zeros), renormalize to make sure the rows still sum to 1
  sigmas[:, ~all_zeros_sigmas & new_zeros_sigmas] /= np.broadcast_to(
          np.expand_dims(
              np.sum(sigmas[:, ~all_zeros_sigmas & new_zeros_sigmas], axis = 0),
              axis=0
          ),
          np.shape(sigmas[:, ~all_zeros_sigmas & new_zeros_sigmas])
      )
  sigmas = np.nan_to_num(sigmas)
      
  # new_zeros_sigma_hats = (alloc_params[N+K:] <= 0)
  # sigma_hats[i, new_zeros_sigma_hats] = 0
  # all_zeros_sigma_hats = (np.sum(sigmas, axis=0) == 0)

  # # for all columns that now have new zeros (but are not all zeros), renormalize to make sure the rows still sum to 1
  # sigma_hats[:, ~all_zeros_sigma_hats & new_zeros_sigma_hats] /= np.broadcast_to(
          # np.expand_dims(
              # np.sum(sigma_hats[:, ~all_zeros_sigma_hats & new_zeros_sigma_hats], axis = 0),
              # axis=0
          # ),
          # np.shape(sigma_hats[:, ~all_zeros_sigma_hats & new_zeros_sigma_hats])
      # )    
   
  # Create views of the part for resource users
  user_beta_tildes = beta_tildes[0, :N]
  user_beta_hats = beta_hats[0, :N]
  user_betas = betas[0, :N]
  user_zeros = all_zeros_sigmas[:N]

  user_betas[user_zeros] = 0
  user_beta_sums = user_beta_hats[user_zeros] + user_betas[user_zeros] + user_beta_tildes[user_zeros]
  user_beta_hats[user_zeros] /= user_beta_sums
  user_beta_hats = np.nan_to_num(user_beta_hats)
  user_beta_tildes[user_zeros] /= user_beta_sums
  user_beta_tildes = np.nan_to_num(user_beta_tildes)

  # Create views of the part for non-resource users
  nonuser_beta_tildes = beta_tildes[0, N:]
  nonuser_betas = betas[0, N:]
  nonuser_beta_bars = beta_bars[0, N:]
  nonuser_zeros = all_zeros_sigmas[N:]
  nonuser_du_dx = du_dx_plus[N:]

  nonuser_beta_tildes[nonuser_zeros] = 0
  nonuser_du_dx[nonuser_zeros] *= nonuser_beta_bars[nonuser_zeros]
  nonuser_beta_bars[nonuser_zeros] = 1
     
  new_zeros_lambdas = (alloc_params >= 0)
  lambdas[i, new_zeros_lambdas] = 0
  all_zeros_lambdas = (np.sum(lambdas, axis=0) == 0)
  # for all columns that now have new zeros (but are not all zeros), renormalize to make sure the rows still sum to 1

  lambdas[:, ~all_zeros_lambdas & new_zeros_lambdas] /= np.broadcast_to(
          np.expand_dims(
              np.sum(lambdas[:, ~all_zeros_lambdas & new_zeros_lambdas], axis = 0),
              axis=0
          ),
          np.shape(lambdas[:, ~all_zeros_lambdas & new_zeros_lambdas])
      )
      
      
  # new_zeros_lambda_hats = (alloc_params[N+K:] >= 0)
  # lambda_hats[i, new_zeros_lambda_hats] = 0
  # all_zeros_lambda_hats = (np.sum(lambda_hats, axis=0) == 0)
  # # for all columns that now have new zeros (but are not all zeros), renormalize to make sure the rows still sum to 1

  # lambda_hats[:, ~all_zeros_lambda_hats & new_zeros_lambda_hats] /= np.broadcast_to(
          # np.expand_dims(
              # np.sum(lambda_hats[:, ~all_zeros_lambda_hats & new_zeros_lambda_hats], axis = 0),
              # axis=0
          # ),
          # np.shape(lambda_hats[:, ~all_zeros_lambda_hats & new_zeros_lambda_hats])
      # )

  etas[0, all_zeros_lambdas] = 0
  eta_bars = 1 - eta_hats[0] - etas[0]

  

# If strategy does not have all efforts >= 0, project onto space of legal strategies
def boundary_projection(mu, strategy, plane):
  return np.sum(np.maximum(strategy*plane - mu, 0)) - 1


def grad_descent_constrained(initial_point, objective, alpha, i, N, K, M, R, tot,
        phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus, drdot_dG, dxdot_dG, drdot_dE, dxdot_dE, drdot_dH, dxdot_dH, drdot_dT, dxdot_dT, drdot_dC_plus, dxdot_dC_plus, drdot_dC_minus,dxdot_dC_minus, drdot_dP_plus, dxdot_dP_plus, drdot_dP_minus, dxdot_dP_minus):
    
  '''
  inputs:
    initial_point is the initial strategy
    n is the actor whose objective we want to optimize
    l is the actor whose strategy it is
    J is the Jacobian
    N,K,M,T meta parameters
    scale parameters
    exponent parameters
    strategy parameters????
  return the new and improved strategy
  '''
  x = initial_point  # strategy
  #print('')
  #print('strategy:',x[403])

  # calculate how steady state changes with respect to strategy parameters
  dR_dG, dX_dG, dR_dE, dX_dE, dR_dT, dX_dT, dR_dH, dX_dH, dR_dC_plus, dX_dC_plus, dR_dC_minus, dX_dC_minus, dR_dP_plus, dX_dP_plus, dR_dP_minus, dX_dP_minus, stability, G, E, T, H, C, P\
  = steady_state_gradient(x, objective, i, N, K, M, R, tot,
          phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus, drdot_dG, dxdot_dG, drdot_dE, dxdot_dE, drdot_dH, dxdot_dH, drdot_dT, dxdot_dT, drdot_dC_plus, dxdot_dC_plus, drdot_dC_minus,dxdot_dC_minus, drdot_dP_plus, dxdot_dP_plus, drdot_dP_minus, dxdot_dP_minus)
          
          
  # calculate how objective changes wrt strategy parameters (the gradient)
  grad = objective_grad(x, objective, i, N, K, M, R, tot,
          psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus, dR_dG, dX_dG, dR_dE, dX_dE, dR_dT, dX_dT, dR_dH, dX_dH, dR_dC_plus, dX_dC_plus, dR_dC_minus, dX_dC_minus, dR_dP_plus, dX_dP_plus, dR_dP_minus, dX_dP_minus)
  #print('gradient:',grad[403])

  d = len(x)
  #v[n] = 0.9*v[n] + alpha*grad
  # Follow the projected gradient for a fixed step size alpha
  if np.sum(abs(x + alpha*grad)) > 1:
    alpha = alpha/10
  x = x + alpha*grad
  plane = np.sign(x)
  plane[abs(plane)<0.00001] = 1
  if sum(abs(x)) > 1:
    #project point onto plane
    x = x + plane*(1-sum(plane*x))/d
    x[abs(x)<0.00001] = 0
    if np.sum(x*plane) != 0:
      x /= np.sum(x*plane) # Normalize to be sure (get some errors without this)

    # If strategy does not have all efforts >= 0, project onto space of legal strategies
    if np.any(x*plane < -0.0001):
      try:
        ub = np.sum(abs(x)) #np.sum(abs(x[x*plane>0]))
        mu = optimize.brentq(boundary_projection, 0, ub, args=(x, plane))
      except:
        print('bisection bounds did not work')
        raise Exception('bisection bounds did not work')
      x = plane * np.maximum(x*plane - mu, 0)

  return x, stability, grad, alpha# normally return only x

def ODE_gradient(N, K, M, tot, R, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus):
  # Compute how the rhs of system changes with respect to each strategy parameter
  
  # For strategy G ####################
  drdot_dG = np.zeros((R,N+K,R,M,N))
  drdot_dG[0,:,0] = -phis[0]*psis[0]*np.multiply(psi_tildes[0].reshape(1,1,N),np.multiply(de_dg[0],dg_dG[0]))
  drdot_dG[1,:,1] = -phis[1]*np.multiply(psi_tildes[1].reshape(1,1,N),np.multiply(de_dg[1],dg_dG[1]))
  drdot_dG[1,:,0] = -phis[1]*de2_de1*np.multiply(psi_tildes[1].reshape(1,1,N),np.multiply(de_dg[0],dg_dG[0]))
  drdot_dG[2,:,2] = phis[1]*np.multiply(psi_tildes[2].reshape(1,1,N),np.multiply(de_dg[2],dg_dG[2])) 
  
  dxdot_dG = np.zeros([N+K+M,N+K,R,M,N])
#  for i in range(N):
#    dxdot_dF[i,:,:,i] = alphas[0,i]*betas[0,i]*db_de[0,i]*de_dg[0,:,i]*dg_dG[:,:,i]
  dxdot_dG[np.arange(0,N),:,0,:,np.arange(0,N)] = np.transpose(np.multiply(
             #n k m (gets rid of last index)
        np.reshape(alphas[0,:N]*beta_tildes[0,:N]*(sigma_tildes[0,:N]*db_de[0,:N] + sigma_tildes[1,:N]*db_de[1,:N]*de2_de1), 
        (1,1,N)),
                    # 1n    1n    1n
        np.multiply(de_dg[0],dg_dG[0])
                  # 1mn    kmn
      ), (2,0,1))  # transpose kmn -> nkm
    
  dxdot_dG[np.arange(0,N),:,1,:,np.arange(0,N)] = np.transpose(np.multiply(
           #n k m (gets rid of last index)
      np.reshape(alphas[0,:N]*beta_tildes[0,:N]*sigma_tildes[1,:N]*db_de[1,:N], 
      (1,1,N)),
                  # 1n    1n    1n
      np.multiply(de_dg[1],dg_dG[1])
                # 1mn    kmn
    ), (2,0,1))
    
  dxdot_dG[0,:,2,:,0] = np.multiply(
           #n k m (gets rid of last index)
      alphas[0,0]*beta_tildes[0,0]*sigma_tildes[2,0]*db_de[2,0],
      np.multiply(de_dg[2,:,0],dg_dG[2,:,:,0])
    )    
  
  # For strategy E
  
  drdot_dE = np.zeros((R,N+K,R,N))
  drdot_dE[0,:,0] = (-phis[0]*psis[0]*psi_tildes[0,])[np.newaxis]*de_dE[0,:,:]
  drdot_dE[1,:,1] = (-phis[1]*psi_tildes[1])[np.newaxis]*de_dE[1,:,:]
  drdot_dE[1,:,0] = (-phis[1]*psi_tildes[1])[np.newaxis]*de2_de1*de_dE[0,:,:]
  drdot_dE[2,:,2] = (phis[1]*psi_tildes[2])[np.newaxis]*de_dE[2,:,:]
  
  dxdot_dE = np.zeros((N+K+M,N+K,R,N))
  dxdot_dE[np.arange(0,N),:,0,np.arange(0,N)] = np.transpose(np.multiply(
             #n k m (gets rid of last index)
        np.reshape(alphas[0,:N]*beta_tildes[0,:N]*(sigma_tildes[0,:N]*db_de[0,:N] + sigma_tildes[1,:N]*db_de[1,:N]*de2_de1), 
        (1,N)),
        de_dE[0]
      ))  
  dxdot_dE[np.arange(0,N),:,1,np.arange(0,N)] = np.transpose(np.multiply(
        np.reshape(alphas[0,:N]*beta_tildes[0,:N]*sigma_tildes[1,:N]*db_de[1,:N], 
        (1,N)),
        de_dE[1]
        # N+K,N
      ))

  dxdot_dE[0,:,2,0] = np.transpose(np.multiply(
             #n k m (gets rid of last index)
        alphas[0,0]*beta_tildes[0,0]*sigma_tildes[2,0]*db_de[2,0],
        de_dE[2,:,0]
      ))      
 
  # For strategy H (recharge policy)
  drdot_dH = np.zeros((R,N+K,M))
  drdot_dH[0] = -phis[0]*psi_bars[0]*np.multiply(np.transpose(dt_dh), dh_dH)
  drdot_dH[1] = phis[1]*psi_bars[1]*np.multiply(np.transpose(dt_dh), dh_dH)
  #print('drdot/dH:', drdot_dH[1,0,3])
  drdot_dH[2] = -phis[1]*psi_bars[1]*np.multiply(np.transpose(dt_dh), dh_dH)
  
  dxdot_dH = np.zeros((N+K+M,N+K,M))
  
  # For strategy T (directly influence recharge)
  drdot_dT = np.zeros((R,N+K))
  drdot_dT[0] = -phis[0]*psi_bars[0]*dt_dT
  drdot_dT[1] = phis[1]*psi_bars[1]*dt_dT
  drdot_dT[2] = -phis[1]*psi_bars[1]*dt_dT
  
  dxdot_dT = np.zeros((N+K+M,N+K))

  # For strategy C
  drdot_dC_plus = np.zeros((R,N+K,tot))
  drdot_dC_minus = np.zeros((R,N+K,tot))
  # dxdot_n/dC_k,i is nxkxi
  dxdot_dC_plus = np.zeros((tot,N+K,tot))
  dxdot_dC_plus[np.arange(0,tot),:,np.arange(0,tot)] =  np.transpose(np.multiply(alphas*betas,np.multiply(sigmas,dc_dC)))
  
  dxdot_dC_minus = np.zeros((tot,N+K,tot))
  dxdot_dC_minus[np.arange(0,tot),:,np.arange(0,tot)] =  np.transpose(np.multiply(alphas*etas,np.multiply(lambdas,dc_dC)))

  # For strategy P
  drdot_dP_plus = np.zeros((R,N+K,M,tot))
  drdot_dP_minus = np.zeros((R,N+K,M,tot))
  dxdot_dP_plus = np.zeros((tot,N+K,M,tot))
  dxdot_dP_plus[np.arange(0,tot),:,:,np.arange(0,tot)] = np.transpose(np.multiply(alphas*beta_hats,np.multiply(sigma_hats,dp_dP)),(2,0,1))
  
  dxdot_dP_minus = np.zeros((tot,N+K,M,tot))
  dxdot_dP_minus[np.arange(0,tot),:,:,np.arange(0,tot)] = np.transpose(np.multiply(alphas*eta_hats,np.multiply(lambda_hats,dp_dP)),(2,0,1))


  return (drdot_dG, dxdot_dG, drdot_dE, dxdot_dE, drdot_dH, dxdot_dH, drdot_dT, dxdot_dT, drdot_dC_plus, dxdot_dC_plus, drdot_dC_minus,dxdot_dC_minus, drdot_dP_plus, dxdot_dP_plus, drdot_dP_minus, dxdot_dP_minus)
  

def optimize_strategy(max_iters, i, N, K, M, tot, R, 
    phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus):
  '''
  inputs:
    max_iters - maximum number of iterations
    i - the actor for whom we are optimizing
    J - Jacobian
    N,K,M,T meta parameters
    scale parameters
    exponent parameters
    strategy parameters (initial value given by sample function)
  returns
    optimized strategy parameters
    updated sigmas and lamdas
  '''
  # Initialize strategy
  strategy = np.zeros(len(G[0].flatten()) + len(E[0].flatten()) + len(T[0].flatten()) + len(H[0].flatten()) + len(C[0].flatten()) + len(P[0].flatten()))
  # Make scale parameters of actor's effect on others non-zero
  sigmas[i,:] += np.random.uniform(0.1,1) # set all to the same random number  
  #sigmas[i,:] += np.random.uniform(0.1,1,(np.shape(sigmas[i,:]))) # set all to the same random number
  #sigmas[i,10] += 1 # GET RID OF THIS - JUST FOR AN EXPERIMENT ############################################
  sigmas = sigmas/np.sum(sigmas,axis=0) # renormalize to make sure columns sum to 1
  lambdas[i,:] = np.random.uniform(0.1,1)
  lambdas = lambdas/np.sum(lambdas,axis=0)
  
  #v = np.zeros((N, len(strategy)) # velocity for gradient descent with momentum
  max_diff = 1  # arbitrary initial value, List of differences in euclidean distance between strategies in consecutive iterations
  iterations = 0
  objective = i # this isn't the case for non-resource users
  alpha = 0.0001
  tolerance = alpha/10 #
  
  #things to keep track of
  stability_history = np.zeros(max_iters)
  strategy_diffs = []
  strategy_history = []  # a list of the strategies at each iteration
  strategy_history.append(strategy.copy())
  converged = True
  grad_history = []
  sum_below_1 = True

  (drdot_dG, dxdot_dG, drdot_dE, dxdot_dE, drdot_dH, dxdot_dH, drdot_dT, dxdot_dT, drdot_dC_plus, dxdot_dC_plus, drdot_dC_minus,dxdot_dC_minus, drdot_dP_plus, dxdot_dP_plus, drdot_dP_minus, dxdot_dP_minus) = ODE_gradient(N, K, M, tot, R, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus)
  
  
  while (max_diff > tolerance) and iterations < max_iters:
    new_strategy, stability, grad, alpha = grad_descent_constrained(strategy, objective, alpha, i, N, K, M, R, tot,
        phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus, drdot_dG, dxdot_dG, drdot_dE, dxdot_dE, drdot_dH, dxdot_dH, drdot_dT, dxdot_dT, drdot_dC_plus, dxdot_dC_plus, drdot_dC_minus,dxdot_dC_minus, drdot_dP_plus, dxdot_dP_plus, drdot_dP_minus, dxdot_dP_minus)
    
    stability_history[iterations] = stability


    # Check if there are new zeros or changes in the sign of the strategy parameters to see if we need to update scale parameters
    # (e.g. for portion of gain through collaboration) to make sure they are consistent with our new
    # strategy parameters.
    
    # if there are more zeros in the new strategy than the previous, or changes in the sign, and the scale parameters aren't already all zeros
    if np.count_nonzero(new_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]) < np.count_nonzero(strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]) or np.any(strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot] * new_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot] < 0) and (np.any(sigmas[i] > 0) or np.any(lambdas[i] > 0)):
        
      correct_scale_params(sigmas, lambdas, sigma_hats, lambda_hats, new_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot], i, N, K, betas, beta_hats, beta_tildes, beta_bars, du_dx_plus, du_dx_minus, etas, eta_hats, eta_bars)
      
      (drdot_dG, dxdot_dG, drdot_dE, dxdot_dE, drdot_dH, dxdot_dH, drdot_dT, dxdot_dT, drdot_dC_plus, dxdot_dC_plus, drdot_dC_minus,dxdot_dC_minus, drdot_dP_plus, dxdot_dP_plus, drdot_dP_minus, dxdot_dP_minus) = ODE_gradient(N, K, M, tot, R, phis, psis, psi_bars, eq_R_ratio, psi_tildes, alphas, beta_tildes, sigma_tildes, betas, beta_hats, beta_bars, sigmas, sigma_hats, etas, eta_bars, eta_hats, lambdas, lambda_hats, G, E, T, H, C, P, ds_dr, de_dr, dt_dr, de2_de1, de_dg, de_dE, dg_dG, dh_dH, dg_dy, dh_dy, dt_dh, dt_dT, db_de, dc_dC, dp_dP, dp_dy, du_dx_plus, du_dx_minus)
        #print('updating scale params')
      # update strategy and gradient for this actor
    strategy = new_strategy
    strategy_history.append(strategy.copy())
    grad_history.append(grad.copy())
    
    
    if iterations >= 30:
    # compute difference in strategies to check convergence
      strategy_history_10 = np.array(strategy_history[-30:]).reshape((30,len(strategy)))
      strategy_diff = np.linalg.norm(strategy_history_10[:29,:]-strategy_history_10[-1,:], axis = 1)
      strategy_diffs.append(strategy_diff[-1])
      max_diff = max(strategy_diff)

    iterations += 1
    if iterations == max_iters - 1:
      converged = False
      
  stability_2 = stability_history[0] == True
  stability_3 = np.all(stability_history[1:]==True)
    

  return strategy, stability_2, stability_3, converged, np.array(strategy_history), grad_history