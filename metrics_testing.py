# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:22:30 2023

@author: nmoll
"""
# L1 version
impact_L1 = np.zeros((3,3))
for i in range(3): # which one is being activated (the one impacting)
  for j in range(3): # the mode
    perturbation = np.zeros((3))
    perturbation[i] = 1
    impact_L1[i] += np.abs(v_L1[:,j])*np.dot(np.abs(w_L1[:,j]),perturbation)/EVS_test[j]
    
#vectorized version of above code 
impact_L1_vectorized = np.transpose(np.abs(v_L1)@np.transpose(w_L1/-np.broadcast_to(EVS_test,(3,3))))
    
# L1 version with absolute value only of each contribution (not each contribution and mode)
impact_L1 = np.zeros((3,3))
for i in range(3): # which one is being activated (the one impacting)
  for j in range(3): # the mode
    perturbation = np.zeros((3))
    perturbation[i] = 1
    impact_L1[i] += v_L1[:,j]*np.dot(w_L1[:,j],perturbation)/EVS_test[j]
  impact_L1[i] = np.abs(impact_L1[i])


# L2 version
impact = np.zeros((3,3))
for i in range(3):
  for j in range(3):
    perturbation = np.zeros((3))
    perturbation[i] = w[:,j] [i]
    impact[i] += np.abs(v[:,j])*np.dot(np.abs(w[:,j]),perturbation)/EVS_test[j]
    
    
# L2 version
impact = np.zeros((3,3))
for i in range(3):
  for j in range(3):
    perturbation = np.zeros((3))
    perturbation[i] = w[:,j] [i]
    impact[i] += v[:,j]**2*np.dot(np.abs(w[:,j]),perturbation)/EVS_test[j]
    
impact = np.sqrt(-impact)
    
    
