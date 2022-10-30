import numpy as np
#from mpi4py import MPI
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


entities = pd.read_excel('parameter_files\\base\entity_list.xlsx',sheet_name=None, header=None)
N_list=entities['N'].values[:,0]

K_list=entities['K'].values[:,0]

M_list=entities['M'].values[:,0]

N = len(N_list)
K = len(K_list)
M = len(M_list)

tot = N+K+M

R = 3

#resource_user = 'rural communities'
resource_user = 'small growers'
#resource_user = 'small growers (white area)'
#resource_user = 'investor growers'
#resource_user = 'investor growers (white area)'

path = Path.cwd().joinpath('data')

with open(path.joinpath('strategies_%s'%(resource_user)), 'rb') as f:
  strategies = pickle.load(f)

with open(path.joinpath('stabilities_%s'%(resource_user)), 'rb') as f:
  stabilities = pickle.load(f)
  
all_zeros = np.all(stabilities==[0,0,0,0],axis=1)
stabilities = stabilities[~all_zeros] # filter out entries that have no stabilities recorded (were skipped)

all_stable = np.all(stabilities==[1,1,1,1],axis=1)

strategies = np.array(strategies[all_stable]) 
 
#strategies[np.abs(strategies)<0.01] = 0
avg_strategy = np.average(strategies,axis=0)
avg_strategy[np.abs(avg_strategy)<0.0001] = 0
stdev_strategy = np.std(strategies,axis=0)
median_strategy = np.median(strategies, axis=0)
strategies_binary = np.zeros(np.shape(strategies))
strategies_binary[strategies < 0] = -1
strategies_binary[strategies > 0] = 1
strategies_binary[np.abs(strategies)<5e-3] = 0
strategy_binary = np.sum(strategies_binary,axis=0)

plt.figure()
plt.plot(np.transpose(strategies[:,abs(np.sum(strategies,axis=0))>1e-5]),'.')
#plt.savefig('%s_full_strategies.svg'%(resource_user))

G = avg_strategy[0:R*M*N].reshape((R,M,N))
H = avg_strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M]
C = avg_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]
P = avg_strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot]

threshold = 0.001 # try to filter out some of the noise with the strategies

# sum over target resources and actors
sum_along_G = np.sum(np.sum(G,axis=0),axis=1)
target_M_G = M_list[abs(sum_along_G) > threshold]
G_stdev = stdev_strategy[0:R*M*N].reshape((R,M,N))
stdev_target_G = np.average(np.average(G_stdev,axis=0),axis=1)[abs(sum_along_G)> threshold]
sum_along_G = sum_along_G[abs(sum_along_G)> threshold]
G_median = np.sum(np.sum(median_strategy[0:R*M*N].reshape((R,M,N)),axis=0),axis=1)
target_M_G_median = M_list[abs(G_median) > threshold]
G_median = G_median[abs(G_median)> threshold]

G_counts_agg = np.sum(np.sum(strategy_binary[0:R*M*N].reshape((R,M,N)),axis=0),axis=1)
G_counts = np.where(np.abs(G_counts_agg)>0,G_counts_agg/(len(strategies)*G_counts_agg),0)
#G_counts[G_counts > 1] = 1
#G_counts[G_counts < -1] = -1
target_M_G_counts = M_list[abs(G_counts) > 0.01]
G_counts = G_counts[abs(G_counts)> 0.01]


entity_list = np.concatenate((N_list, K_list, M_list))
target_C = entity_list[np.unique(np.nonzero(abs(C) > threshold))]
C_median = median_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]
target_C_median = entity_list[abs(C_median) > threshold]
C_median = C_median[abs(C_median)> threshold]
C_counts = strategy_binary[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]/len(strategies)
target_C_counts = entity_list[abs(C_counts) > 0.1]
C_counts = C_counts[abs(C_counts)> 0.1]

stdev_target_C = stdev_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot][abs(C) > threshold]


# Average amount of effort put into each of the types of actions
# effort_G = np.sum(np.abs(avg_strategy[0:R*M*N]))
# effort_H = np.sum(np.abs(avg_strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M]))
# effort_C = np.sum(np.abs(avg_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]))
# effort_P = np.sum(np.abs(avg_strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot]))

effort_G = np.sum(np.abs(strategy_binary[0:R*M*N]))/(R*M*N*len(strategies))
effort_H = np.sum(np.abs(strategy_binary[R*M*N + R*N + 1:R*M*N + R*N + 1 + M]))/(M*len(strategies))
effort_C = np.sum(np.abs(strategy_binary[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]))/(tot*len(strategies))
effort_P = np.sum(np.abs(strategy_binary[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot]))/(M*tot*len(strategies))

# standard deviation across runs in amount of effort put into each type of action (averaged across strategy parameters for each type)
std_G = np.average(np.abs(stdev_strategy[0:R*M*N]))
std_H = np.average(np.abs(stdev_strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M]))
std_C = np.average(np.abs(stdev_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]))
std_P = np.average(np.abs(stdev_strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot]))
std_devs = [std_G, std_H, std_C, std_P]

# Actual/pre-optimization strategies (for comparison)
# effort allocation parameters 
G0 = np.zeros((N+K,R,M,N))  # F_i,m,n is ixmxn positive effort for influencing resource extraction governance $
# get indices
EJ_groups = np.nonzero(K_list=='EJ groups')
DACs_idx = np.nonzero(N_list == 'rural communities')
growers = np.nonzero(np.any([N_list == 'small growers', N_list =='investor growers', N_list == 'small growers (white area)', N_list =='investor growers (white area)'],axis=0))[0]
# EJ groups help DACs receive funding for water supply and water
# treatment infrastructure from the state
G0[N+EJ_groups[0],[1,2],np.nonzero(M_list=='Financial Assistance (SWRCB)')[0],DACs_idx] = np.ones((1,2))*1.5 
G0[N+EJ_groups[0],[1,2],np.nonzero(M_list=='Local Water Boards')[0],DACs_idx] = np.ones((1,2))*1.5
G0[DACs_idx,[1,2],np.nonzero(M_list=='Local Water Boards')[0],DACs_idx] = 1.5
G0[DACs_idx,[1,2],np.nonzero(M_list=='County Board of Supervisors')[0],DACs_idx] = 0.75
# UCCE helps growers get grants from NRCS grants
G0[N+np.nonzero(K_list=='UC Extension/research community')[0],2,np.nonzero(M_list=='NRCS')[0],growers] = np.ones((1,1,1,4))
G0 = G0[np.nonzero(N_list == resource_user)] # take part of G for the RU whose strategy we are looking at

# These strategies do not apply to resource users
# E0 = np.zeros((N+K,3,N))
# E0[N+EJ_groups[0],[1,2],DACs_idx] = np.ones((1,2))*0.75
# E0 = np.divide(E0,np.sum(E0,axis=0))
# E0 = np.nan_to_num(E0)
# T0 = np.zeros(N+K)
# T0[N+np.nonzero(K_list == 'Sustainable conservation')[0]] = 0.75
# T0 = T0/np.sum(T0)

path = Path.cwd().joinpath('parameter_files', 'base', 'sigmas.csv')
sigmas_df = pd.read_csv(path)
sigma_weights = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
sigma_weights = np.array(sigma_weights, dtype=[('O', float)]).astype(float)

path = Path.cwd().joinpath('parameter_files', 'base', 'lambdas.xlsx')
lambdas_df = pd.read_excel(path)
lambdas_weights = lambdas_df.fillna(0).values[:,1:] # array of weights for sampling
lambdas_weights = np.array(lambdas_weights, dtype=[('O', float)]).astype(float)

H0 = np.zeros((N+K,M))  # effort for influencing recharge policy 
H0[N + np.nonzero(K_list=='Flood-MAR network')[0], np.nonzero(M_list=='Water Rights Division (SWRCB)')[0]] = 0.4
H0 = np.divide(H0,np.sum(H0,axis=0))
H0 = np.nan_to_num(H0)
H0 = H0[np.nonzero(N_list == resource_user)]

C0 = sigma_weights[:N+K]  # effort for collaboration. C_i,n is ixn 
C0[lambdas_weights[:N+K]>0] = -1*lambdas_weights[:N+K][lambdas_weights[:N+K]>0]
C0 = C0[np.nonzero(N_list == resource_user)]

P0 = np.zeros((N+K,M,tot))
P0[EJ_groups,np.nonzero(M_list=='Local Water Boards')[0],DACs_idx] = 0.75
P0 = P0[np.nonzero(N_list == resource_user)]

total = np.sum(abs(G0))+np.sum(abs(H0))+np.sum(abs(C0))+np.sum(abs(P0))
G0 = np.squeeze(G0)/total
H0 = np.squeeze(H0)/total
C0 = np.squeeze(C0)/total
P0 = np.squeeze(P0)/total


sum_along_G0 = np.sum(np.sum(G0,axis=0),axis=1)
target_M_G0 = M_list[abs(sum_along_G0)>threshold]
sum_along_G0 = sum_along_G0[abs(sum_along_G0)>threshold]

entity_list = np.concatenate((N_list, K_list, M_list))
target_C0 = entity_list[np.unique(np.nonzero(abs(C0)>threshold))]


# Average amount of effort put into each of the types of actions
effort_G0 = np.sum(np.abs(G0))
effort_H0 = np.sum(np.abs(H0))
effort_C0 = np.sum(np.abs(C0))
effort_P0 = np.sum(np.abs(P0))

############################################################################
# plot optimized strategies
fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (15,5))
x = np.arange(len(target_M_G_counts))
plot_G = axs[0].bar(x, G_counts, alpha=0.5)
axs[0].set_xticks(x)
axs[0].set_xticklabels(target_M_G_counts, rotation=30, ha='right')
axs[0].set_ylabel('Proportion of Runs', fontsize = 12)
axs[0].set_title('Target of Policy Change Efforts (G)', fontsize = 16)

x = np.arange(len(target_C_counts))
plot_C = axs[1].bar(x, C_counts, alpha=0.5)
axs[1].set_xticks(x)
axs[1].set_xticklabels(target_C_counts, rotation=30, ha='right')
axs[1].set_title('Target of Support or Undermining (C)', fontsize = 16)

strategy_options = ('Extraction Policy', 'Recharge Policy', 'Direct supporting/undermining', 'Assistance Policy')
y_pos = np.arange(len(strategy_options))
efforts = [effort_G, effort_H, effort_C, effort_P]
axs[2].bar(y_pos, efforts, alpha=0.5)
axs[2].set_xticks(y_pos)
axs[2].set_xticklabels(strategy_options, rotation=30, ha='right')
axs[2].set_title('Strategy Types', fontsize = 16)
#plt.savefig('%s_opt_strategy_counts.svg'%(resource_user),bbox_inches = 'tight')



# plot actual strategies
fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (15,5))
x = np.arange(len(target_M_G0))
bar1_G = axs[0].bar(x,sum_along_G0, alpha=0.5)
axs[0].set_xticks(x)
axs[0].set_xticklabels(target_M_G0, rotation=30, ha='right')
axs[0].set_ylabel('Average Fraction of Effort', fontsize = 12)
axs[0].set_title('Target of Policy Change Efforts (G)', fontsize = 18)

x = np.arange(len(target_C0))
bar1_C = axs[1].bar(x,C0[np.abs(C0)>threshold], alpha=0.5)
axs[1].set_xticks(x)
axs[1].set_xticklabels(target_C0, rotation=30, ha='right')
axs[1].set_ylabel('Average Fraction of Effort', fontsize = 12)
axs[1].set_title('Target of Support or Undermining (C)', fontsize = 18)

efforts = [effort_G0, effort_H0, effort_C0, effort_P0]
axs[2].bar(y_pos, efforts, alpha=0.5)
axs[2].set_xticks(y_pos)
axs[2].set_xticklabels(strategy_options, rotation=30, ha='right')
axs[2].set_title('Strategy Types', fontsize = 18)
plt.savefig('%s_strategy_avgs.svg'%(resource_user),bbox_inches = 'tight')