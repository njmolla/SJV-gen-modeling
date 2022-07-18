import numpy as np
#from mpi4py import MPI
import pickle
import pandas as pd
import matplotlib.pyplot as plt


entities = pd.read_excel('parameter_files\entity_list.xlsx',sheet_name=None, header=None)
N_list=entities['N'].values[:,0]

K_list=entities['K'].values[:,0]

M_list=entities['M'].values[:,0]

N = len(N_list)
K = len(K_list)
M = len(M_list)

tot = N+K+M

R = 3

resource_user = 'rural communities'
#resource_user = 'small growers'
#resource_user = 'investor growers'
#resource_user = 'investor growers (white area)'

with open('strategies_%s_looping'%(resource_user), 'rb') as f:
  strategies = pickle.load(f)  
 
strategies[np.abs(strategies)<0.0015] = 0
avg_strategy = np.average(strategies,axis=0)
stdev_strategy = np.std(strategies,axis=0)

plt.figure()
#plt.plot(np.transpose(strategies))
plt.figure()
plt.plot(np.transpose(strategies[:,abs(np.sum(strategies,axis=0))>1e-5]),'.')
plt.savefig('%s_full_strategies.svg'%(resource_user))

G = avg_strategy[0:R*M*N].reshape((R,M,N))
H = avg_strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M]
C = avg_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]
P = avg_strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot]


target_M_G = M_list[np.unique(np.nonzero(G)[1])]
sum_along_G = np.sum(np.sum(G,axis=0),axis=1)

entity_list = np.concatenate((N_list, K_list, M_list))
target_C = entity_list[np.unique(np.nonzero(C))]

plt.figure()
x = np.arange(len(target_M_G))
plt.bar(x,sum_along_G[np.abs(sum_along_G)>0], alpha=0.5)
plt.xticks(x, target_M_G, rotation=30, ha='right')
plt.ylabel('Average Fraction of Effort')
plt.title('Target of Policy Change Efforts (G)')
plt.savefig('%s_M_targets.svg'%(resource_user),bbox_inches = 'tight')
plt.show()

plt.figure()
x = np.arange(len(target_C))
plt.bar(x,C[np.abs(C)>0], alpha=0.5)
plt.xticks(x, target_C, rotation=30, ha='right')
plt.ylabel('Average Fraction of Effort')
plt.title('Target of Support or Undermining (C)')
plt.savefig('%s_C_targets.svg'%(resource_user),bbox_inches = 'tight')
plt.show()

# Average amount of effort put into each of the types of actions
effort_G = np.sum(np.abs(avg_strategy[0:R*M*N]))
effort_H = np.sum(np.abs(avg_strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M]))
effort_C = np.sum(np.abs(avg_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]))
effort_P = np.sum(np.abs(avg_strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot]))

# standard deviation across runs in amount of effort put into each type of action (averaged across strategy parameters for each type)
std_G = np.average(np.abs(stdev_strategy[0:R*M*N]))
std_H = np.average(np.abs(stdev_strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M]))
std_C = np.average(np.abs(stdev_strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot]))
std_P = np.average(np.abs(stdev_strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot]))
std_devs = [std_G, std_H, std_C, std_P]

strategy_options = ('Extraction Policy', 'Recharge Policy', 'Direct supporting/undermining', 'Assistance Policy')
y_pos = np.arange(len(strategy_options))
efforts = [effort_G, effort_H, effort_C, effort_P]

plt.bar(y_pos, efforts, yerr = std_devs, alpha=0.5)
plt.xticks(y_pos, strategy_options, rotation=30, ha='right')
plt.ylabel('Average Fraction of Effort')
plt.title('Strategy Types')
plt.savefig('%s_Strategy_types.svg'%(resource_user),bbox_inches = 'tight')
plt.show()