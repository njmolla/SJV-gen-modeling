import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


entities = pd.read_excel('parameter_files\\base\entity_list.xlsx',sheet_name=None, header=None)
N_list=entities['N'].values[:,0]

K_list=entities['K'].values[:,0]

M_list=entities['M'].values[:,0]

R_list = ['surface water', 'groundwater', 'groundwater quality']

# N_list=['rural communities','investor growers']
# N = len(N_list)

# K_list=['EJ groups']
# K = len(K_list)

# M_list=['Water Rights Division (SWRCB)','Drinking Water Division (SWRCB)','County Board of Supervisors','Local Water Boards','CV SALTS management zones']

N = len(N_list)
K = len(K_list)
M = len(M_list)

R = 3

tot = N+K+M
resource_user = 'rural communities'
#resource_user = 'small growers'
#resource_user = 'small growers (white area)'
#resource_user = 'investor growers'
#resource_user = 'investor growers (white area)'

with open('data\strategies_%s'%(resource_user), 'rb') as f:
  strategies = pickle.load(f) 
  
with open('data\stabilities_%s'%(resource_user), 'rb') as f:
  stabilities = pickle.load(f)
  
all_zeros = np.all(stabilities==[0,0,0,0],axis=1)
stabilities = stabilities[~all_zeros] # filter out entries that have no stabilities recorded (were skipped)

all_stable = np.all(stabilities==[1,1,1,1],axis=1)
list = plt.plot(np.transpose(strategies[all_stable][:,abs(np.sum(strategies[all_stable],axis=0))>1e-3]),'.')

strategies = np.array(strategies[all_stable])
strategies_binary = np.zeros(np.shape(strategies))
strategies_binary[strategies < 0] = -1
strategies_binary[strategies > 0] = 1
strategies_binary[np.abs(strategies)<1e-2] = 0


unique_strategies, counts = np.unique(strategies_binary, axis=0, return_counts = True)

def translate_strategies(strategy):
  strategy[np.abs(strategy)<0.01] = 0 # do this again in case we're translating an individual strategy
  translation = [] # list of actions within each strategy
  G = strategy[0:R*M*N].reshape((R,M,N)) # affect extraction policy (for particular actor)
  H = strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M] # affect recharge policy
  C = strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot] # collaborate with/support other actors
  P = strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot].reshape((M,tot))
  
  for i in range(len(np.nonzero(G)[0])):
    if G[np.nonzero(G)][i] > 0:
      action = ['support %s effect on %s extraction of %s' %(M_list[np.nonzero(G)[1][i]],
          N_list[np.nonzero(G)[2][i]], R_list[np.nonzero(G)[0][i]])]
    else:
      action = ['oppose %s effect on %s extraction of %s' %(M_list[np.nonzero(G)[1][i]], 
          N_list[np.nonzero(G)[2][i]], R_list[np.nonzero(G)[0][i]])]
    translation += action
      
  for i in range(len(np.nonzero(H)[0])):
    if H[np.nonzero(H)][i] > 0:
      action = ['support %s effect on recharge' %(M_list[np.nonzero(H)[0][i]])]
    else:
      action = ['oppose %s effect on recharge' %(M_list[np.nonzero(H)[0][i]])]
    translation += action
    
  entity_list = np.concatenate((N_list, K_list, M_list))
  
  for i in range(len(np.nonzero(C)[0])):
    if C[np.nonzero(C)][i] > 0:
      action = ['support/collaborate with %s' %(entity_list[np.nonzero(C)[0][i]])]
    else:
      action = ['undermine %s' %(entity_list[np.nonzero(C)[0][i]])]
    translation += action
  
  for i in range(len(np.nonzero(P)[0])):
    if P[np.nonzero(P)][i] > 0:
      action = ['support %s support of %s' %(M_list[np.nonzero(P)[0][i]],entity_list[np.nonzero(P)[1][i]])]
    else:
      action = ['oppose %s support of %s' %(M_list[np.nonzero(P)[0][i]],entity_list[np.nonzero(P)[1][i]])]
    translation += action  
  return translation

def translate_strategies_shortver(strategy):
  #strategy[np.abs(strategy)<0.01] = 0 # do this again in case we're translating an individual strategy
  translation = [] # list of actions within each strategy
  G = strategy[0:R*M*N].reshape((R,M,N)) # affect extraction policy (for particular actor)
  H = strategy[R*M*N + R*N + 1:R*M*N + R*N + 1 + M] # affect recharge policy
  C = strategy[R*M*N + R*N + 1 + M:R*M*N + R*N + 1 + M + tot] # collaborate with/support other actors
  P = strategy[R*M*N + R*N + 1 + M + tot:R*M*N + R*N + 1 + M + tot + M*tot].reshape((M,tot))
  
  for i in range(len(np.nonzero(G)[0])):
    if G[np.nonzero(G)][i] > 0:
      action = ['%s effect on %s' %(M_list[np.nonzero(G)[1][i]],
          N_list[np.nonzero(G)[2][i]])]
    else:
      action = ['%s effect on %s' %(M_list[np.nonzero(G)[1][i]], 
          N_list[np.nonzero(G)[2][i]])]
    translation += action
      
  for i in range(len(np.nonzero(H)[0])):
    if H[np.nonzero(H)][i] > 0:
      action = ['%s effect on recharge' %(M_list[np.nonzero(H)[0][i]])]
    else:
      action = ['%s effect on recharge' %(M_list[np.nonzero(H)[0][i]])]
    translation += action
    
  entity_list = np.concatenate((N_list, K_list, M_list))
  
  for i in range(len(np.nonzero(C)[0])):
    if C[np.nonzero(C)][i] > 0:
      action = ['support/collaborate with %s' %(entity_list[np.nonzero(C)[0][i]])]
    else:
      action = ['undermine %s' %(entity_list[np.nonzero(C)[0][i]])]
    translation += action
  
  for i in range(len(np.nonzero(P)[0])):
    if P[np.nonzero(P)][i] > 0:
      action = ['%s support of %s' %(M_list[np.nonzero(P)[0][i]],entity_list[np.nonzero(P)[1][i]])]
    else:
      action = ['%s support of %s' %(M_list[np.nonzero(P)[0][i]],entity_list[np.nonzero(P)[1][i]])]
    translation += action  
  return translation
  
strategies_translated = [] # list of strategies across ensemble
 
# for strategy in unique_strategies:
  # translation = translate_strategies_shortver(strategy)
  # strategies_translated += [translation]
  
strategies_translated = translate_strategies_shortver(np.sum(strategies_binary, axis=0)/len(strategies))
# strategies_translated = np.concatenate(strategies_translated)
strategies_translated, indices = np.unique(strategies_translated, return_index = True)
counts = np.sum(strategies_binary, axis=0)/len(strategies)
counts_nonzero = counts[abs(counts)>0]
counts_nonzero = counts_nonzero[indices]
strategies_translated = np.array(strategies_translated)[abs(counts_nonzero)>0.25]
counts = counts_nonzero[abs(counts_nonzero)>0.25]

# x = np.arange(len(strategies_translated))
# plt.figure()
# plt.figure(figsize=(4, 12))
# #plt.barh(x, counts, alpha=0.5)
# plt.barh(x, counts, alpha=0.5, height = 1)
# plt.yticks(ticks = x, labels = strategies_translated) #, rotation=30, ha='right')
# #axs.set_xticklabels(strategies_translated, rotation=30, ha='right')
# plt.xlabel('Proportion of Runs', fontsize = 12)
# plt.title('Strategies', fontsize = 16)
# plt.savefig('%s_opt_strategy_translated.svg'%(resource_user),bbox_inches = 'tight')

## plotting for rural communities' strategy (more complicated than the others) ##

# aggregate actions
wq_strategies_indices = np.concatenate((np.arange(0,6), np.arange(9,16), np.array([17])))
strategies_translated = np.delete(strategies_translated, wq_strategies_indices)
wq_strategies_counts = max(counts[wq_strategies_indices])
water_rights_counts = min(counts[[15,17]])
counts = np.delete(counts, wq_strategies_indices)
strategies_translated = np.append(strategies_translated, 'Water quality regulations effect on growers')
counts = np.append(counts, wq_strategies_counts)
strategies_translated = np.append(strategies_translated, 'Water Rights effect on growers')
counts = np.append(counts, water_rights_counts)

x = np.arange(len(strategies_translated))
#x = np.linspace(0,55,len(strategies_translated))
plt.figure(figsize=(6, 12))
plt.barh(x, counts, alpha=0.5)
plt.yticks(ticks = x, labels = strategies_translated) #, rotation=30, ha='right')
plt.xlabel('Proportion of Runs', fontsize = 12)
plt.title('Strategies', fontsize = 16)
plt.savefig('%s_opt_strategy_translated.pdf'%(resource_user),bbox_inches = 'tight')


# def index_to_action(index):
  # if index < R*M*N:
    # resource = index%3
  