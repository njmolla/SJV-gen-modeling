import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# entities = pd.read_excel('parameter_files\\base\entity_list.xlsx',sheet_name=None, header=None)
# N_list=entities['N'].values[:,0]

# K_list=entities['K'].values[:,0]

# M_list=entities['M'].values[:,0]

R_list = ['surface water', 'groundwater', 'groundwater quality']

N_list=['rural communities','investor growers']
N = len(N_list)

K_list=[]
K = len(K_list)

M_list=['Water Rights Division (SWRCB)','Drinking Water Division (SWRCB)']

N = len(N_list)
K = len(K_list)
M = len(M_list)

R = 3

tot = N+K+M
resource_user = 'rural communities'
#resource_user = 'small growers'
#resource_user = 'investor growers'
#resource_user = 'investor growers (white area)'

with open('strategies_%s_looping_mini'%(resource_user), 'rb') as f:
  strategies = pickle.load(f) 

strategies = np.array(strategies)
strategies_binary = np.zeros(np.shape(strategies))
strategies_binary[strategies < 0] = -1
strategies_binary[strategies > 0] = 1
strategies_binary[np.abs(strategies)<1e-3] = 0


unique_strategies, counts = np.unique(strategies_binary, axis=0, return_counts = True)

def translate_strategies(strategy):
  #strategy[np.abs(strategy)<0.01] = 0 # do this again in case we're translating an individual strategy
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

strategies_translated = [] # list of strategies across ensemble
 
for strategy in unique_strategies:
  translation = translate_strategies(strategy)
  strategies_translated += [translation]

print(np.array(strategies_translated)[counts>=3])

# def index_to_action(index):
  # if index < R*M*N:
    # resource = index%3
  