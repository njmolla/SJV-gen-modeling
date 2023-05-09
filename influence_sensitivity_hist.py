import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path
from scipy import stats

path = Path.cwd().joinpath('parameter_files', 'base', 'entity_list.xlsx')
entities_list = pd.read_excel(path,sheet_name=None, header=None)
N_list=entities_list['N'].values[:,0]

# set up data frames for storing results of T-tests
sensitivities_df = pd.DataFrame(index = N_list, columns = ['v1','v2','v3','v4'])
influences_df = pd.DataFrame(index = N_list, columns = ['v1','v2','v3','v4'])

# import data from baseline, which all other scenarios will be compared to
filepath = Path('data\influences_sensitivities\sensitivities_base')
with open(filepath, 'rb') as f:
  sensitivities_list_base = pickle.load(f)

filepath = Path('data\influences_sensitivities\influences_base')
with open(filepath, 'rb') as f:
  influences_list_base = pickle.load(f)
  
filepath = Path('data\influences_sensitivities\sensitivities_v1')
with open(filepath, 'rb') as f:
  sensitivities_list_v1 = pickle.load(f)

filepath = Path('data\influences_sensitivities\influences_v1')
with open(filepath, 'rb') as f:
  influences_list_v1 = pickle.load(f)

filepath = Path('data\influences_sensitivities\sensitivities_v2')
with open(filepath, 'rb') as f:
  sensitivities_list_v2 = pickle.load(f)

filepath = Path('data\influences_sensitivities\influences_v2')
with open(filepath, 'rb') as f:
  influences_list_v2 = pickle.load(f)
  
filepath = Path('data\influences_sensitivities\sensitivities_v3')
with open(filepath, 'rb') as f:
  sensitivities_list_v3 = pickle.load(f)

filepath = Path('data\influences_sensitivities\influences_v3')
with open(filepath, 'rb') as f:
  influences_list_v3 = pickle.load(f)

filepath = Path('data\influences_sensitivities\sensitivities_v4')
with open(filepath, 'rb') as f:
  sensitivities_list_v4 = pickle.load(f)

filepath = Path('data\influences_sensitivities\influences_v4')
with open(filepath, 'rb') as f:
  influences_list_v4 = pickle.load(f)

    
for resource_user_key in N_list:
  print(resource_user_key)
  RU = np.nonzero(N_list==resource_user_key)[0][0] + 3
  
  sensitivities_base = np.array(sensitivities_list_base)[:,RU]
  influences_base = np.array(influences_list_base)[:,RU]

  sensitivities_v1 = np.array(sensitivities_list_v1)[:,RU]
  influences_v1 = np.array(influences_list_v1)[:,RU]

  stat, p = stats.ttest_ind(sensitivities_base, sensitivities_v1, equal_var=False)
  sensitivities_df.loc[resource_user_key,'v1'] = p
  stat, p = stats.ttest_ind(influences_base, influences_v1, equal_var=False)
  influences_df.loc[resource_user_key,'v1'] = p
   
  sensitivities_v2 = np.array(sensitivities_list_v2)[:,RU]
  influences_v2 = np.array(influences_list_v2)[:,RU]

  stat, p = stats.ttest_ind(sensitivities_base, sensitivities_v2, equal_var=False)
  sensitivities_df.loc[resource_user_key,'v2'] = p
  stat, p = stats.ttest_ind(influences_base, influences_v2, equal_var=False)
  influences_df.loc[resource_user_key,'v2'] = p
    
  sensitivities_v3 = np.array(sensitivities_list_v3)[:,RU]
  influences_v3 = np.array(influences_list_v3)[:,RU]

  stat, p = stats.ttest_ind(sensitivities_base, sensitivities_v3, equal_var=False)
  sensitivities_df.loc[resource_user_key,'v3'] = p
  stat, p = stats.ttest_ind(influences_base, influences_v3, equal_var=False)
  influences_df.loc[resource_user_key,'v3'] = p
    
  sensitivities_v4 = np.array(sensitivities_list_v4)[:,RU]
  influences_v4 = np.array(influences_list_v4)[:,RU]

  stat, p = stats.ttest_ind(sensitivities_base, sensitivities_v4, equal_var=False)
  sensitivities_df.loc[resource_user_key,'v4'] = p
  stat, p = stats.ttest_ind(influences_base, influences_v4, equal_var=False)
  influences_df.loc[resource_user_key,'v4'] = p
  
  fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
  bins = np.linspace(0, 10, 30)
  axs[0,0].hist(sensitivities_base, bins, alpha=0.5, label='base')
  axs[0,0].hist(sensitivities_v1, bins, alpha=0.5)
  axs[0,0].set_title('Ending Regulatory Drought')

  axs[0,1].hist(sensitivities_base, bins, alpha=0.5, label='base')
  axs[0,1].hist(sensitivities_v2, bins, alpha=0.5)
  axs[0,1].set_title('Improved water management')

  axs[1,0].hist(sensitivities_base, bins, alpha=0.5, label='base')
  axs[1,0].hist(sensitivities_v3, bins, alpha=0.5)
  axs[1,0].set_title('Increased state oversight')

  axs[1,1].hist(sensitivities_base, bins, alpha=0.5, label='base')
  axs[1,1].hist(sensitivities_v4, bins, alpha=0.5)
  axs[1,1].set_title('Change Nature of Agriculture')

  plt.savefig('%s_sensitivities_dists.pdf'%(resource_user_key), bbox_inches='tight')

  fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
  bins = np.linspace(0, 10, 30)
  axs[0,0].hist(influences_base, bins, alpha=0.5, label='base')
  axs[0,0].hist(influences_v1, bins, alpha=0.5)
  axs[0,0].set_title('Ending Regulatory Drought')

  axs[0,1].hist(influences_base, bins, alpha=0.5, label='base')
  axs[0,1].hist(influences_v2, bins, alpha=0.5)
  axs[0,1].set_title('Improved water management')

  axs[1,0].hist(influences_base, bins, alpha=0.5, label='base')
  axs[1,0].hist(influences_v3, bins, alpha=0.5)
  axs[1,0].set_title('Increased state oversight')

  axs[1,1].hist(influences_base, bins, alpha=0.5, label='base')
  axs[1,1].hist(influences_v4, bins, alpha=0.5)
  axs[1,1].set_title('Change Nature of Agriculture')

  plt.savefig('%s_influences_dists.pdf'%(resource_user_key), bbox_inches='tight')

filepath = Path('data/sensitivities_t-test.csv')
sensitivities_df.to_csv(filepath)

filepath = Path('data/influences_t-test.csv')
influences_df.to_csv(filepath)



