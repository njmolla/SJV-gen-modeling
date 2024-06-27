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
N = len(N_list)

def influence_sensitivity_lists(parameterization):
  filepath = Path('data\influences_sensitivities\%s\partial_impacts_%s_alt'
                                %(parameterization, parameterization))

  # filepath_sensitivity = Path('data\influences_sensitivities\%s\sensitivities_%s'
  #                               %(parameterization, parameterization))
  # filepath_influence = Path('data\influences_sensitivities\%s\influences_%s'
  #                             %(parameterization, parameterization))

  with open(filepath, 'rb') as f:
    impacts_list = pickle.load(f)
    
  impacts_array = np.array(impacts_list)
  # Get influences and sensitivities for resource users
  influences = np.sum(impacts_array, axis=2)[:,3:3+N-2]
  sensitivities = np.sum(impacts_array,axis=1)[:,3:3+N-2]
 
  # check for non-zero imaginary part of sensitivity or influence values
  if np.any(abs(sensitivities)-abs(np.real(sensitivities)) > 1e-10):
    print('complex sensitivity values!')
    print(parameterization)
  else:
    sensitivities = np.real(sensitivities)
    
  if np.any(abs(influences)-abs(np.real(influences)) > 1e-10):
    print('complex influence values!')
    print(parameterization)
  else:
    influences = np.real(influences)
    
    
  return influences, sensitivities


# set up data frames for storing results of T-tests
sensitivities_df = pd.DataFrame(index = N_list[:-2], columns = ['v1','v2','v3','v4'])
influences_df = pd.DataFrame(index = N_list[:-2], columns = ['v1','v2','v3','v4'])

influences_base, sensitivities_base = influence_sensitivity_lists('base')
influences_v1, sensitivities_v1 = influence_sensitivity_lists('v1')
influences_v2, sensitivities_v2 = influence_sensitivity_lists('v2')
influences_v3, sensitivities_v3 = influence_sensitivity_lists('v3')
influences_v4, sensitivities_v4 = influence_sensitivity_lists('v4')
    
for i in range(len(N_list[:-2])):
  resource_user_key = N_list[i]
  stat, p = stats.ttest_ind(sensitivities_base[:,i], sensitivities_v1[:,i], equal_var=False)
  sensitivities_df.loc[resource_user_key,'v1'] = p
  stat, p = stats.ttest_ind(influences_base[:,i], influences_v1[:,i], equal_var=False)
  influences_df.loc[resource_user_key,'v1'] = p
  

  stat, p = stats.ttest_ind(sensitivities_base[:,i], sensitivities_v2[:,i], equal_var=False)
  sensitivities_df.loc[resource_user_key,'v2'] = p
  stat, p = stats.ttest_ind(influences_base[:,i], influences_v2[:,i], equal_var=False)
  influences_df.loc[resource_user_key,'v2'] = p
    

  stat, p = stats.ttest_ind(sensitivities_base[:,i], sensitivities_v3[:,i], equal_var=False)
  sensitivities_df.loc[resource_user_key,'v3'] = p
  stat, p = stats.ttest_ind(influences_base[:,i], influences_v3[:,i], equal_var=False)
  influences_df.loc[resource_user_key,'v3'] = p
    

  stat, p = stats.ttest_ind(sensitivities_base[:,i], sensitivities_v4[:,i], equal_var=False)
  sensitivities_df.loc[resource_user_key,'v4'] = p
  stat, p = stats.ttest_ind(influences_base[:,i], influences_v4[:,i], equal_var=False)
  influences_df.loc[resource_user_key,'v4'] = p
  
  # fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
  # bins = np.linspace(0, 10, 30)
  # axs[0,0].hist(sensitivities_base, bins, alpha=0.5, label='base')
  # axs[0,0].hist(sensitivities_v1, bins, alpha=0.5)
  # axs[0,0].set_title('Ending Regulatory Drought')

  # axs[0,1].hist(sensitivities_base, bins, alpha=0.5, label='base')
  # axs[0,1].hist(sensitivities_v2, bins, alpha=0.5)
  # axs[0,1].set_title('Improved water management')

  # axs[1,0].hist(sensitivities_base, bins, alpha=0.5, label='base')
  # axs[1,0].hist(sensitivities_v3, bins, alpha=0.5)
  # axs[1,0].set_title('Increased state oversight')

  # axs[1,1].hist(sensitivities_base, bins, alpha=0.5, label='base')
  # axs[1,1].hist(sensitivities_v4, bins, alpha=0.5)
  # axs[1,1].set_title('Change Nature of Agriculture')

  # plt.savefig('%s_sensitivities_dists.pdf'%(resource_user_key), bbox_inches='tight')

  # fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
  # bins = np.linspace(0, 10, 30)
  # axs[0,0].hist(influences_base, bins, alpha=0.5, label='base')
  # axs[0,0].hist(influences_v1, bins, alpha=0.5)
  # axs[0,0].set_title('Ending Regulatory Drought')

  # axs[0,1].hist(influences_base, bins, alpha=0.5, label='base')
  # axs[0,1].hist(influences_v2, bins, alpha=0.5)
  # axs[0,1].set_title('Improved water management')

  # axs[1,0].hist(influences_base, bins, alpha=0.5, label='base')
  # axs[1,0].hist(influences_v3, bins, alpha=0.5)
  # axs[1,0].set_title('Increased state oversight')

  # axs[1,1].hist(influences_base, bins, alpha=0.5, label='base')
  # axs[1,1].hist(influences_v4, bins, alpha=0.5)
  # axs[1,1].set_title('Change Nature of Agriculture')

  # plt.savefig('%s_influences_dists.pdf'%(resource_user_key), bbox_inches='tight')

filepath = Path('data/sensitivities_t-test.csv')
sensitivities_df.to_csv(filepath)

filepath = Path('data/influences_t-test.csv')
influences_df.to_csv(filepath)



