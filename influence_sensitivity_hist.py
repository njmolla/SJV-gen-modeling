import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path
from scipy import stats

# import data from baseline, which will be plotted on all plots
with open('sensitivities_base', 'rb') as f:
  sensitivities_list_base = pickle.load(f)

with open('influences_base', 'rb') as f:
  influences_list_base = pickle.load(f)
  
#resource_user_key = 'rural communities'
resource_user_key = 'small growers'
print(resource_user_key)
#resource_user_key = 'investor growers'
#resource_user_key = 'investor growers (white area)'
path = Path.cwd().joinpath('parameter_files', 'base', 'entity_list.xlsx')
entities_list = pd.read_excel(path,sheet_name=None, header=None)
N_list=entities_list['N'].values[:,0]
RU = np.nonzero(N_list==resource_user_key)[0][0] + 3

sensitivities_list_base = np.array(sensitivities_list_base)[:,RU]
influences_list_base = np.array(influences_list_base)[:,RU]

with open('sensitivities_v1', 'rb') as f:
  sensitivities_list_v1 = pickle.load(f)

with open('influences_v1', 'rb') as f:
  influences_list_v1 = pickle.load(f)

sensitivities_list_v1 = np.array(sensitivities_list_v1)[:,RU]
influences_list_v1 = np.array(influences_list_v1)[:,RU]

stats.ttest_ind(sensitivities_list_base, sensitivities_list_v1, equal_var=False)
stats.ttest_ind(influences_list_base, influences_list_v1, equal_var=False)


with open('sensitivities_v2', 'rb') as f:
  sensitivities_list_v2 = pickle.load(f)

with open('influences_v2', 'rb') as f:
  influences_list_v2 = pickle.load(f)
  
sensitivities_list_v2 = np.array(sensitivities_list_v2)[:,RU]
influences_list_v2 = np.array(influences_list_v2)[:,RU]

stats.ttest_ind(sensitivities_list_base, sensitivities_list_v2, equal_var=False)
stats.ttest_ind(influences_list_base, influences_list_v2, equal_var=False)

with open('sensitivities_v3', 'rb') as f:
  sensitivities_list_v3 = pickle.load(f)

with open('influences_v3', 'rb') as f:
  influences_list_v3 = pickle.load(f)
  
sensitivities_list_v3 = np.array(sensitivities_list_v3)[:,RU]
influences_list_v3 = np.array(influences_list_v3)[:,RU]

stats.ttest_ind(sensitivities_list_base, sensitivities_list_v3, equal_var=False)
stats.ttest_ind(influences_list_base, influences_list_v3, equal_var=False)

with open('sensitivities_v4', 'rb') as f:
  sensitivities_list_v4 = pickle.load(f)

with open('influences_v4', 'rb') as f:
  influences_list_v4 = pickle.load(f)
  
sensitivities_list_v4 = np.array(sensitivities_list_v4)[:,RU]
influences_list_v4 = np.array(influences_list_v4)[:,RU]

stats.ttest_ind(sensitivities_list_base, sensitivities_list_v4, equal_var=False)
stats.ttest_ind(influences_list_base, influences_list_v4, equal_var=False)


fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
bins = np.linspace(0, 10, 40)
axs[0,0].hist(sensitivities_list_base, bins, alpha=0.5, label='base')
axs[0,0].hist(sensitivities_list_v1, bins, alpha=0.5)
axs[0,0].set_title('Regulatory Burden')

axs[0,1].hist(sensitivities_list_base, bins, alpha=0.5, label='base')
axs[0,1].hist(sensitivities_list_v2, bins, alpha=0.5)
axs[0,1].set_title('Physical Availability/Quality of Water')

axs[1,0].hist(sensitivities_list_base, bins, alpha=0.5, label='base')
axs[1,0].hist(sensitivities_list_v3, bins, alpha=0.5)
axs[1,0].set_title('Lack of State Oversight/Regulation')

axs[1,1].hist(sensitivities_list_base, bins, alpha=0.5, label='base')
axs[1,1].hist(sensitivities_list_v4, bins, alpha=0.5)
axs[1,1].set_title('Exploitative Nature of Agriculture')

plt.savefig('%s_sensitivities_dists.svg'%(resource_user_key), bbox_inches='tight')

fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
bins = np.linspace(0, 10, 20)
axs[0,0].hist(influences_list_base, bins, alpha=0.5, label='base')
axs[0,0].hist(influences_list_v1, bins, alpha=0.5)
axs[0,0].set_title('Regulatory Burden')

axs[0,1].hist(influences_list_base, bins, alpha=0.5, label='base')
axs[0,1].hist(influences_list_v2, bins, alpha=0.5)
axs[0,1].set_title('Physical Availability/Quality of Water')

axs[1,0].hist(influences_list_base, bins, alpha=0.5, label='base')
axs[1,0].hist(influences_list_v3, bins, alpha=0.5)
axs[1,0].set_title('Lack of State Oversight/Regulation')

axs[1,1].hist(influences_list_base, bins, alpha=0.5, label='base')
axs[1,1].hist(influences_list_v4, bins, alpha=0.5)
axs[1,1].set_title('Exploitative Nature of Agriculture')

plt.savefig('%s_influences_dists.svg'%(resource_user_key), bbox_inches='tight')

