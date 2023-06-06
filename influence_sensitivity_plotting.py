import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path

entities = pd.read_excel('parameter_files\\base\entity_list.xlsx',sheet_name=None, header=None)
N_list=entities['N'].values[:,0]
N = len(N_list)

K_list=entities['K'].values[:,0]
K = len(K_list)

M_list=entities['M'].values[:,0]
M = len(M_list)

def plot_comparison(parameterization, pos_x, pos_y, labels = N_list, num_points = N, effect_type = None):
  if effect_type == None:
    filepath_sensitivity = Path('data\influences_sensitivities\%s\sensitivities_%s'
                                %(parameterization, parameterization))
    filepath_influence = Path('data\influences_sensitivities\%s\influences_%s'
                              %(parameterization, parameterization))
  else: 
    filepath_sensitivity = Path('data\influences_sensitivities\%s\sensitivities_%s_%s'
                                %(parameterization, parameterization, effect_type))
    filepath_influence = Path('data\influences_sensitivities\%s\influences_%s_%s'
                              %(parameterization, parameterization, effect_type))
  with open(filepath_sensitivity, 'rb') as f:
    sensitivities_list = pickle.load(f)  
  with open(filepath_influence, 'rb') as f:
    influences_list = pickle.load(f)
  
  # Just get influences and sensitivities for resource users, and average 
  # across trials
  influences_avg = np.mean(influences_list, axis=0)[3:3+N]
  sensitivities_avg = np.mean(sensitivities_list, axis=0)[3:3+N]

  # check for non-zero imaginary part of sensitivity or influence values
  if np.any(abs(sensitivities_avg)-abs(np.real(sensitivities_avg)) > 1e-20):
    print('complex sensitivity values!')
  else:
    sensitivities_avg = np.real(sensitivities_avg)
    
  if np.any(abs(influences_avg)-abs(np.real(influences_avg)) > 1e-20):
    print('complex influence values!')
  else:
    influences_avg = np.real(influences_avg)
  
  for i in range(N):
    axs[pos_x, pos_y].annotate('', xy = (sensitivities_avg[i], influences_avg[i]), xytext =(sensitivities_avg_base[i], influences_avg_base[i]), arrowprops=dict(color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[i]], arrowstyle='-|>', mutation_scale=15))
 
# Use to compare all four scenarios to the base scenario
comparison_type = None

if comparison_type == None:
  filepath_sensitivity = Path('data\influences_sensitivities\\base\sensitivities_base')
  filepath_influence = Path('data\influences_sensitivities\\base\influences_base')
else: 
  filepath_sensitivity = Path('data\influences_sensitivities\\base\sensitivities_base_%s'%(comparison_type))
  filepath_influence = Path('data\influences_sensitivities\\base\influences_base_%s'%(comparison_type))

# import data from baseline, which will be plotted on all plots
with open(filepath_sensitivity, 'rb') as f:
  sensitivities_list_base = pickle.load(f)

filepath = Path('data\influences_sensitivities\\base\influences_base_%s'%(comparison_type))
with open(filepath_influence, 'rb') as f:
  influences_list_base = pickle.load(f)
  
influences_avg_base = np.mean(influences_list_base, axis=0)[3:3+N]
sensitivities_avg_base = np.mean(sensitivities_list_base, axis=0)[3:3+N]

# check for non-zero imaginary part of sensitivity or influence values
if np.any(abs(sensitivities_avg_base)-abs(np.real(sensitivities_avg_base)) > 1e-20):
  print('complex sensitivity values!')
else:
  sensitivities_avg_base = np.real(sensitivities_avg_base)
  
if np.any(abs(influences_avg_base)-abs(np.real(influences_avg_base)) > 1e-20):
  print('complex influence values!')
else:
  influences_avg_base = np.real(influences_avg_base)
  

sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
a = sns.scatterplot(ax = axs[0,0], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
plot_comparison('v1', 0, 0, effect_type = comparison_type)
axs[0,0].set_title('Ending Regulatory Drought')
b = sns.scatterplot(ax = axs[0,1], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
plot_comparison('v2', 0, 1, effect_type = comparison_type)
axs[0,1].set_title('Improved water management')
c = sns.scatterplot(ax = axs[1,0], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
plot_comparison('v3', 1, 0, effect_type = comparison_type)
axs[1,0].set_title('Increased state oversight')
#axs[1,0].set_title('test')
d = sns.scatterplot(ax = axs[1,1], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
plot_comparison('v4', 1, 1, effect_type = comparison_type)
axs[1,1].set_title('Change Nature of Agriculture')



axs[1,0].set_xlabel('Sensitivity', fontsize = 18)
axs[1,1].set_xlabel('Sensitivity', fontsize = 18)
axs[0,0].set_ylabel('Influence', fontsize = 18)
axs[1,0].set_ylabel('Influence', fontsize = 18)

#plt.ylabel('Influence', fontsize = 18)
axs[0,0].legend([],[], frameon=False)
axs[1,0].legend([],[], frameon=False)
axs[1,1].legend([],[], frameon=False)
handles, labels = axs[0,1].get_legend_handles_labels()
axs[0,1].legend(handles[:N], labels[:N],bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.savefig('comparisons_%s.svg'%(comparison_type), bbox_inches='tight')

#####################################################################################
# Plot partial sensitivities and influences for base scenario


with open(Path('data\influences_sensitivities\\base\partial_impacts_base'), 'rb') as f:
  partial_impacts_base = pickle.load(f)


impacts_base_mean = np.mean(partial_impacts_base, axis=0)
import pandas as pd

entities = pd.read_excel('parameter_files\\base\entity_list.xlsx',sheet_name=None, header=None)

R_list = ['surface water', 'groundwater', 'groundwater quality']

all_list = np.concatenate((np.array(R_list), N_list, K_list, M_list))

categories = ['resource users', 'non-governmental organizations', 'government entities']

df = pd.DataFrame(index=N_list, columns=R_list + categories)

for resource_user in N_list:
  df.loc[resource_user]['surface water'] = np.real(impacts_base_mean[all_list==resource_user] [:,all_list=='surface water'])
  df.loc[resource_user]['groundwater'] = np.real(impacts_base_mean[all_list==resource_user] [:,all_list=='groundwater'])
  df.loc[resource_user]['groundwater quality'] = np.real(impacts_base_mean[all_list==resource_user] [:,all_list=='groundwater quality'])
  df.loc[resource_user]['resource users'] = np.real(np.sum(impacts_base_mean[all_list==resource_user][:,3:N+3]))
  df.loc[resource_user]['non-governmental organizations'] = np.real(np.sum(impacts_base_mean[all_list==resource_user][:,N+3:N+3+K]))
  df.loc[resource_user]['government entities'] = np.real(np.sum(impacts_base_mean[all_list==resource_user][:,N+3+K:]))

colors = plt.cm.viridis(df.to_numpy(dtype = float))

fig = plt.figure()
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])

table=plt.table(rowLabels=df.index, colLabels=df.columns, 
                    loc='center', cellColours=colors)

table.auto_set_font_size(False)
table.set_fontsize(10)
plt.show()

