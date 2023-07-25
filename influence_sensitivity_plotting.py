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

def plot_comparison(parameterization, pos_x, pos_y, labels = N_list, num_points = N):
  filepath = Path('data\influences_sensitivities\%s\partial_impacts_%s_alt'
                                %(parameterization, parameterization))

  # filepath_sensitivity = Path('data\influences_sensitivities\%s\sensitivities_%s'
  #                               %(parameterization, parameterization))
  # filepath_influence = Path('data\influences_sensitivities\%s\influences_%s'
  #                             %(parameterization, parameterization))

  with open(filepath, 'rb') as f:
    impacts_list = pickle.load(f)
    
  impacts_mean = np.mean(impacts_list, axis=0)
  influences_avg = np.sum(impacts_mean,axis=1)[3:3+N-2]
  sensitivities_avg = np.sum(impacts_mean,axis=0)[3:3+N-2]
 
  # with open(filepath_sensitivity, 'rb') as f:
  #   sensitivities_list = pickle.load(f)  
  # with open(filepath_influence, 'rb') as f:
  #   influences_list = pickle.load(f)
  
  # Just get influences and sensitivities for resource users

  # check for non-zero imaginary part of sensitivity or influence values
  if np.any(abs(sensitivities_avg)-abs(np.real(sensitivities_avg)) > 1e-10):
    print('complex sensitivity values!')
    print(parameterization)
  else:
    sensitivities_avg = np.real(sensitivities_avg)
    
  if np.any(abs(influences_avg)-abs(np.real(influences_avg)) > 1e-10):
    print('complex influence values!')
    print(parameterization)
  else:
    influences_avg = np.real(influences_avg)
    
  
  for i in range(N-2):
    axs[pos_x, pos_y].annotate('', xy = (sensitivities_avg[i], influences_avg[i]), xytext =(sensitivities_avg_base[i], influences_avg_base[i]), arrowprops=dict(color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[i]], arrowstyle='-|>', mutation_scale=15))

  
# Use to compare all four scenarios to the base scenario

with open(Path('data\influences_sensitivities\\base\partial_impacts_base_alt'), 'rb') as f:
  partial_impacts_base = pickle.load(f)


impacts_base_mean = np.mean(partial_impacts_base, axis=0)
influences_avg_base = np.sum(impacts_base_mean,axis=1)[3:3+N-2]
sensitivities_avg_base = np.sum(impacts_base_mean,axis=0)[3:3+N-2]

# filepath_sensitivity = Path('data\influences_sensitivities\\base\sensitivities_base')
# filepath_influence = Path('data\influences_sensitivities\\base\influences_base')


# # import data from baseline, which will be plotted on all plots
# with open(filepath_sensitivity, 'rb') as f:
#   sensitivities_list_base = pickle.load(f)

# with open(filepath_influence, 'rb') as f:
#   influences_list_base = pickle.load(f)
  
# influences_avg_base = np.mean(influences_list_base, axis=0)[3:3+N]
# sensitivities_avg_base = np.mean(sensitivities_list_base, axis=0)[3:3+N]
  
#influences_avg_base = np.log(np.mean(influences_list_base, axis=0)[3:3+N])
#sensitivities_avg_base = np.log(np.mean(sensitivities_list_base, axis=0)[3:3+N])

# check for non-zero imaginary part of sensitivity or influence values
if np.any(abs(sensitivities_avg_base-np.real(sensitivities_avg_base)) > 1e-20):
  print('complex sensitivity values!')
else:
  sensitivities_avg_base = np.real(sensitivities_avg_base)
  
if np.any(abs(influences_avg_base-np.real(influences_avg_base)) > 1e-20):
  print('complex influence values!')
else:
  influences_avg_base = np.real(influences_avg_base)


sns.set_style("darkgrid")
fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
a = sns.scatterplot(ax = axs[0,0], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list[:-2], marker = 'o', s=50)
plot_comparison('v1', 0, 0)
axs[0,0].set_title('Ending "Regulatory Drought"')
b = sns.scatterplot(ax = axs[0,1], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list[:-2], marker = 'o', s=50)
plot_comparison('v2', 0, 1)
axs[0,1].set_title('Improved water management')
c = sns.scatterplot(ax = axs[1,0], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list[:-2], marker = 'o', s=50)
plot_comparison('v3', 1, 0)
axs[1,0].set_title('Increased state oversight')
#axs[1,0].set_title('test')
d = sns.scatterplot(ax = axs[1,1], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list[:-2], marker = 'o', s=50)
plot_comparison('v4', 1, 1)
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
plt.savefig('comparisons.svg', bbox_inches='tight')

#####################################################################################
# Plot partial sensitivities and influences

import pandas as pd
def partial_impacts_table(parameterization):

  with open(Path('data\influences_sensitivities\\%s\partial_impacts_%s_alt'%(parameterization,parameterization)), 'rb') as f:
    partial_impacts = pickle.load(f)


  impacts_mean = np.mean(partial_impacts, axis=0)
  
  R_list = ['surface water', 'groundwater', 'groundwater quality']
  
  entities = pd.read_excel('parameter_files\\%s\entity_list.xlsx'%(parameterization),sheet_name=None, header=None)
  N_list=entities['N'].values[:,0]
  N = len(N_list)

  K_list=entities['K'].values[:,0]
  K = len(K_list)

  M_list=entities['M'].values[:,0]
  M = len(M_list)

  all_list = np.concatenate((np.array(R_list), N_list, K_list, M_list))
  categories = ['resource users', 'non-governmental organizations', 'government entities']

  # rows are influences, columns are sensitivities
  df_summary = pd.DataFrame(index=N_list, columns=R_list + categories)

  df_all = pd.DataFrame(np.real(impacts_mean),index=all_list, columns=all_list)
  df_summary = df_all.loc[:'investor growers (white area)',:'investor growers (white area)']


  for resource_user in N_list[:-2]:
    # RU influence on orgs
    df_summary.loc[resource_user,'non-governmental organizations'] = np.real(np.sum(impacts_mean[all_list==resource_user,N+3:N+3+K]))
    # orgs influence on RU
    df_summary.loc['non-governmental organizations', resource_user] = np.real(np.sum(impacts_mean[N+3:N+3+K,all_list==resource_user]))
    # RU influence on govt
    df_summary.loc[resource_user,'government entities'] = np.real(np.sum(impacts_mean[all_list==resource_user,N+3+K:]))
    #govt influence on RU
    df_summary.loc['government entities',resource_user] = np.real(np.sum(impacts_mean[N+3+K:,all_list==resource_user]))

  # Fill in rest of table
  for resource in R_list:
    df_summary.loc[resource,'non-governmental organizations'] = np.real(np.sum(impacts_mean[all_list==resource,N+3:N+3+K]))
    df_summary.loc['non-governmental organizations', resource] = np.real(np.sum(impacts_mean[N+3:N+3+K,all_list==resource]))
    df_summary.loc[resource,'government entities'] = np.real(np.sum(impacts_mean[all_list==resource,N+3+K:]))
    #govt influence on resource
    df_summary.loc['government entities', resource] = np.real(np.sum(impacts_mean[N+3+K:,all_list==resource]))

  df_summary.loc['non-governmental organizations','non-governmental organizations'] = np.real(np.sum(impacts_mean[N+3:N+3+K,N+3:N+3+K]))
  df_summary.loc['non-governmental organizations','government entities'] = np.real(np.sum(impacts_mean[N+3:N+3+K,N+3+K:]))
  df_summary.loc['government entities','non-governmental organizations'] = np.real(np.sum(impacts_mean[N+3+K:,N+3:N+3+K]))
  df_summary.loc['government entities','government entities'] = np.real(np.sum(impacts_mean[N+3+K:,N+3+K:]))


  colors = plt.cm.BuPu(df_summary.to_numpy(dtype = float))

  #fig = plt.figure(figsize=(25, 10))
  fig = plt.figure()
  ax = fig.add_subplot(111, xticks=[], yticks=[])

  #create dummy colormap (not visible) for the colorbar
  img = plt.imshow(df_summary.to_numpy(dtype = float), cmap="BuPu")
  img.set_visible(False)
  plt.colorbar(orientation="horizontal", pad=0.1)

  import textwrap
  rows = []
  cols = []

  for i in range(len(df_summary.index)):
    rows.append(textwrap.fill(df_summary.index[i],25))
    cols.append(textwrap.fill(df_summary.columns[i],25))

    
  table=ax.table(rowLabels=rows, colLabels=cols, 
                      loc='center',cellLoc='left', cellColours=colors)

  table.auto_set_font_size(False)
  table.set_fontsize(10)
  ax.axis('off')

  table.auto_set_column_width((2,3,4,5,6,7,8,9))

  plt.tight_layout()
  plt.show()
  plt.savefig('partial_impacts_table_%s.svg'%(parameterization), bbox_inches='tight')

#partial_impacts_table('base')
#partial_impacts_table('v4')