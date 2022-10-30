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

def plot_comparison(parameterization, pos_x, pos_y, labels = N_list, num_points = N):
  
  with open('sensitivities_%s'%(parameterization), 'rb') as f:
    sensitivities_list = pickle.load(f)

  with open('influences_%s'%(parameterization), 'rb') as f:
    influences_list = pickle.load(f)
  
  RU = 7
  influences_avg = np.mean(influences_list, axis=0)[RU]#[3:3+N]
  sensitivities_avg = np.mean(sensitivities_list, axis=0)[RU]#[3:3+N]
  sensitivities_list = np.array(sensitivities_list)[:,RU]
  influences_list = np.array(influences_list)[:,RU]

  # check for non-zero imaginary part of sensitivity or influence values
  if np.any(abs(sensitivities_avg)-abs(np.real(sensitivities_avg)) > 1e-20):
    print('complex sensitivity values!')
  else:
    sensitivities_avg = np.real(sensitivities_avg)
    
  if np.any(abs(influences_avg)-abs(np.real(influences_avg)) > 1e-20):
    print('complex influence values!')
  else:
    influences_avg = np.real(influences_avg)
  
  sns.scatterplot(ax = axs[pos_x, pos_y], x = sensitivities_list, y = influences_list, marker = 'o', s=25, alpha = 0.4)
  sns.scatterplot(ax = axs[pos_x, pos_y], x = sensitivities_avg, y = influences_avg, hue = N_list, marker = 'x', s = 100)
  
  # for i in range(N):
    # axs[pos_x, pos_y].annotate('', xy = (sensitivities_avg[i], influences_avg[i]), xytext =(sensitivities_avg_base[i], influences_avg_base[i]), arrowprops=dict(color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS)[i]], arrowstyle='-|>', mutation_scale=20))
 
############################################################################# 

# # import data from baseline, which will be plotted on all plots
# with open('sensitivities_base', 'rb') as f:
  # sensitivities_list_base = pickle.load(f)

# with open('influences_base', 'rb') as f:
  # influences_list_base = pickle.load(f)
  
# influences_avg_base = np.mean(influences_list_base, axis=0)[3:3+N]
# sensitivities_avg_base = np.mean(sensitivities_list_base, axis=0)[3:3+N]

# # check for non-zero imaginary part of sensitivity or influence values
# if np.any(abs(sensitivities_avg_base)-abs(np.real(sensitivities_avg_base)) > 1e-20):
  # print('complex sensitivity values!')
# else:
  # sensitivities_avg_base = np.real(sensitivities_avg_base)
  
# if np.any(abs(influences_avg_base)-abs(np.real(influences_avg_base)) > 1e-20):
  # print('complex influence values!')
# else:
  # influences_avg_base = np.real(influences_avg_base)
  

# sns.set_style("darkgrid")
# fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
# a = sns.scatterplot(ax = axs[0,0], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
# plot_comparison('v1', 0, 0)
# axs[0,0].set_title('Regulatory Burden')
# b = sns.scatterplot(ax = axs[0,1], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
# plot_comparison('v2', 0, 1)
# axs[0,1].set_title('Physical Availability/Quality of Water')
# c = sns.scatterplot(ax = axs[1,0], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
# plot_comparison('v3', 1, 0)
# axs[1,0].set_title('Lack of State Oversight/Regulation')
# d = sns.scatterplot(ax = axs[1,1], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
# plot_comparison('test', 1, 1)
# axs[1,1].set_title('Exploitative Nature of Agriculture')



# axs[1,0].set_xlabel('Sensitivity', fontsize = 18)
# axs[1,1].set_xlabel('Sensitivity', fontsize = 18)
# axs[0,0].set_ylabel('Influence', fontsize = 18)
# axs[1,0].set_ylabel('Influence', fontsize = 18)

# #plt.ylabel('Influence', fontsize = 18)
# axs[0,0].legend([],[], frameon=False)
# axs[1,0].legend([],[], frameon=False)
# axs[1,1].legend([],[], frameon=False)
# handles, labels = axs[0,1].get_legend_handles_labels()
# axs[0,1].legend(handles[:N], labels[:N],bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# plt.savefig('comparisons_test.svg', bbox_inches='tight')

##########################################################################################################
# import data from baseline, which will be plotted on all plots
with open('sensitivities_base', 'rb') as f:
  sensitivities_list_base = pickle.load(f)

with open('influences_base', 'rb') as f:
  influences_list_base = pickle.load(f)
  
RU = 7
  
influences_avg_base = np.mean(influences_list_base, axis=0)[RU]#[3:3+N]
sensitivities_avg_base = np.mean(sensitivities_list_base, axis=0)[RU]#[3:3+N]

sensitivities_list_base = np.array(sensitivities_list_base)[:,RU]
influences_list_base = np.array(influences_list_base)[:,RU]

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
a_2 = sns.scatterplot(ax = axs[0,0], x = sensitivities_list_base, y = influences_list_base, marker = 'o', s=25, alpha = 0.4)
a = sns.scatterplot(ax = axs[0,0], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
plot_comparison('v1', 0, 0)
axs[0,0].set_title('Regulatory Burden')
b_2 = sns.scatterplot(ax = axs[0,1], x = sensitivities_list_base, y = influences_list_base, marker = 'o', s=25, alpha = 0.4)
b = sns.scatterplot(ax = axs[0,1], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
plot_comparison('v2', 0, 1)
axs[0,1].set_title('Physical Availability/Quality of Water')
c_2 = sns.scatterplot(ax = axs[1,0], x = sensitivities_list_base, y = influences_list_base, marker = 'o', s=25, alpha = 0.4)
c = sns.scatterplot(ax = axs[1,0], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
plot_comparison('v3', 1, 0)
axs[1,0].set_title('Lack of State Oversight/Regulation')
d_2 = sns.scatterplot(ax = axs[1,1], x = sensitivities_list_base, y = influences_list_base, marker = 'o', s=25, alpha = 0.4)
d = sns.scatterplot(ax = axs[1,1], x = sensitivities_avg_base, y = influences_avg_base, hue = N_list, marker = 'o', s=50)
plot_comparison('test', 1, 1)
axs[1,1].set_title('Exploitative Nature of Agriculture')



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
plt.savefig('comparisons_test.svg', bbox_inches='tight')
