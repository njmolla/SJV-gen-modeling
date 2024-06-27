import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

entities = pd.read_excel('parameter_files\\base\entity_list.xlsx',sheet_name=None, header=None)
N_list=entities['N'].values[:,0]
N = len(N_list)

def compute_influence_sensitivity_breakdown(parameterization):
  filepath = Path('data\influences_sensitivities\%s\partial_impacts_%s_alt'
                                %(parameterization, parameterization))

  with open(filepath, 'rb') as f:
    impacts_list = pickle.load(f)
    
  impacts_mean = np.mean(impacts_list, axis=0)
  # influence over water
  influences_water = np.real(np.sum(impacts_mean[:,:3], axis=1)[3:3+N-2])
  # influence over other entities
  influences_social = np.real(np.sum(impacts_mean[:,3:], axis=1)[3:3+N-2])
  # sensitivity to water resources
  sensitivities_water = np.real(np.sum(impacts_mean[:3],axis=0)[3:3+N-2])
  # sensitivity to other entities
  sensitivities_social = np.real(np.sum(impacts_mean[3:],axis=0)[3:3+N-2])
    
    
  return influences_water, influences_social, sensitivities_water, sensitivities_social


influences_water = np.zeros((5,5))
influences_social = np.zeros((5,5))
sensitivities_water = np.zeros((5,5))
sensitivities_social = np.zeros((5,5))
param_versions = ['base', 'v1', 'v2', 'v3', 'v4']
parameterization_labels = ['Base','Ending "Regulatory Drought"', 'Improved water management', 'Increased state oversight', 'Change Nature of Agriculture']


for i in range(5):
  influences_water[i], influences_social[i], sensitivities_water[i], sensitivities_social[i] = \
    compute_influence_sensitivity_breakdown(param_versions[i])

fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all')
# Color scheme
colors_base = plt.cm.viridis(np.linspace(0, 1, len(param_versions)))
colors_light = colors_base.copy()
colors_light[:, -1] = 0.6  # Adjust alpha to make lighter versions
# Create an index for each tick position
ind = np.arange(N-2)

# # Figure size
# plt.figure(figsize=(15, 8))

# Bar width
width = 0.15

# Generate the stacked bars
# Plot of influences
for stack in range(len(param_versions)):
    bottoms = np.zeros(len(param_versions))
    for section in range(2): # for the water vs social
        values = influences_water if section == 0 else influences_social
        color = colors_light[stack] if section == 0 else colors_base[stack]
        axs[0].bar(ind + stack * width, values[stack], width,
                label=parameterization_labels[stack] if section == 0 else "",
                bottom=bottoms, color=color)
        bottoms += values[stack]
        
# Plot of sensitivities
for stack in range(len(param_versions)):
    bottoms = np.zeros(len(param_versions))
    for section in range(2): # for the water vs social
        values = sensitivities_water if section == 0 else sensitivities_social
        color = colors_light[stack] if section == 0 else colors_base[stack]
        axs[1].bar(ind + stack * width, values[stack], width,
                label=parameterization_labels[stack] if section == 0 else "",
                bottom=bottoms, color=color)
        bottoms += values[stack]
        
# Describe the data
axs[1].set_xlabel('Resource User')
axs[0].set_ylabel('Influences')
axs[1].set_ylabel('Sensitivities')
axs[0].set_title('Comparing sources of sensitivities and influences across parameterizations')

# Position and labels for ticks
axs[0].set_xticks(ind + width * ((N-2) - 1) / 2, N_list[:-2])
axs[1].set_xticks(ind + width * ((N-2) - 1) / 2, N_list[:-2])
axs[0].legend()
plt.savefig('influence_sensitivity_breakdown.svg', bbox_inches='tight')