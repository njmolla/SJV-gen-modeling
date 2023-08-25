import numpy as np
import pandas as pd
from pathlib import Path


def parameters_comparison(parameter,scenario1,scenario2):
  path = Path.cwd().joinpath('parameter_files',scenario1, '%s.xlsx'%(parameter))
  df1 = pd.read_excel(path, index_col = 0)
  df1 = df1.fillna(0) # array of weights for sampling
  #df1 = df1.fillna(0).values[:,1:] # array of weights for sampling
  #param_1 = np.array(param_1, dtype=[('O', float)]).astype(float)
  path = Path.cwd().joinpath('parameter_files',scenario2, '%s.xlsx'%(parameter))
  df2 = pd.read_excel(path, index_col = 0)
  df2 = df2.fillna(0) # array of weights for sampling
  #df2 = df2.fillna(0).values[:,1:] # array of weights for sampling
  #param_2 = np.array(param_2, dtype=[('O', float)]).astype(float)
  
    # Find the dataframe with more columns
  if len(df1.columns) < len(df2.columns):
      missing_cols = set(df2.columns) - set(df1.columns)
      for col in missing_cols:
          df1[col] = 0  # add missing column filled with 0s
  elif len(df2.columns) < len(df1.columns):
      missing_cols = set(df1.columns) - set(df2.columns)
      for col in missing_cols:
          df2[col] = 0  # add missing column filled with 0s

  # Ensure both dataframes have the same column order
  df1 = df1.reindex(columns=df2.columns)

  # Reindexing rows to match all unique row labels from both dataframes
  all_indices = df1.index.union(df2.index)
  df1 = df1.reindex(all_indices, fill_value=0)
  df2 = df2.reindex(all_indices, fill_value=0)

  # Compare the two dataframes
  comparison = df1.compare(df2)

  # Return the comparison result dataframe, and the updated dataframes
  return comparison, df1, df2

comparison, df1, df2 = parameters_comparison('de_dg_sw_upper','base','v4')
  
  # if np.shape(param_1)==np.shape(param_2):
  #   diff = df1.compare(df2)
  #   #diff = param_2-param_1
  #   # if np.sum(diff)<1e-5:
  #   #   print('no difference')
  # else:
    
  #   if parameter == 'sigmas':
  #     dim = (36,36)
  #     if np.shape(param_1) != dim:
  #       param_1()
      
  #   if np.shape(param_1)[0]<np.shape(param_2)[0]:
  #     param_1_padded = np.zeros(np.shape(param_2))
      
  #   diff = None
  #   print('parameters are different shapes')
      
  # return diff, param_1, param_2
  
#diff, sigma_weights, sigma_weights_v1 = parameters_comparison('sigmas','base','v1')
#diff, de_dg, de_dg_v1 = parameters_comparison('de_dg_sw_lower','base','v1')

# path = Path.cwd().joinpath('parameter_files', 'base', 'sigmas.csv')
# sigmas_df = pd.read_csv(path)
# sigma_weights = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
# path = Path.cwd().joinpath('parameter_files', 'v1', 'sigmas.xlsx')
#   sigmas_df_v1 = pd.read_excel(path)
#   sigma_weights_v1 = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
# path = Path.cwd().joinpath('parameter_files', 'v1', 'sigmas.xlsx')
# sigmas_df_v1 = pd.read_excel(path)
# sigma_weights_v1 = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
# diff = sigma_weights_v1-sigma_weights
# np.any(diff>0)
# np.sum(diff)
# lambdas = np.zeros((N+K,tot))  # lambda_k,n is kxn $
# lambda_hats = np.zeros((M,tot))
# path = Path.cwd().joinpath('parameter_files', 'base', 'lambdas.xlsx')
# lambdas_df = pd.read_excel(path)
# lambdas_weights = lambdas_df.fillna(0).values[:,1:] # array of weights for sampling
# lambdas_weights = np.array(lambdas_weights, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'lambdas.xlsx')
# lambdas_df = pd.read_excel(path)
# lambdas_weights = lambdas_df.fillna(0).values[:,1:] # array of weights for sampling
# lambdas_weights = np.array(lambdas_weights, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v1', 'lambdas.xlsx')
# lambdas_df_v1 = pd.read_excel(path)
# lambdas_weights_v1 = lambdas_df_v1.fillna(0).values[:,1:] # array of weights for sampling
# lambdas_weights_v1 = np.array(lambdas_weights_v1, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v1', 'sigmas.xlsx')
# sigmas_df_v1 = pd.read_excel(path)
# sigma_weights_v1 = sigmas_df_v1.fillna(0).values[:,1:] # array of weights for sampling
# diff = sigma_weights_v1-sigma_weights
# np.sum(diff)
# diff = lambdas_weights_v1-lambdas_weights
# np.sum(diff)
# runfile('C:/Users/nmoll/OneDrive/Documents/SJV-gen-modeling/parameterization_v2.py', wdir='C:/Users/nmoll/OneDrive/Documents/SJV-gen-modeling')
# runfile('C:/Users/nmoll/OneDrive/Documents/SJV-gen-modeling/influence_experiment.py', wdir='C:/Users/nmoll/OneDrive/Documents/SJV-gen-modeling')
# path = Path.cwd().joinpath('parameter_files', 'v2', 'sigmas.xlsx')
# sigmas_df = pd.read_excel(path)
# sigma_weights_v2 = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
# sigma_weights_v2 = np.array(sigma_weights_v2, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v3', 'sigmas.xlsx')
# sigmas_df = pd.read_excel(path)
# sigma_weights_v3 = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
# sigma_weights_v3 = np.array(sigma_weights_v3, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v4', 'sigmas.xlsx')
# sigmas_df = pd.read_excel(path)
# sigma_weights_v3 = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
# sigma_weights_v4 = np.array(sigma_weights_v4, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v4', 'sigmas.xlsx')
# sigmas_df = pd.read_excel(path)
# sigma_weights_v4 = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
# sigma_weights_v4 = np.array(sigma_weights_v4, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v3', 'sigmas.xlsx')
# sigmas_df = pd.read_excel(path)
# sigma_weights_v3 = sigmas_df.fillna(0).values[:,1:] # array of weights for sampling
# sigma_weights_v3 = np.array(sigma_weights_v3, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v2', 'lambdas.xlsx')
# lambdas_df = pd.read_excel(path)
# lambdas_weights_v2 = lambdas_df.fillna(0).values[:,1:] # array of weights for sampling
# lambdas_weights_v2 = np.array(lambdas_weights_v2, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v3', 'lambdas.xlsx')
# lambdas_df = pd.read_excel(path)
# lambdas_weights_v3 = lambdas_df.fillna(0).values[:,1:] # array of weights for sampling
# lambdas_weights_v3 = np.array(lambdas_weights_v3, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v4', 'lambdas.xlsx')
# lambdas_df = pd.read_excel(path)
# lambdas_weights_v4 = lambdas_df.fillna(0).values[:,1:] # array of weights for sampling
# lambdas_weights_v4 = np.array(lambdas_weights_v4, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_sw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# sw_lower = df.fillna(0).values[:,1:]
# sw_lower = np.array(sw_lower, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_sw_upper.xlsx')
# df = pd.read_excel(path)
# sw_upper = df.fillna(0).values[:,1:]
# sw_upper = np.array(sw_upper, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# gw_lower = df.fillna(0).values[:,1:]
# gw_lower = np.array(gw_lower, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gw_upper.xlsx')
# gw_upper = df.fillna(0).values[:,1:]
# gw_upper = np.array(gw_upper, dtype=[('O', float)]).astype(float)
# gwq_lower = df.fillna(0).values[:,1:]
# gwq_lower = np.array(gwq_lower, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gwq_upper.xlsx')
# df = pd.read_excel(path)
# gwq_upper = df.fillna(0).values[:,1:]
# gwq_upper = np.array(gwq_upper, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'dt_dh.xlsx')
# data = pd.read_excel(path, sheet_name=None) #lower bounds for de_dg for sw
# lower = data['lower'].fillna(0).values[:,1:]
# lower = np.array(lower, dtype=[('O', float)]).astype(float)
# upper  = data['upper'].fillna(0).values[:,1:]
# upper = np.array(upper, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v1', 'de_dg_sw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# sw_lower_v1 = df.fillna(0).values[:,1:]
# sw_lower_v1 = np.array(sw_lower_v1, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v1', 'de_dg_sw_upper.xlsx')
# df = pd.read_excel(path)
# sw_upper_v1 = df.fillna(0).values[:,1:]
# sw_upper_v1 = np.array(sw_upper, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v2', 'de_dg_sw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# sw_lower_v2 = df.fillna(0).values[:,1:]
# sw_lower_v2 = np.array(sw_lower_v2, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v2', 'de_dg_sw_upper.xlsx')
# df = pd.read_excel(path)
# sw_upper_v2 = df.fillna(0).values[:,1:]
# sw_upper_v2 = np.array(sw_upper_v2, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v1', 'de_dg_sw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# sw_lower_v1 = df.fillna(0).values[:,1:]
# sw_lower_v1 = np.array(sw_lower_v1, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v1', 'de_dg_sw_upper.xlsx')
# df = pd.read_excel(path)
# sw_upper_v1 = df.fillna(0).values[:,1:]
# sw_upper_v1 = np.array(sw_upper_v1, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v3', 'de_dg_sw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# sw_lower_v3 = df.fillna(0).values[:,1:]
# sw_lower_v3 = np.array(sw_lower_v3, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v3', 'de_dg_sw_upper.xlsx')
# df = pd.read_excel(path)
# sw_upper_v3 = df.fillna(0).values[:,1:]
# sw_upper_v3 = np.array(sw_upper_v3, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v4', 'de_dg_sw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# sw_lower_v4 = df.fillna(0).values[:,1:]
# sw_lower_v4 = np.array(sw_lower_v4, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v4', 'de_dg_sw_upper.xlsx')
# df = pd.read_excel(path)
# sw_upper_v4 = df.fillna(0).values[:,1:]
# sw_upper_v4 = np.array(sw_upper_v4, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v1', 'de_dg_gw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# gw_lower_v1 = df.fillna(0).values[:,1:]
# gw_lower_v1 = np.array(gw_lower_v1, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gw_upper.xlsx')
# gw_upper_v1 = df.fillna(0).values[:,1:]
# gw_upper_v1 = np.array(gw_upper_v1, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v2', 'de_dg_gw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# gw_lower_v2 = df.fillna(0).values[:,1:]
# gw_lower_v2 = np.array(gw_lower_v2, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gw_upper.xlsx')
# gw_upper_v2 = df.fillna(0).values[:,1:]
# gw_upper_v2 = np.array(gw_upper_v2, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v3', 'de_dg_gw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# gw_lower_v3 = df.fillna(0).values[:,1:]
# gw_lower_v3 = np.array(gw_lower_v3, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gw_upper.xlsx')
# gw_upper_v3 = df.fillna(0).values[:,1:]
# gw_upper_v3 = np.array(gw_upper_v3, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v4', 'de_dg_gw_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# gw_lower_v4 = df.fillna(0).values[:,1:]
# gw_lower_v4 = np.array(gw_lower_v4, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gw_upper.xlsx')
# gw_upper_v4 = df.fillna(0).values[:,1:]
# gw_upper_v4 = np.array(gw_upper_v4, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v1', 'de_dg_gwq_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# gwq_lower_v1 = df.fillna(0).values[:,1:]
# gwq_lower_v1 = np.array(gwq_lower_v1, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gwq_upper.xlsx')
# df = pd.read_excel(path)
# gwq_upper_v1 = df.fillna(0).values[:,1:]
# gwq_upper_v1 = np.array(gwq_upper_v1, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'v2', 'de_dg_gwq_lower.xlsx')
# df = pd.read_excel(path) #lower bounds for de_dg for sw
# gwq_lower_v2 = df.fillna(0).values[:,1:]
# gwq_lower_v2 = np.array(gwq_lower_v2, dtype=[('O', float)]).astype(float)
# path = Path.cwd().joinpath('parameter_files', 'base', 'de_dg_gwq_upper.xlsx')
# df = pd.read_excel(path)
# gwq_upper_v2 = df.fillna(0).values[:,1:]
# gwq_upper_v2 = np.array(gwq_upper_v2, dtype=[('O', float)]).astype(float)