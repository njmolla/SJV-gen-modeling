import numpy as np
import pandas as pd
from pathlib import Path


def parameters_comparison(parameter,scenario1,scenario2):
  path = Path.cwd().joinpath('parameter_files',scenario1, '%s.xlsx'%(parameter))
  param_df_1 = pd.read_excel(path)
  param_1 = param_df_1.fillna(0).values[:,1:] # array of weights for sampling
  param_1 = np.array(param_1, dtype=[('O', float)]).astype(float)
  path = Path.cwd().joinpath('parameter_files',scenario2, '%s.xlsx'%(parameter))
  param_df_2 = pd.read_excel(path)
  param_2 = param_df_2.fillna(0).values[:,1:] # array of weights for sampling
  param_2 = np.array(param_2, dtype=[('O', float)]).astype(float)
  if np.shape(param_1)==np.shape(param_2):
    diff = param_2-param_1
  else:
    diff = None
    print('parameters are different shapes')
    
  if np.sum(diff)<1e-5:
    print('no difference')
    
  return diff, param_1, param_2
  
diff, sigma_weights, sigma_weights_v1 = parameters_comparison('sigmas','base','v1')
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