# %%
# import libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime as dt

# %%
# csv load
df = pd.read_csv('data/test-20200930.csv', delimiter=',')

# %%
# SSIM -> 1 - SSIM
SSIM = df["SSIM"].copy()
SSIM_modified = 1 - SSIM
df['SSIM'] = SSIM_modified

# %%
# Sound -> log(Sound)
Sound = df["Sound"].copy()
Sound_boolean = Sound == 0
Sound[Sound_boolean] = 1
Sound_modified = np.log(Sound)
df["Sound"] = Sound_modified

# %%
# new column list
l_mse_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_mse_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_mse_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_ssim_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_sound_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_radar_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_30 = ['None', 'None', 'None', 'None', 'None']

l_pir_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_30 = ['None', 'None', 'None', 'None', 'None']

# update previous sensor info
for i in range(5, len(df['Time'])):

    l_mse_diff_10.append(df['MSE'][i-1])
    l_mse_diff_20.append(df['MSE'][i-2])
    l_mse_diff_30.append(df['MSE'][i-3])

    l_ssim_diff_10.append(df['SSIM'][i-1])
    l_ssim_diff_20.append(df['SSIM'][i-2])
    l_ssim_diff_30.append(df['SSIM'][i-3])

    l_sound_diff_10.append(df['Sound'][i-1])
    l_sound_diff_20.append(df['Sound'][i-2])
    l_sound_diff_30.append(df['Sound'][i-3])

    l_radar_diff_10.append(df['Radar'][i-1])
    l_radar_diff_20.append(df['Radar'][i-2])
    l_radar_diff_30.append(df['Radar'][i-3])

    l_pir_diff_10.append(df['PIR'][i-1])
    l_pir_diff_20.append(df['PIR'][i-2])
    l_pir_diff_30.append(df['PIR'][i-3])

# %%
# add t-10, t-20, t-30 columns to dataframe for 
# Moving Average calculation
mse_pre = pd.DataFrame()
mse_pre['MSE_10'] = l_mse_diff_10
mse_pre['MSE_20'] = l_mse_diff_20
mse_pre['MSE_30'] = l_mse_diff_30
df['MSE_MA'] = mse_pre[5:].mean(axis=1)

ssim_pre = pd.DataFrame()
ssim_pre['SSIM_10'] = l_ssim_diff_10
ssim_pre['SSIM_20'] = l_ssim_diff_20
ssim_pre['SSIM_30'] = l_ssim_diff_30
df['SSIM_MA'] = ssim_pre[5:].mean(axis=1)

sound_pre = pd.DataFrame()
sound_pre['Sound_10'] = l_sound_diff_10
sound_pre['Sound_20'] = l_sound_diff_20
sound_pre['Sound_30'] = l_sound_diff_30
df['Sound_MA'] = sound_pre[5:].mean(axis=1)

radar_pre = pd.DataFrame()
radar_pre['Radar_10'] = l_radar_diff_10
radar_pre['Radar_20'] = l_radar_diff_20
radar_pre['Radar_30'] = l_radar_diff_30
df['Radar_MA'] = radar_pre[5:].mean(axis=1)

pir_pre = pd.DataFrame()
pir_pre['PIR_10'] = l_pir_diff_10
pir_pre['PIR_20'] = l_pir_diff_20
pir_pre['PIR_30'] = l_pir_diff_30
df['PIR_MA'] = pir_pre[5:].mean(axis=1)

# %%
df_sample = df.iloc[5:].reset_index(drop=True)
df_sample['Hour'] = pd.to_datetime(df_sample['Time'], format='%H:%M:%S').dt.hour
df_sample.to_csv('data/test1.csv', index = False, sep=',', encoding='utf-8')
    
# %%
