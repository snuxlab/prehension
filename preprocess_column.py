import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime as dt

# csv load
df = pd.read_csv('sensor_sampled_minmax.csv', delimiter=',')

# SSIM -> 1 - SSIM
SSIM = df["SSIM"].copy()
SSIM_modified = 1 - SSIM
df['SSIM'] = SSIM_modified

# Sound -> log(Sound)
Sound = df["Sound"].copy()
Sound_boolean = Sound == 0
Sound[Sound_boolean] = 1
Sound_modified = np.log(Sound)
df["Sound"] = Sound_modified

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

# add t-10, t-20, t-30 columns to dataframe
# Moving Average = done via excel
# Moving Avg. code can be inserted here.
df['MSE_10'] = l_mse_diff_10
df['MSE_20'] = l_mse_diff_20
df['MSE_30'] = l_mse_diff_30

df['SSIM_10'] = l_ssim_diff_10
df['SSIM_20'] = l_ssim_diff_20
df['SSIM_30'] = l_ssim_diff_30

df['Sound_10'] = l_sound_diff_10
df['Sound_20'] = l_sound_diff_20
df['Sound_30'] = l_sound_diff_30

df['Radar_10'] = l_radar_diff_10
df['Radar_20'] = l_radar_diff_20
df['Radar_30'] = l_radar_diff_30

df['PIR_10'] = l_pir_diff_10
df['PIR_20'] = l_pir_diff_20
df['PIR_30'] = l_pir_diff_30

drop_list = df[df['MSE_10'] == 'None'].index
df_sample = df.drop(drop_list)
df_sample.to_csv('sensor_sampled_minmax_ma.csv', index = False, sep=',', encoding='utf-8')
    
