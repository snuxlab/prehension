import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime as dt

# csv load
df = pd.read_csv('data_10sec_sensitive.csv', delimiter=',')

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
l_mse_diff_40 = ['None', 'None', 'None', 'None', 'None']
l_mse_diff_50 = ['None', 'None', 'None', 'None', 'None']

l_ssim_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_30 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_40 = ['None', 'None', 'None', 'None', 'None']
l_ssim_diff_50 = ['None', 'None', 'None', 'None', 'None']

l_sound_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_30 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_40 = ['None', 'None', 'None', 'None', 'None']
l_sound_diff_50 = ['None', 'None', 'None', 'None', 'None']

l_radar_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_30 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_40 = ['None', 'None', 'None', 'None', 'None']
l_radar_diff_50 = ['None', 'None', 'None', 'None', 'None']

l_pir_diff_10 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_20 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_30 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_40 = ['None', 'None', 'None', 'None', 'None']
l_pir_diff_50 = ['None', 'None', 'None', 'None', 'None']

# update previous sensor info
for i in range(5, len(df['Time'])):

    l_mse_diff_10.append(df['MSE'][i-1])
    l_mse_diff_20.append(df['MSE'][i-2])
    l_mse_diff_30.append(df['MSE'][i-3])
    l_mse_diff_40.append(df['MSE'][i-4])
    l_mse_diff_50.append(df['MSE'][i-5])

    l_ssim_diff_10.append(df['SSIM'][i-1])
    l_ssim_diff_20.append(df['SSIM'][i-2])
    l_ssim_diff_30.append(df['SSIM'][i-3])
    l_ssim_diff_40.append(df['SSIM'][i-4])
    l_ssim_diff_50.append(df['SSIM'][i-5])

    l_sound_diff_10.append(df['Sound'][i-1])
    l_sound_diff_20.append(df['Sound'][i-2])
    l_sound_diff_30.append(df['Sound'][i-3])
    l_sound_diff_40.append(df['Sound'][i-4])
    l_sound_diff_50.append(df['Sound'][i-5])

    l_radar_diff_10.append(df['Radar'][i-1])
    l_radar_diff_20.append(df['Radar'][i-2])
    l_radar_diff_30.append(df['Radar'][i-3])
    l_radar_diff_40.append(df['Radar'][i-4])
    l_radar_diff_50.append(df['Radar'][i-5])

    l_pir_diff_10.append(df['PIR'][i-1])
    l_pir_diff_20.append(df['PIR'][i-2])
    l_pir_diff_30.append(df['PIR'][i-3])
    l_pir_diff_40.append(df['PIR'][i-4])
    l_pir_diff_50.append(df['PIR'][i-5])

# add new column to dataframe
df['MSE-10'] = l_mse_diff_10
df['MSE-20'] = l_mse_diff_20
df['MSE-30'] = l_mse_diff_30
df['MSE-40'] = l_mse_diff_40
df['MSE-50'] = l_mse_diff_50

df['SSIM-10'] = l_ssim_diff_10
df['SSIM-20'] = l_ssim_diff_20
df['SSIM-30'] = l_ssim_diff_30
df['SSIM-40'] = l_ssim_diff_40
df['SSIM-50'] = l_ssim_diff_50

df['Sound-10'] = l_sound_diff_10
df['Sound-20'] = l_sound_diff_20
df['Sound-30'] = l_sound_diff_30
df['Sound-40'] = l_sound_diff_40
df['Sound-50'] = l_sound_diff_50

df['Radar-10'] = l_radar_diff_10
df['Radar-20'] = l_radar_diff_20
df['Radar-30'] = l_radar_diff_30
df['Radar-40'] = l_radar_diff_40
df['Radar-50'] = l_radar_diff_50

df['PIR-10'] = l_pir_diff_10
df['PIR-20'] = l_pir_diff_20
df['PIR-30'] = l_pir_diff_30
df['PIR-40'] = l_pir_diff_40
df['PIR-50'] = l_pir_diff_50

drop_list = df[df['MSE-10'] == 'None'].index
df_sample = df.drop(drop_list)
df_sample.to_csv('data_10sec_sensitive_trend.csv', index = False, sep=',', encoding='utf-8')
    
