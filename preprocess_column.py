import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime as dt

# csv load
df = pd.read_csv('sensor_sampled_minmax.csv', delimiter=',')

# # SSIM -> 1 - SSIM
# SSIM = df["SSIM"].copy()
# SSIM_modified = 1 - SSIM
# df['SSIM'] = SSIM_modified

# # Sound -> log(Sound)
# Sound = df["Sound"].copy()
# Sound_boolean = Sound == 0
# Sound[Sound_boolean] = 1
# Sound_modified = np.log(Sound)
# df["Sound"] = Sound_modified

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

    l_mse_diff_10.append(df['mm_MSE'][i-1])
    l_mse_diff_20.append(df['mm_MSE'][i-2])
    l_mse_diff_30.append(df['mm_MSE'][i-3])

    l_ssim_diff_10.append(df['mm_SSIM'][i-1])
    l_ssim_diff_20.append(df['mm_SSIM'][i-2])
    l_ssim_diff_30.append(df['mm_SSIM'][i-3])

    l_sound_diff_10.append(df['mm_Sound'][i-1])
    l_sound_diff_20.append(df['mm_Sound'][i-2])
    l_sound_diff_30.append(df['mm_Sound'][i-3])

    l_radar_diff_10.append(df['mm_Radar'][i-1])
    l_radar_diff_20.append(df['mm_Radar'][i-2])
    l_radar_diff_30.append(df['mm_Radar'][i-3])

    l_pir_diff_10.append(df['mm_PIR'][i-1])
    l_pir_diff_20.append(df['mm_PIR'][i-2])
    l_pir_diff_30.append(df['mm_PIR'][i-3])

# add new column to dataframe
df['MSE-10'] = l_mse_diff_10
df['MSE-20'] = l_mse_diff_20
df['MSE-30'] = l_mse_diff_30

df['SSIM-10'] = l_ssim_diff_10
df['SSIM-20'] = l_ssim_diff_20
df['SSIM-30'] = l_ssim_diff_30

df['Sound-10'] = l_sound_diff_10
df['Sound-20'] = l_sound_diff_20
df['Sound-30'] = l_sound_diff_30

df['Radar-10'] = l_radar_diff_10
df['Radar-20'] = l_radar_diff_20
df['Radar-30'] = l_radar_diff_30

df['PIR-10'] = l_pir_diff_10
df['PIR-20'] = l_pir_diff_20
df['PIR-30'] = l_pir_diff_30

drop_list = df[df['MSE-10'] == 'None'].index
df_sample = df.drop(drop_list)
df_sample.to_csv('sensor_sampled_minmax_ma.csv', index = False, sep=',', encoding='utf-8')
    
