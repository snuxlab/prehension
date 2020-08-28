import pandas as pd
import numpy as np

def z_score_normalize(df):
  column_list = ["MSE", "SSIM", "Sound", "Radar", "PIR"]
  for column in df:
    if column in column_list:
      z_score = (df[column].subtract(df[column].mean())).divide(df[column].std(), fill_value = 0).round(3)

      column_name = "z_" + column
      df[column_name] = z_score

  return df

def min_max_normalize(df):
  column_list = ["MSE", "SSIM", "Sound", "Radar", "PIR"]
  for column in df:
    if column in column_list:
      min_max = (df[column].subtract(df[column].min())).divide(df[column].max() - df[column].min()).round(3)

      column_name = "mm_" + column
      df[column_name] = min_max

  return df

# import finalized sampled csv file
# order: 5sensor -> normalization -> column
df = pd.read_csv("/content/sensor_sampled.csv", delimiter=',')
z_df = z_score_normalize(df)
mm_df = min_max_normalize(z_df)
mm_df.to_csv('normalized_sensor_sample.csv', index = False, sep=',', encoding='utf-8')