import pandas as pd
from pandas import DataFrame
import datetime
from random import sample

###########################################  여러 CSV 합치기  ###########################################
"""
# csv load
df_1 = pd.read_csv('sensor_1.csv', delimiter=',')
df_2 = pd.read_csv('sensor_2.csv', delimiter=',')
df_3 = pd.read_csv('sensor_3.csv', delimiter=',')
df_4 = pd.read_csv('sensor_4.csv', delimiter=',')
df_5 = pd.read_csv('sensor_5.csv', delimiter=',')

df_merge = pd.concat([df_1, df_2, df_3, df_4, df_5]) # row bind : axis = 0, default
print(df_merge.head())

df_merge.to_csv('sensor_merged.csv', sep=',', encoding='utf-8')
"""

###############################################  NoP 0명  ###############################################
# CSV파일 불러오기
df = pd.read_csv('sensor_merged.csv', delimiter=',')
df["Time"] = pd.to_datetime(df["Time"])

# 날짜 기준 선정하기 - (년, 월, 일, 시간, 분, 초)
start_time = datetime.datetime(2020, 8, 24, 0, 0, 0)
end_time = datetime.datetime(2020, 8, 24, 23, 59, 59)
work_time_start = datetime.datetime(2020, 8, 24, 10, 0, 0)
work_time_end = datetime.datetime(2020, 8, 24, 18, 0, 0)

# 선정한 날짜별로 필터링하기
# 10-6
df_working = df[(df["Time"] >= work_time_start) & (df["Time"] <= work_time_end) & (df["NoP"] == 0)]
# 그 외 시간
df_notworking = df[(df["Time"] < end_time) & (df["Time"] > work_time_end) & (df["NoP"] == 0)]


# 필터링된 데이터프레임에서 샘플링하기
df_sample_notworking = df_notworking.sample(n=3000, random_state=1)
df_sample_working = df_working.sample(n=7000, random_state=1)
df_sample_zero = pd.concat([df_sample_notworking, df_sample_working])

###############################################  NoP 1명  ###############################################

# 선정한 NoP로 필터링하기
df_one = df[(df["NoP"] == 1)]

# 필터링된 데이터프레임에서 샘플링하기
df_sample_one = df_one.sample(n=10000, random_state=1)

###############################################  NoP N명  ###############################################

# 선정한 NoP로 필터링하기
df_345camera = df[(df["Cam"] > 2) & (df["NoP"] > 1) & (df["NoP"] < 6)]
df_012camera = df[(df["Cam"]<= 2) & (df["NoP"] > 1) & (df["NoP"] < 6)]

# 필터링된 데이터프레임에서 샘플링하기
df_sample_012camera = df_012camera.sample(n=8947, random_state=1)
df_sample_many = pd.concat([df_sample_012camera, df_345camera])

###########################################  NoP 기준 DF 합치기  ###########################################

df_sample_merge = pd.concat([df_sample_zero, df_sample_one, df_sample_many])

# 데이터프레임 시간 타입 맞추기
# 년월일 빼고 시/분/초로 (원래 형식대로 원복)
time_list = df_sample_merge["Time"].to_list()
temp_list = []
for line in time_list:
  temp_list.append(line.strftime("%H:%M:%S"))

df_sample_merge["Time"] = temp_list
df_sample_merge.to_csv('sensor_sampled.csv', sep=',', encoding='utf-8')