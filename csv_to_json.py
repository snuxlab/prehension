import pandas as pd
import json

# read csv file and convert it to dictionary
df = pd.read_csv(r'C:\Users\cross\Documents\GitHub\prehension\csv_data\10sec_sensor_fix.csv')
date = df["Date"].tolist()
time = df["Time"].tolist()
nop = df["NoP"].tolist()
temp_dict = {"Date":date, "Time":time, "NoP":nop}

# write dictionary to JSON file
with open(r'C:\Users\cross\Documents\GitHub\prehension\csv_data\temp.json', 'w', encoding='utf-8') as fp:
    json.dump(temp_dict, fp)