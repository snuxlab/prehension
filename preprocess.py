import pandas as pd
from pandas import DataFrame

# csv load
df = pd.read_csv('temp.csv', delimiter=',')

# iteration
condition = True
time_count = 10
print("Original : ", df.shape)
while condition:
    # column information
    sound = df['Sound'].tolist()
    time = df['Time'].tolist()
    NoP = df['NoP'].tolist()

    index = 0
    index_list = []
    # Balance data
    while True:   
        if index + 1 >= len(sound):
            index_list.append(index)
            break

        # Sound condition
        sound_condition = False
        if sound[index] <= 6:
            if sound[index+1] <= 6:
                sound_condition = True
        
        if sound_condition:
            start_time = int(''.join(time[index].split(':')))
            end_time = int(''.join(time[index + 1].split(':')))
            time_diff = end_time - start_time

            # Time difference condition
            if time_diff <= time_count and NoP[index] > 1 and NoP[index + 1] > 1:
                index_list.append(index)
                index += 1
            else:
                index_list.append(index)
        else:
            index_list.append(index)

        index += 1
        # Break condition
        if index >= len(sound):
            break

    df = df.iloc[index_list, :]

    num_of_people = df["NoP"]
    bool_p_zero = num_of_people == 0
    bool_p_one = num_of_people == 1
    #bool_p_two = num_of_people == 2
    bool_p_n = num_of_people > 1

    p_zero = num_of_people[bool_p_zero]
    p_one = num_of_people[bool_p_one]
    #p_two = num_of_people[bool_p_two]
    p_n = num_of_people[bool_p_n]

   
    print("NoP 0: {}, NoP 1: {}, NoP n: {}, time: {}".format(len(p_zero), len(p_one), len(p_n), time_count))

    # break loop
    if len(p_n) < 1800:
        condition = False
    else:
        time_count += 1
        # print("Time: {}".format(time_count))

print("Processed: ", df.shape)
df.to_csv('data_final_01n.csv', sep=',', encoding='utf-8')
