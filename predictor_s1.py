import requests
import math
from bs4 import BeautifulSoup

# import libraries for keras model and scaler
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np
import sklearn

# load model and scaler
# comment out scaler if importing v2 &
# make inputs input_pre only on lines 80-85
model = keras.models.load_model('model/prehension_v1')
scaler = joblib.load('model/scaler.save')

# Google AutoML API function
def deploy():
    # Sensor data Crawling
    URL = "http://13.125.216.41/"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    iframes = soup.find_all('iframe')

    #src_list stores each sensor's iframe source url
    src_list = []
    for iframe in iframes:
        src_list.append(iframe['src'])

    #우선은 센서1에 대해서만 진행 --> src_list[0] 이 Sensor 1 and so on...
    s = requests.Session()
    r = s.get(f"http://13.125.216.41/{src_list[0]}")
    # r = s.get(f"http://13.125.216.41/{src_list[1]}")
    # r = s.get(f"http://13.125.216.41/{src_list[2]}")
    # r = s.get(f"http://13.125.216.41/{src_list[3]}")
    # r = s.get(f"http://13.125.216.41/{src_list[4]}")

    soup = BeautifulSoup(r.content, "html.parser")
    li = soup.prettify().split('\r\n')

    data_t = li[0].split(',')

    SSIM_1 = round(1-float(data_t[3]), 2)
    if int(data_t[4]) <= 0:
        log_Sound = 0
    else:
        log_Sound = round(math.log(int(data_t[4])), 2)
    MSE = round(float(data_t[2]),2)
    PIR = int(data_t[6])
    Radar = int(data_t[5])

    final_list = [SSIM_1, log_Sound, MSE, PIR, Radar]
    print("Final List: ", final_list)


    data_t10 = li[1].split(',')
    data_t20 = li[2].split(',')
    data_t30 = li[3].split(',')
    moving_data = [data_t10, data_t20, data_t30]

    MA_SSIM_1, MA_log_Sound, MA_MSE, MA_PIR, MA_Radar = 0, 0, 0, 0, 0
    for data in moving_data:
        MA_SSIM_1 += round(1-float(data[3]), 2)
        if int(data[4]) <= 0:
            MA_log_Sound += 0
        else:
            MA_log_Sound += round(math.log(int(data[4])), 2)
        MA_MSE += round(float(data[2]),2)
        MA_PIR += int(data[6])
        MA_Radar += int(data[5])

    MA_SSIM_1, MA_log_Sound, MA_MSE, MA_PIR, MA_Radar = round(MA_SSIM_1/5,2), round(MA_log_Sound/5,2), round(MA_MSE/5,2), round(MA_PIR/5,2), round(MA_Radar/5,2)

    ma_final_list = [MA_SSIM_1, MA_log_Sound, MA_MSE, MA_PIR, MA_Radar]
    print("MA Final List: ", ma_final_list)

    # create final list and apply scaler (MinMax) before feeding into model
    input_pre = [
        MSE, SSIM_1, log_Sound, Radar, PIR, MA_MSE, MA_SSIM_1, MA_log_Sound, MA_Radar, MA_PIR]
    input_pre = np.asarray(input_pre).astype(np.float32)
    input_scaled = scaler.transform(input_pre.reshape(1,-1))
    input_scaled = tf.convert_to_tensor(input_scaled)
    
    pred = model.predict(input_scaled)
    result_class = pred.argmax(axis=-1)[0]
    result_score = round(pred[0][result_class] * 100, 2)

    return(result_class, result_score)
    
