import requests
import math
from bs4 import BeautifulSoup
from google.cloud import automl_v1beta1 as automl
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/jp/Downloads/prehension_credential.json"

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
    # r = s.get(f"http://13.125.216.41/{src_list[0]}")
    r = s.get(f"http://13.125.216.41/{src_list[1]}")
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

    # Google AutoML API 
    # TODO(developer): Uncomment and set the following variables
    project_id = 'prehension-282501'
    compute_region = 'us-central1'
    model_display_name = 'final_0825_og_model'
    inputs = {'SSIM': final_list[0],
            'Sound': final_list[1],
            'MSE': final_list[2],
            'PIR': final_list[3],
            'Radar': final_list[4],
            'MA_SSIM': ma_final_list[0],
            'MA_Sound': ma_final_list[1],
            'MA_Radar': ma_final_list[4],
            'MA_PIR': ma_final_list[3],
            'MA_MSE': ma_final_list[2]
            }

    client = automl.TablesClient(project=project_id, region=compute_region)
    feature_importance = False

    if feature_importance:
        response = client.predict(
            model_display_name=model_display_name,
            inputs=inputs,
            feature_importance=True,
        )
    else:
        response = client.predict(
            model_display_name=model_display_name, inputs=inputs
        )

    # Prediction Results
    class_name = []
    class_score = []
    for result in response.payload:
        class_name.append(result.tables.value.string_value)
        class_score.append(result.tables.score)

        ## original API code (commented out)
        # print("Predicted class name: {}".format(result.tables.value.string_value))
        # print("Predicted class score: {}".format(result.tables.score))

        # if feature_importance:
        #     # get features of top importance
        #     feat_list = [
        #         (column.feature_importance, column.column_display_name)
        #         for column in result.tables.tables_model_column_info
        #     ]
        #     feat_list.sort(reverse=True)
        #     if len(feat_list) < 10:
        #         feat_to_show = len(feat_list)
        #     else:
        #         feat_to_show = 10

        #     print("Features of top importance:")
        #     for feat in feat_list[:feat_to_show]:
        #         print(feat)
    
    index = class_score.index(max(class_score))
    result_class = class_name[index]   # NoP
    result_score = round(class_score[index] * 100, 2)  # Precision

    return(result_class, result_score)
    