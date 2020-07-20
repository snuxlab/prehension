import requests
import math
from bs4 import BeautifulSoup
from google.cloud import automl_v1beta1 as automl
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/minjoon/Documents/GitHub/prehension/prehension_credential.json"

# Google AutoML API function
def deploy():
    # Senosr data Crawling
    URL = "http://15.164.250.196//"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    iframe_src = soup.find('iframe')['src']

    s = requests.Session()
    r = s.get(f"http://15.164.250.196/{iframe_src}")


    soup = BeautifulSoup(r.content, "html.parser")
    data_list=list(soup)[0]
    data = data_list.split(',')

    SSIM_1 = round(1-float(data[3]), 2)
    if int(data[4]) <= 0:
        log_Sound = 0
    else:
        log_Sound = round(math.log(int(data[4])), 2)
    MSE = round(float(data[2]),2)
    PIR = int(data[6])
    Radar = int(data[5])

    final_list = [SSIM_1, log_Sound, MSE, PIR, Radar]
    print("Final List: ", final_list)

    # Google AutoML API 
    # TODO(developer): Uncomment and set the following variables
    project_id = 'prehension-282501'
    compute_region = 'us-central1'
    model_display_name = 'tensec_0_1_n'
    inputs = {'1_SSIM': final_list[0],
            'log_Sound': final_list[1],
            'MSE': final_list[2],
            'PIR': final_list[3],
            'Radar': final_list[4]
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
    