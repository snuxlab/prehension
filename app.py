from flask import Flask, render_template
import threading
import random

import autoML  # GOOGLE AUTOML API fucntion

app = Flask(__name__)

@app.route('/')
def index():
    # class name = NoP
    # class precision = precision
    class_name, class_precision = autoML.deploy()
    if(class_name == '0'):
        result_name = 'Empty'
    elif(class_name == '1'):
        result_name = 'Occupied'
    elif(class_name == 'n'):
        result_name = 'Crowded'     # Empty/Occupied/Crowded
    return render_template('index.html', name=class_name, result=result_name, precision=class_precision)

if __name__ == '__main__':
    app.run()
