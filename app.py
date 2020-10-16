from flask import Flask, render_template
from flask import jsonify
import threading
import random

# import predictor_s1, predictor2, predictor3 
import predictor_s1, predictor_s2

app = Flask(__name__)

# just in case...
@app.route('/')
def hello():
    return "Hello World!"

@app.route('/dashboard1')
def dashboard1():
    # class name = NoP
    # class precision = precision
    class_name, class_precision = predictor_s1.deploy()
    return render_template('dashboard.html', name=class_name, precision=class_precision)

@app.route('/dashboard2')
def dashboard2():
    # class name = NoP
    # class precision = precision
    class_name, class_precision = predictor_s2.deploy()
    if(class_name == 0):
        result_name = 'Empty'
    elif(class_name == 1):
        result_name = 'Occupied'
    elif(class_name == 2):
        result_name = 'Crowded'     # Empty/Occupied/Crowded
    return render_template('index.html', name=class_name, result=result_name, precision=class_precision)

# @app.route('/dashboard3')
# def dashboard3():
#     # class name = NoP
#     # class precision = precision
#     class_name, class_precision = predictor3.deploy()
#     if(class_name == '0'):
#         result_name = 'Empty'
#     elif(class_name == '1'):
#         result_name = 'Occupied'
#     elif(class_name == 'n'):
#         result_name = 'Crowded'     # Empty/Occupied/Crowded
#     return render_template('index.html', name=class_name, result=result_name, precision=class_precision)

if __name__ == '__main__':
    app.run()
