# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:50:45 2022

@author: hp
"""
#web: gunicorn tushar:app
#creating flask environment

import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle

#loading model 

model = pickle.load(open("model.pkl","rb"))
#print(model)


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('application.html')

@app.route("/predict",methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('application.html', prediction_text='Employee Salary should be $ {}'.format(output))
    
    

if __name__ == "__main__":
    app.run(debug=True)
