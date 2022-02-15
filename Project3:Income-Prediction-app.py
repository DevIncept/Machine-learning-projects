# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 15:54:50 2021

@author: utsav gada
"""
from flask import Flask, render_template, request
import os
import jsonify
import requests
import pickle
import numpy as np
from sklearn import *
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('incomeprediction.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['age'])
        Workclass = str(request.form['workclass'])
        Education = str(request.form['education'])
        Educationnum = int(request.form['educationnum'])
        Maritalstatus = str(request.form['maritalstatus'])
        Occupation = str(request.form['occupation'])
        Relationship = str(request.form['relationship'])
        Race = str(request.form['race'])
        Sex = str(request.form['sex'])
        Capitalgain = int(request.form['capitalgain'])
        Capitalloss = int(request.form['capitalloss'])
        Hoursperweek = int(request.form['hoursperweek'])
        Nativecountry = str(request.form['nativecountry'])
        prediction=model.predict([[Age,Workclass,Education,Educationnum,Maritalstatus,Occupation,Relationship,Race,Sex,Capitalgain,Capitalloss,Hoursperweek,Nativecountry]])
        output=prediction[0]
        print(output)
        if output == 0:
            return render_template('index.html',prediction_text="Income is Less than or equal to 50,000")
        else:
            return render_template('index.html',prediction_text="Income is Greater than 50,000")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
