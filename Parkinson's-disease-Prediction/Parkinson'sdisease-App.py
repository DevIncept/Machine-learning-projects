import numpy as np
from flask import Flask , request , jsonify , render_template
import requests
import sklearn
import pickle
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
model = pickle.load(open('parkinson_model.pkl','rb'))
@app.route('/',methods=['GET'])
def Home():
     return render_template('index.html')
 

scaler =MinMaxScaler()
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        feature_1= float(request.form['MDVP:Fo(Hz)'])
        feature_2=float(request.form['MDVP:Fhi(Hz)'])
        feature_3=float(request.form['MDVP:Flo(Hz)'])
        feature_4=float(request.form['MDVP:Jitter(%)'])
        feature_5=float(request.form['MDVP:Jitter(Abs)'])
        feature_6=float(request.form['MDVP:RAP']) 
        feature_7=float(request.form['MDVP:PPQ']) 
        feature_8=float(request.form['Jitter:DDP'])
        feature_9=float(request.form['MDVP:Shimmer'])
        feature_10=float(request.form['MDVP:Shimmer(dB)'])
        feature_11=float(request.form['Shimmer:APQ3'])
        feature_12=float(request.form['Shimmer:APQ5'])
        feature_13=float(request.form['MDVP:APQ'])
        feature_14=float(request.form['Shimmer:DDA'])
        feature_15=float(request.form['NHR'])
        feature_16=float(request.form['HNR'])
        feature_17=float(request.form['RPDE'])
        feature_18=float(request.form['DFA'])
        feature_19=float(request.form['spread1'])
        feature_20=float(request.form['spread2'])
        feature_21=float(request.form['D2'])
        feature_22=float(request.form['PPE'])
        
        values = np.array([[feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,feature_12,feature_13,feature_14,feature_15,feature_16,feature_17,feature_18,feature_19,feature_20,feature_21,feature_22]])
        
   
        prediction=model.predict(values) 
        
           
        if prediction == 1:
            return render_template('index.html', prediction_text='Result-Postitive')
       
        else:
            return render_template('index.html', prediction_text='Result-Negative')
    
    
    else:
        return render_template('index.html')
    

if __name__ == "__main__":
    app.run(debug=True)
