# Flight Fare Prediction
## About the Dataset
- Airline: This column contains the name of the arilines. 
- Date of Journey: This column contains Date of the journey.  
- Source: This column contains from where the passenger is travelling. 
- Destination: This column contains to where the passenger is travelling.
- Route: This column contains the route of the airline.
- Dep Time: This column contains the deparature time of the flight.
- Arrival Time: This column contains the arrival time of the flight.
- Duration: This column contains total duration of the flight time.
- Total Stops: This column contains the if any total number stops made before reaching the final destination.
- Additional Info: This column contains the additional info like baggage allowances etc.
- Price: This column contains the price or fare of the airlines the target variable.

## Code Explanation
### Data Exploration
```sh
# importing necessary liberies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

# import the dataset
df = pd.read_excel('Data_Train.xlsx')
df.head()
```
![head](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/head.PNG?raw=true)

The above fig shows the top 5 rows of the dataset. It contains the column 'Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops', 'Additional_Info'and 'Price'.

### Exploratory Data Analysis (EDA)
```sh
# Target variable: Price of tickets
sns.distplot(df['Price'])
plt.show()
```
![Hist](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/price_hist.PNG?raw=true)

The above plot is the distribution of price column. It is left skewed.

```sh
# count plot for Airlines
plt.figure(figsize=(12,6))
sns.countplot(df['Airline'], palette='Set3')
plt.title('Number of Aeroplanes', size=15)
plt.xticks(rotation=90)
plt.show()
```
![Airline](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/air_count.PNG?raw=true)

The above plot shows the count of airlines. Jet airways have got the most number of journey.

```sh
# count plot for Source
plt.figure(figsize=(10,6))
sns.countplot(df['Source'], palette='Set2')
plt.title('Source Count Plot', size=30)
plt.xticks(rotation=90)
plt.show()
```
![Source](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/source_count.PNG?raw=true)

The above plot shows the count of source i.e. from where the passengers are travelling. Delhi airport have got the highest number of journey followed by Kolkata and Bangalore.

```sh
# coount plot for Destination
plt.figure(figsize=(10,6))
sns.countplot(df['Destination'], palette='Set2')
plt.title('Destination Count Plot', size=30)
plt.xticks(rotation=90)
plt.show()
```
![Destination](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/destination_count.PNG?raw=true)

The above plot shows the count of destination i.e. to where the passengers are travelling. Cochin airport have got the highest number of journey followed by Delhi.

```sh
# count plot for total stops of airline
plt.figure(figsize=(6,4))
sns.countplot(df['Total_Stops'], palette='Set2')
plt.title('Total Stop Count Plot', size=15)
plt.xticks(rotation=90)
plt.show()
```
![Total Stops](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/stop_count.PNG?raw=true)

The above plot shows the Total stop count of the airlines. 1 stop airlines are maximum while 4 stops are the least.

```sh
# count plot for additional info
plt.figure(figsize=(6,4))
sns.countplot(df['Additional_Info'], palette='Set2')
plt.title('Additional Info Count Plot', size=15)
plt.xticks(rotation=90)
plt.show()
```
![Add Info](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/adInfo_count.PNG?raw=true)

This fig show the count of additional info for the passengers like in-flight meal, baggage checkin etc. There is not much of variance into column where 'no info' have had the maximum count. 

```sh
# bar plot with default statistic=mean
plt.figure(figsize=(12,8))
sns.barplot(x='Price', y='Airline', data=df)
plt.title('Price vs Airline', size=20)
plt.show()
```
![Price vs Airline](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/airP_bar.png?raw=true)

The above plot shows the comparision between price and airline. Jet Airway have got the maximum also they had the maximum airlines as well as seen in the plot above section.

### Model Building
```sh
# Feature Selection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# Dividing the X & y features
X = df.drop(['Price'],axis=1)
y = df.Price

# Splitting into train and test with 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model trainning for feature selection
model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X_train,y_train) # fitting the trained model 
features_selected = X_train.columns[model.get_support()] # getting the important features

# All features selected except Year
X_train = X_train.drop(['Year'],axis=1)
X_test = X_test.drop(['Year'],axis=1)
```
The work of the codes above are to select the best fearures which will be helpful for building the model. `SelectFromModel; Lasso` what it does is build a model from which we get the top features. 

```sh
# Random Forest Regressor model with default hyperparameter
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(X_train,y_train)
```
RandomForestRegressor() is called assigned to reg variable. Here we are building the final model with top selected from features. 

```sh
from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred=reg.predict(X_test)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
```
Evaluating the model with y_test and matrix used are Mean Absolute Error and Root Mean Square Error. The values retures are:-
1. MAE: 1239.051
2. RMSE: 2206.635

The model is doing a decent job and it will predict near accurately. 

```sh
# creating pickle file for model deployment
pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
```
The pickle file is being created which will be used by the deployment file (app.py) during routing for prediction.

### Deployment
The above coding is done using the Jupyter Notebook however for deployment we will be shifting to PyCharm since we need to create a web application and a web server needs to be run for capturing the request from the user for prediction.

```sh
import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    ##For rendering result on HTML interface
    if request.method=='POST':
        features = [x for x in request.form.values()]
        source_dict = {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Kolkata': 3, 'Mumbai': 4}
        destination_dict = {'Bangalore':0,'Cochin':1,'Delhi':2,'Kolkata': 3,'Hyderabad':4,'New Delhi':5}
        airline_dict = {'IndiGo': 3, 'Air India': 1, 'Jet Airways': 4, 'SpiceJet': 8, 'Multiple carriers': 6, 'GoAir': 2, 'Vistara': 10, 'Air Asia': 0, 'Vistara Premium economy': 11, 'Jet Airways Business': 5, 'Multiple carriers Premium economy': 7, 'Trujet': 9}
        source_value = features[0]
        dest_value = features[1]
        date_value = features[2]
        airline_value = features[3]
        
        stops_value = int(features[4])   

        a= pd.Series(source_value)
        source = a.map(source_dict).values[0]   
        b= pd.Series(dest_value)
        destination = b.map(destination_dict).values[0] 
        c= pd.Series(airline_value)
        airline = c.map(airline_dict).values[0]   

        day = int(pd.to_datetime(date_value, format="%Y-%m-%dT%H:%M").day)    
        month = int(pd.to_datetime(date_value, format="%Y-%m-%dT%H:%M").month)  

        hour = int(pd.to_datetime(date_value, format ="%Y-%m-%dT%H:%M").hour)
        minute = int(pd.to_datetime(date_value, format ="%Y-%m-%dT%H:%M").minute)
           
        if  source==destination:
            return render_template('index.html',pred='Source and Destination City cannot be same. Please try again! ')
            
        else:
            pred_features = [np.array([day,month,stops_value,hour,minute,airline,source,destination])]
            prediction = model.predict(pred_features)

            if stops_value==0:
                output = round(prediction[0],0)

            else:
                output = round(prediction[0],0)-2000


            return render_template('index.html',pred='The Flight Fare for the given date is:-  INR {}'.format(output))
    else:
        return render_template('index.html')
if __name__=='__main__':
    app.run(debug=True)
```
##### _Outout_
![Out 1](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/out1.PNG?raw=true)

The above screenshot is of the home page of our model in Heroku where it is being deployed.

![Out 1](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/out2.PNG?raw=true)

The above figure shows the predicted output after clicking the predict button.










