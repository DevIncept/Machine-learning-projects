# Flight Fare Prediction deployment using Flask in Heroku
## Problem Statement
The aim of this project is to predict the fares of the flights based on different factors available in the dataset. The flight ticket prices either increase or decrease depending on various factors like timing of the flights, destination, and duration of flights etc.

## Description
- AIM:- Airway is the fastest mean of travelling long distance destinations and it has become an integral part of our life. The flight ticket prices either increase or decrease depending on various factors like timing of the flight, destination and durations of the flight etc. So to get an idea of the ticket price beforehand will surely help in planning the trip and save some money on travelling expenses. 
- Approach:- Data Exploration, Data Cleaning, Feature Engineering, Model Building & Testing and Deployment. 
- Alorithm:- Random Forest Regressor is used to build the model also Lasso is used for feature selection.
- Flowchart:-

![Flowchart](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/Flowchart.png?raw=true)

## Project Structure
In the deployment folder the project have got four major parts:-
1. Flight_Fare_Prediction_Model - This file contains the code of our trained ML model to predict flight fare based on the data in excel file name 'Data_Train.xlsx'.
2. app.py - This file contains the deployment codes and the API of Flask.
3. templates - This folder contains the HTML template to allow user to enter flight detail and displays the predicted flight fare basically the webpage.
4. requirement.txt - This files contains the all the required necessary packages used by the model during the training process and its very important file used by Heroku app.

### Installation
First clone the repository then run the command to download:-
```sh
pip install -r requirements.txt
```
This will install all the required liberies used by the model.

### Running the project
1. Execute the file name `Flight_Fare_Prediction_Model`, its a Jupyter notebook where the machine learning model is being trained, this would create a serialized version of our model into a file model.pkl.
> Please ensure that you are in the clonned directory. 

2. Run app.py file using Anaconda prompt to start Flask API using the command line
`python app.py`. It will navigate to URL http://127.0.0.1:8000/ this is the local host.  

After running the URL in brower the home page will be displayed like the one below:
![Home](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/out1.PNG?raw=true)

## Deployment in Heroku
For deploying in Heroku first we need to login to the platform. After that connect the GitHub account to Heroku. Then go to new section and create an app. After that go to deploy section and search for the repository in the search bar but ensure that Github account is connected. 
<br>
Select the deployment depository and click on connect. After connections are made click on deploy, this will automatically install all the necessary files and run the model. After successful compilation URL will be generated.
![Heroku](https://github.com/snozh5/temp/blob/main/Pic_Flight_Pred/heroku.PNG?raw=true)

