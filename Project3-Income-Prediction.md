# Income Prediction

The project focuses on predicting the income range based on some parameters
which are not based on skills rather than background and environment Even though 
the person is working the results can provide a lot of insights.

![](https://github.com/Sara-cos/Intern-Projects/blob/main/int%20ml-5/Income-Prediction/static/Retrain%20Model.jpg)

​
Whoever apply for a job or the new candidates or even any person who is 
curious to know the range of income possibilities regardless of the skills did not get the idea of the 
same. The provision revolves around the skills and education. Going through the data present 
globally its observed that there is some kind of pattern in the income. Using this pattern, a person 
can know the possible range of income one may expect. The income may or may not match the actual income but its interesting to provide insights over the possibilities using the patterns already 
observed.

## About the Dataset:

Using the dataset with around 33000 inputs from around the global with about 14 
parameters. Using these inputs, the model is trained to predict the possible income ranges. After 
some cleaning the model is trained. The insights are drawn and pulled up to understand how the 
pattern works.
​
The dataset is present in the [repository](https://github.com/Sara-cos/Income_Prediction).

### Attributes of the Dataset:

* Age: Describes the age of individuals. Continuous.

* Workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.

* fnlwgt: Continuous.

* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

* education-num: Number of years spent in education. Continuous.

* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

* relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

* sex: Female, Male.

* capital-gain: Continuous.

* capital-loss: Continuous.

* hours-per-week: Continuous.

* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

* salary: >50K,<=50K

### lets take a look at first 5 rows of our Train dataset:


```
data.head()
```
![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/Screenshot%20(31).png)


### Lets get some information of our datset

```
data.info()
```
![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/info.png)

#### We Can make the following observations from above information:

* The dataset contains absolutely no null values!

* Age, Final Weight, Education Number, Capital Gain, Capital Loss and Hours Per Week are integer columns.

* There are no Float Datatypes in the dataset.

* Workclass, Education, Marital Status, Occupation, Relationship, Race, Sec, Native Country and Income are of object datatypes.

* Dataset contains no null values.

### Lets describe the dataset:

```
data.describe()
```
![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/describe.png)

### Observation:

* The minimum and maximum age of people in the dataset is 19 and 90 years respectively, while the average age is 37.

* The minimum and maximum years spent on education is 1 and 16 respectively, whereas the mean education level is 10 years.

* While the minimum and average capital gain is 0, maximum is 99999. 

* The number of hours spent per week varies between 1 to 99 and the average being 40 hours.

### Exploratory Data Analysis of our Dataset:

![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/1.png)

* Private Workclass are paid more than any other Workclass.

![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/2.png)

* Adults in Exec-managerial role are equally likely to earn more than 50K dollars an year.

* Adults working in Farming-fishing, Machine-op-inspect, Other-service, Adm-clerical, Transport-moving are very less likely to earn more than 50K dollars an year.

![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/3.png)

* Married-civ-spouse are paid more than 50k dollars.

* Never married are paid less than 50k Dollars.

![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/4.png)

* Wives get more than 50K dollars.

* For Husbands earn less than 50K dollars.

* Many unmarried people get more than 50k dollars.

![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/5.png)

* Male employees are paid more than Female employees.

![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/6.png)

* White people get to earn more than Black people.

### Converting all Catagorical Values to Numerical Values:

Converting objects to int type.

### Model Implementation:

We will be using Decision Tree Classifier to build the Model.

### About Decision Tree Classifier:

Decision Tree can be used for both classification and Regression models.But mostly used for Classification Problem.
Here we have used Decision Tree as Classification algorithm.
It is Supervised Learning Technique.

```
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
```



#### Accuracy of Decision Tree classifier on training set: 0.98
#### Accuracy of Decision Tree classifier on test set: 0.82

### Finally Predicting Actual Test Values

### Creating a Pickle File:

```
import pickle
file = open('incomeprediction.pkl','wb')
#dump information to that file
pickle.dump(clf, file)

```

### Deployment of the Model:

We used Heroku and deployed our model using flask...

![](https://github.com/Sara-cos/Intern-Projects/blob/main/int%20ml-5/Income-Prediction/static/IP2.png)
![](https://github.com/Sara-cos/Intern-Projects/blob/main/int%20ml-5/Income-Prediction/static/IP3.png)

### Link Of the Application:

https://incomepredictions.herokuapp.com/
![](https://github.com/aishwaryaa-01/Income_Prediction/blob/main/Images/app.png)
