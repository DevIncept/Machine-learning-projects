#                   Detecting Parkinson’s Disease – Machine Learning Project

### What is Parkinson’s Disease?

Parkinson’s disease is a progressive disorder of the central nervous system affecting movement and inducing tremors and stiffness. It has 5 stages to it and affects more than 1 million individuals every year in India. This is chronic and has no cure yet. It is a neurodegenerative disorder affecting dopamine-producing neurons in the brain.

### What is XGBoost?

XGBoost is a new Machine Learning algorithm designed with speed and performance in mind. XGBoost stands for eXtreme Gradient Boosting and is based on decision trees. In this project, we will import the XGBClassifier from the xgboost library; this is an implementation of the scikit-learn API for XGBoost classification.

### Detecting Parkinson’s Disease with XGBoost – Objective

To build a model to accurately detect the presence of Parkinson’s disease in an individual.

### Detecting Parkinson’s Disease with XGBoost – About the Machine Learning Project

In this Python machine learning project, using the Python libraries scikit-learn, numpy, pandas, and xgboost, we will build a model using an XGBClassifier. 
We’ll load the data, get the features and labels, scale the features, then split the dataset, build an XGBClassifier, and then calculate the accuracy of our model.

# Prerequisites

You’ll need to install the following libraries with pip:


```python
pip install numpy pandas sklearn xgboost
```

    Requirement already satisfied: numpy in c:\users\hemac\anaconda3\lib\site-packages (1.21.0)
    Requirement already satisfied: pandas in c:\users\hemac\anaconda3\lib\site-packages (1.2.5)
    Requirement already satisfied: sklearn in c:\users\hemac\anaconda3\lib\site-packages (0.0)
    Requirement already satisfied: xgboost in c:\users\hemac\anaconda3\lib\site-packages (1.4.2)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\users\hemac\anaconda3\lib\site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in c:\users\hemac\anaconda3\lib\site-packages (from pandas) (2021.1)
    Requirement already satisfied: six>=1.5 in c:\users\hemac\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas) (1.16.0)
    Requirement already satisfied: scikit-learn in c:\users\hemac\anaconda3\lib\site-packages (from sklearn) (0.24.2)
    Requirement already satisfied: scipy in c:\users\hemac\anaconda3\lib\site-packages (from xgboost) (1.7.0)
    Requirement already satisfied: joblib>=0.11 in c:\users\hemac\anaconda3\lib\site-packages (from scikit-learn->sklearn) (1.0.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\hemac\anaconda3\lib\site-packages (from scikit-learn->sklearn) (2.1.0)
    Note: you may need to restart the kernel to use updated packages.
    

    WARNING: You are using pip version 21.2.1; however, version 21.2.2 is available.
    You should consider upgrading via the 'C:\Users\hemac\Anaconda3\python.exe -m pip install --upgrade pip' command.
    

###### You’ll also need to install Jupyter jupyter notebook, and then use the command prompt to run it:

Open command prompt and run Pip install jupyter notebook
And open it by running jupyter notebook in cmd

This will open a new Jupyter notebook in your browser. Here, you will create a new console and type in your code, then press Shift+Enter to execute one or more lines at a time.

### Steps for Detecting Parkinson’s Disease with XGBoost

1. Make necessary imports:


```python
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

2. Now, let’s read the data into a DataFrame and get the first 5 records.


```python
# Read the data
df=pd.read_csv('parkinsons.data')
df.head()
```




<div>


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>MDVP:Fo(Hz)</th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>...</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>status</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>phon_R01_S01_1</td>
      <td>119.992</td>
      <td>157.302</td>
      <td>74.997</td>
      <td>0.00784</td>
      <td>0.00007</td>
      <td>0.00370</td>
      <td>0.00554</td>
      <td>0.01109</td>
      <td>0.04374</td>
      <td>...</td>
      <td>0.06545</td>
      <td>0.02211</td>
      <td>21.033</td>
      <td>1</td>
      <td>0.414783</td>
      <td>0.815285</td>
      <td>-4.813031</td>
      <td>0.266482</td>
      <td>2.301442</td>
      <td>0.284654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phon_R01_S01_2</td>
      <td>122.400</td>
      <td>148.650</td>
      <td>113.819</td>
      <td>0.00968</td>
      <td>0.00008</td>
      <td>0.00465</td>
      <td>0.00696</td>
      <td>0.01394</td>
      <td>0.06134</td>
      <td>...</td>
      <td>0.09403</td>
      <td>0.01929</td>
      <td>19.085</td>
      <td>1</td>
      <td>0.458359</td>
      <td>0.819521</td>
      <td>-4.075192</td>
      <td>0.335590</td>
      <td>2.486855</td>
      <td>0.368674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phon_R01_S01_3</td>
      <td>116.682</td>
      <td>131.111</td>
      <td>111.555</td>
      <td>0.01050</td>
      <td>0.00009</td>
      <td>0.00544</td>
      <td>0.00781</td>
      <td>0.01633</td>
      <td>0.05233</td>
      <td>...</td>
      <td>0.08270</td>
      <td>0.01309</td>
      <td>20.651</td>
      <td>1</td>
      <td>0.429895</td>
      <td>0.825288</td>
      <td>-4.443179</td>
      <td>0.311173</td>
      <td>2.342259</td>
      <td>0.332634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>phon_R01_S01_4</td>
      <td>116.676</td>
      <td>137.871</td>
      <td>111.366</td>
      <td>0.00997</td>
      <td>0.00009</td>
      <td>0.00502</td>
      <td>0.00698</td>
      <td>0.01505</td>
      <td>0.05492</td>
      <td>...</td>
      <td>0.08771</td>
      <td>0.01353</td>
      <td>20.644</td>
      <td>1</td>
      <td>0.434969</td>
      <td>0.819235</td>
      <td>-4.117501</td>
      <td>0.334147</td>
      <td>2.405554</td>
      <td>0.368975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>phon_R01_S01_5</td>
      <td>116.014</td>
      <td>141.781</td>
      <td>110.655</td>
      <td>0.01284</td>
      <td>0.00011</td>
      <td>0.00655</td>
      <td>0.00908</td>
      <td>0.01966</td>
      <td>0.06425</td>
      <td>...</td>
      <td>0.10470</td>
      <td>0.01767</td>
      <td>19.649</td>
      <td>1</td>
      <td>0.417356</td>
      <td>0.823484</td>
      <td>-3.747787</td>
      <td>0.234513</td>
      <td>2.332180</td>
      <td>0.410335</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



3. Get the features and labels from the DataFrame (dataset). The features are all the columns except ‘status’, and the labels are those in the ‘status’ column.


```python
# Get the features and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values
```

4. The ‘status’ column has values 0 and 1 as labels; let’s get the counts of these labels for both- 0 and 1.


```python
#Get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
```

    147 48
    

We have 147 ones and 48 zeros in the status column in our dataset.

5. Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them. The MinMaxScaler transforms features by scaling them to a given range. The fit_transform() method fits to the data and then transforms it. We don’t need to scale the labels.




```python
#DataFlair - Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels
```

6. Now, split the dataset into training and testing sets keeping 20% of the data for testing.


```python
# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
```

7. Initialize an XGBClassifier and train the model. This classifies using eXtreme Gradient Boosting- using gradient boosting algorithms for modern data science problems. It falls under the category of Ensemble Learning in ML, where we train and predict using many models to produce one superior output.


```python
# Train the model
model=XGBClassifier()
model.fit(x_train,y_train)
```

    [10:54:28] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    

    C:\Users\hemac\Anaconda3\lib\site-packages\xgboost\sklearn.py:1146: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)



8. Finally, generate y_pred (predicted values for x_test) and calculate the accuracy for the model. Print it out.


```python
#  Calculate the accuracy
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
```

    94.87179487179486
    

## Summary

In this machine learning project, we learned to detect the presence of Parkinson’s Disease in individuals using various factors. We used an XGBClassifier for this and made use of the sklearn library to prepare the dataset. This gives us an accuracy of 94.87%, which is great considering the number of lines of code in this python project.

Hope you enjoyed this project.
