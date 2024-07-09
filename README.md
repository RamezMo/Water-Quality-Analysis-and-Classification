# Water_Quality-Predictor-Model : Machine Learning Model for Forecasting Water_Quality with nearly 82.45%% accuracy

## Introduction

Ensuring safe water sources is critical for public health and environmental sustainability. This project leverages machine learning techniques to predict water safety using the Water Quality Dataset from Kaggle. By analyzing parameters such as barium levels, ammonia, aluminium, and presence of bacteria, this model aims to provide proactive insights for environmental agencies and communities. Detecting potential water quality issues early can enable timely interventions and safeguard public health effectively.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation](#Data-Preparation)
3. [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis) 
4. [Model Evaluation](#Evaluate-models)

# Data Preparation

### Importing Necessary Libraries

First, import the necessary libraries for data analysis and machine learning.

```python
#Importing the Needed Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
```

## Load dataset using its full path
Load the dataset Load the dataset into DataFrame.

```python
water_data = pd.read_csv("/kaggle/input/waterquallity/waterQuality1.csv")
```

## Display the first 10 rows of the dataset
inspect the first few rows to understand its structure.
```python
water_data.head(10)
```

![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-09%20201935.png)


## How Many Instances and Features ?
this Display the number of rows and Columns the dataset have
```python
water_data.shape
```

#Exploratory-Data-Analysis


##Display Variables DataType and count of non NULLs values in

```python
water_data.info()
```
![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-09%20202340.png)

All variables DataType are Numerical except ammonia & is_safe we will handle later


## Checking for missing values
Check for and handle missing values to ensure a clean dataset.
```python
water_data.isnull().sum()
```

![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-09%20202530.png)


## Summary statistics

```python
water_data.describe()
```

![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-09%20205507.png)


## Calculate the correlation matrix

```python
corr=water_data.corr()
```

## Create a correlation heatmap for the subset of features

```python
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True,mask = np.triu(np.ones_like(corr, dtype=bool)))
```
![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-09%20205746.png)


## Distribution of is_safe Column values
Notice it is not balanced so we will work on this problem
```python
print(water_data.is_safe.value_counts())
sns.countplot(data=water_data , x = 'is_safe')
```
![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-10%20021211.png)
It Shows that there is an imbalance in this column


#data preperation
## Data Transformation
Convert categorical variables into numerical ones for machine learning models.
there is a problem in ammonia column that it have numbers as a string DataType 'it is easy to handle' but if it also have multiple Values as a string and it contain Characters?

```python
# Filter out non-numeric strings and convert to numeric
water_data['ammonia'] = water_data['ammonia'].apply(pd.to_numeric, errors='coerce')         

#use to_numeric function that convert numbers with str type into Numbers and if it can not then this is a real str value so coerce make sure to convert these values into NaN to handle it easily

# Drop rows where 'ammonia' could not be converted to numeric 'NULLs'
water_data.dropna(subset=['ammonia'], inplace=True)
```

for is_safe column : it have 0 & 1 values but as a string so just convert it to Int
```python
water_data.is_safe = water_data.is_safe.replace({'0':0,'1':1})
```

##Handling The Imbalance of is_safe values
There are Multiple Methods to Overcome this Problem like OverSample using SMOTE but let's just take a sample from the Most presence Value

```python
balanced_data_category_0 = water_data[water_data['is_safe'] == 0].sample(n=912, random_state=42)

# Combine sampled category 0 rows with category 1 rows
balanced_data = pd.concat([balanced_data_category_0, water_data[water_data['is_safe'] == 1]])
```

##Detect Outliers
Outliers are data points that deviate significantly from other observations in a dataset, potentially impacting statistical analyses and model performance by skewing results or introducing noise.handling outliers is critical to ensure data integrity and reliable analytical outcomes.
Use BoxPlot to See if we have Outliers
```python
# Plot boxplot to visualize distribution of features
plt.figure(figsize=(15, 10))
sns.boxplot(data=balanced_data)
plt.title('Boxplot of Features')
plt.xticks(rotation=45)
plt.show()
```

![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-10%20022840.png)

We Have outliers in arsenic and nitrites Columns


##Handle Outliers
```python
#Let's Remove outliers
 #Calculate IQR for arsenic column
Q1 = balanced_data['arsenic'].quantile(0.25)
Q3 = balanced_data['arsenic'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 0.7 * IQR

# Create mask for outliers
outlier_mask = (balanced_data['arsenic'] < lower_bound) | (balanced_data['arsenic'] > upper_bound)

# Remove outliers
balanced_data_no_outliers = balanced_data[~outlier_mask]

# Visualize box plot after outlier removal
plt.figure()
sns.boxplot(balanced_data_no_outliers['arsenic'])
plt.title('Arsenic Column After Removing Outliers')
plt.show()

#Let's Remove outliers
 #Calculate IQR for arsenic column
Q1 = balanced_data_no_outliers['nitrites'].quantile(0.25)
Q3 = balanced_data_no_outliers['nitrites'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create mask for outliers
outlier_mask = (balanced_data_no_outliers['nitrites'] < lower_bound) | (balanced_data_no_outliers['nitrites'] > upper_bound)

# Remove outliers
balanced_data_no_outliers_final = balanced_data_no_outliers[~outlier_mask]

# Visualize box plot after outlier removal
plt.figure()
sns.boxplot(balanced_data_no_outliers_final['nitrites'])
plt.title('nitrites Column After Removing Outliers')
plt.show()

```
![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-10%20024739.png)

![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-10%20024746.png)



## Split data into training and testing sets

```python
x=balanced_data_no_outliers_final.drop('is_safe',axis=1)
y=balanced_data_no_outliers_final.is_safe
```


## Evaluate models
after training the model and predicting it on test data it makes accuracy of nearly 82.45%

##Creating Confusion Matrix
it shows the predicted values Distribution

```python
cm = confusion_matrix(y_test, predicted_testing)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
```
![image](https://raw.githubusercontent.com/RamezMo/Water-Quality-Analysis-and-Classification/main/Screenshot%202024-07-10%20025017.png)

