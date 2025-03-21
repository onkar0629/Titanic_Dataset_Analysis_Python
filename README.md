# Titanic Dataset Analysis

## Overview
This repository contains an analysis of the Titanic dataset using Python. The dataset provides information about the passengers aboard the Titanic, including whether they survived or not. The analysis includes various data exploration and preprocessing techniques.

## Dataset

[Kaggle](https://www.kaggle.com/c/titanic)

## Table of Contents
1. Display Top 5 Rows of The Dataset
2. Check the Last 3 Rows of The Dataset
3. Find Shape of Our Dataset (Number of Rows & Number of Columns)
4. Get Information About Our Dataset
5. Get Overall Statistics About The Dataframe
6. Data Filtering
7. Check Null Values In The Dataset
8. Drop the Column
9. Handle Missing Values
10. Categorical Data Encoding
11. Univariate Analysis
   - How Many People Survived And How Many Died?
   - How Many Passengers Were In First Class, Second Class, and Third Class?
   - Number of Male And Female Passengers
12. Bivariate Analysis
   - How Has Better Chance of Survival: Male or Female?
   - Which Passenger Class Has Better Chance of Survival?
13. Feature Engineering

## Analysis Details
### 1. Display Top 5 Rows of The Dataset
```python
import pandas as pd

df = pd.read_csv('train.csv')
df.head()
```

### 2. Check the Last 3 Rows of The Dataset
```python
df.tail(3)
```

### 3. Find Shape of Our Dataset
```python
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
```

### 4. Get Information About Our Dataset
```python
df.info()
```

### 5. Get Overall Statistics About The Dataframe
```python
df.describe()
```

### 6. Data Filtering
```python
df[df['Age'] > 30]  # Example: Filtering passengers older than 30
```

### 7. Check Null Values In The Dataset
```python
df.isnull().sum()
```

### 8. Drop the Column
```python
df.drop(columns=['Cabin'], inplace=True)
```

### 9. Handle Missing Values
```python
df.fillna({'Age': df['Age'].median()}, inplace=True)
```

### 10. Categorical Data Encoding
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
```

### 11. Univariate Analysis
#### How Many People Survived And How Many Died?
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Survived', data=df, palette=['red', 'green'])
plt.show()
```

#### How Many Passengers Were In First Class, Second Class, and Third Class?
```python
sns.countplot(x='Pclass', data=df, palette=['blue', 'orange', 'purple'])
plt.show()
```

#### Number of Male And Female Passengers
```python
sns.countplot(x='Sex', data=df, palette=['pink', 'blue'])
plt.show()
```

### 12. Bivariate Analysis
#### How Has Better Chance of Survival: Male or Female?
```python
sns.barplot(x='Sex', y='Survived', data=df)
plt.show()
```

#### Which Passenger Class Has Better Chance of Survival?
```python
sns.barplot(x='Pclass', y='Survived', data=df)
plt.show()
```

### 13. Feature Engineering
Feature engineering involves creating new meaningful features from existing ones to improve model performance. Some examples include:
- Creating a new feature `FamilySize` from `SibSp` and `Parch`
- Extracting titles from names
- Grouping fare and age into bins for better insights

## Conclusion
This analysis provides valuable insights into the Titanic dataset, focusing on survival rates, passenger classes, and other critical factors. Feature engineering and data preprocessing are crucial steps for further predictive modeling.

## Author

- Onkar Ankush Jadhav
  
---



