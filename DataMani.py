import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Set Dependent
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

TTData = [train, test]

# 0.71291
# Pclass cross Sex+Map
for dataset in TTData:
	dataset.loc[(dataset['Sex'] == "female") & (dataset['Pclass'] <= 2),'SexPclass'] = "F12"
	dataset.loc[(dataset['Sex'] == "female") & (dataset['Pclass'] == 3),'SexPclass'] = "F3" 
	dataset.loc[(dataset['Sex'] == "male") & (dataset['Pclass'] >= 2),'SexPclass'] = "M23" 
	dataset.loc[(dataset['Sex'] == "male") & (dataset['Pclass'] == 1),'SexPclass'] = "M1" 
Psex_map = { "F12":0, "F3":1, "M23":2, "M1":3}
for dataset in TTData:
	dataset['SexPclass'] = dataset['SexPclass'].map(Psex_map)

# Age Null
for dataset in TTData:
    AgeAvg = dataset['Age'].mean()
    AgeStd = dataset['Age'].std()
    Null = dataset['Age'].isnull().sum()
    RandAge = np.random.randint(AgeAvg - AgeStd, AgeAvg + AgeStd, size=Null)
    dataset['Age'][np.isnan(dataset['Age'])] = RandAge
    dataset['Age'] = dataset['Age'].astype(int)

# Age Sex # Need to fix this, smaller range for kids
for dataset in TTData:
	dataset.loc[(dataset['Age'] <= 7) ,'Adult'] = "A"
	dataset.loc[(dataset['Sex'] == "female") & (dataset['Age'] > 7) & (dataset['Age'] <= 18),'Adult'] = "B"
	dataset.loc[(dataset['Sex'] == "male") & (dataset['Age'] > 7) & (dataset['Age'] <= 18),'Adult'] = "C"
	dataset.loc[(dataset['Sex'] == "female") & (dataset['Age'] > 18) & (dataset['Age'] <= 48),'Adult'] = "D"
	dataset.loc[(dataset['Sex'] == "male") & (dataset['Age'] > 18) & (dataset['Age'] <= 48),'Adult'] = "E"
	dataset.loc[(dataset['Sex'] == "female") & (dataset['Age'] > 48) & (dataset['Age'] <= 64),'Adult'] = "F"
	dataset.loc[(dataset['Sex'] == "male") & (dataset['Age'] > 48) & (dataset['Age'] <= 64),'Adult'] = "G"
	dataset.loc[(dataset['Age'] > 64) ,'Adult'] = "H"

SAge_map = { "A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7}
for dataset in TTData:
	dataset['Adult'] = dataset['Adult'].map(SAge_map)
# Embarked Null+Map
for dataset in TTData: 
    dataset['Embarked']= dataset['Embarked'].fillna('S')
for dataset in TTData:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Fare Null + Map
for dataset in TTData:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

for dataset in TTData:
	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
	dataset['Fare'] = dataset['Fare'].astype(int)

# Combination of SibSp and Parch
for dataset in TTData:
    dataset['FamSz'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in TTData:
	dataset.loc[dataset['FamSz']==1, 'TravelFam'] = 0
	dataset.loc[(dataset['FamSz'] > 1) & (dataset['FamSz'] <= 4), 'TravelFam'] = 1
	dataset.loc[(dataset['FamSz'] > 4) & (dataset['FamSz'] <= 7), 'TravelFam'] = 2
	dataset.loc[(dataset['FamSz'] > 7 ), 'TravelFam'] = 3

# Dropping Unnecessary Features 
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamSz','Sex']
train = train.drop(features_drop, axis = 1)
test = test.drop(features_drop, axis = 1)
train = train.drop(['PassengerId'], axis=1)
