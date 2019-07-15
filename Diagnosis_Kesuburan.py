# Soal 1 - Diagnosis Kesuburan
# =============================

import numpy as np
import pandas as pd

# read the csv file
data = pd.read_csv(
    'fertility.csv',
)

data = data.drop(['Season'], axis='columns')
# print(data)

# ======================
# USING ONE HOT ENCODER
# ======================

# labelling 
from sklearn.preprocessing import LabelEncoder
label1 = LabelEncoder()
label2=LabelEncoder()
label3=LabelEncoder()
label4=LabelEncoder()
label5=LabelEncoder()
label6=LabelEncoder()
label7=LabelEncoder()

data['Childish diseases'] = label1.fit_transform(data['Childish diseases'])
# print(label1.classes_)
# ['no' 'yes']

data['Accident or serious trauma'] = label2.fit_transform(data['Accident or serious trauma'])
# print(label2.classes_)
# ['no' 'yes']

data['Surgical intervention'] = label3.fit_transform(data['Surgical intervention'])
# print(label3.classes_)
# ['no' 'yes']

data['High fevers in the last year'] = label4.fit_transform(data['High fevers in the last year'])
# print(label4.classes_)
# ['less than 3 months ago' 'more than 3 months ago' 'no']

data['Frequency of alcohol consumption'] = label5.fit_transform(data['Frequency of alcohol consumption'])
# print(label5.classes_)
# ['every day' 'hardly ever or never' 'once a week' 'several times a day' 'several times a week']

data['Smoking habit'] = label6.fit_transform(data['Smoking habit'])
# print(label6.classes_)
# ['daily' 'never' 'occasional']

dataTarget = data.pop('Diagnosis')      # making 'Diagnosis' the target and remove it from the data
dataTarget = label7.fit_transform(dataTarget)

# ====================================================================
# ONE HOT ENCODER
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [4, 5, 6])],
    remainder='passthrough'
)
dataOneHot = coltrans.fit_transform(data)

# splitting training testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    dataOneHot,
    dataTarget,
    test_size = .1
)

# ====================================================================
# applying 3 different machine learning algorithms

# logistic regression
from sklearn.linear_model import LogisticRegression
modelLogistic = LogisticRegression(solver='liblinear', multi_class='auto')
modelLogistic.fit(xtrain, ytrain)

# kmeans
from sklearn.cluster import KMeans
modelKmeans = KMeans(n_clusters = len(label7.classes_))
modelKmeans.fit(xtrain, ytrain)

# random forest classifier
from sklearn.ensemble import RandomForestClassifier
modelRandom = RandomForestClassifier()
modelRandom.fit(xtrain, ytrain)

def target(prediksi):
    if prediksi[0]==0:
        return (label7.classes_[prediksi[0]]).upper()
    elif prediksi[0]==1:
        return (label7.classes_[prediksi[0]]).upper()       

# ======================================================================
# based on the predictions story from 5 different profiles

arin = [[0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]]
bebi = [[0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]]
caca = [[1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]]
dini = [[0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]]
enno = [[0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]]

# ======================================================================
# printing out the results

print('Arin, prediksi kesuburan: ',target(modelLogistic.predict(arin)),'(Logistic Regression)')
print('Arin, prediksi kesuburan: ',target(modelKmeans.predict(arin)),'(K-Means)')
print('Arin, prediksi kesuburan: ',target(modelRandom.predict(arin)),'(Random Forest Classifier)')

print('\nBebi, prediksi kesuburan: ',target(modelLogistic.predict(bebi)),'(Logistic Regression)')
print('Bebi, prediksi kesuburan: ',target(modelKmeans.predict(bebi)),'(K-Means)')
print('Bebi, prediksi kesuburan: ',target(modelRandom.predict(bebi)),'(Random Forest Classifier)')

print('\nCaca, prediksi kesuburan: ',target(modelLogistic.predict(caca)),'(Logistic Regression)')
print('Caca, prediksi kesuburan: ',target(modelKmeans.predict(caca)),'(K-Means)')
print('Caca, prediksi kesuburan: ',target(modelRandom.predict(caca)),'(Random Forest Classifier)')

print('\nDini, prediksi kesuburan: ',target(modelLogistic.predict(dini)),'(Logistic Regression)')
print('Dini, prediksi kesuburan: ',target(modelKmeans.predict(dini)),'(K-Means)')
print('Dini, prediksi kesuburan: ',target(modelRandom.predict(dini)),'(Random Forest Classifier)')

print('\nEnno, prediksi kesuburan: ',target(modelLogistic.predict(enno)),'(Logistic Regression)')
print('Enno, prediksi kesuburan: ',target(modelKmeans.predict(enno)),'(K-Means)')
print('Enno, prediksi kesuburan: ',target(modelRandom.predict(enno)),'(Random Forest Classifier)')