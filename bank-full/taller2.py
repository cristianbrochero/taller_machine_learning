import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

url = 'bank-full.csv'

#DATASET BANK FULL

data = pd.read_csv(url)

#Remplazos
data.default.replace(['no','yes'], [0,1], inplace= True)
data.job.replace(['blue-collar','management','technician','admin.','services','retired','self-employed','entrepreneur','unemployed','housemaid','student','unknown'], [0,1,2,3,4,5,6,7,8,9,10,11], inplace= True)
data.marital.replace(['married','single','divorced'], [0,1,2], inplace= True)
data.education.replace(['secondary','tertiary','primary','unknown'], [0,1,2,3], inplace= True)
data.housing.replace(['no','yes'], [0,1], inplace= True)
data.loan.replace(['no','yes'], [0,1], inplace= True)
data.contact.replace(['cellular','unknown','telephone'], [0,1,2], inplace= True)
data.poutcome.replace(['unknown','failure','other','success'], [0,1,2,3], inplace= True)
data.y.replace(['no','yes'], [0,1], inplace= True)

rangos = [18,25,40,60,100]
nombres = ['1','2','3','4',]
data.age = pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0,how='any', inplace=True)

#Columnas Innecesarias
data.drop(['balance', 'day', 'month', 'duration','campaign','pdays','previous'], axis= 1, inplace = True)

#Dividir Data
data_train = data[:30000]
data_test = data[30000:]
x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)


#REGRESION LOGISTICA
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
logreg.fit(x_train,y_train)
print('*'*50)
print('Regresión Logística')
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

