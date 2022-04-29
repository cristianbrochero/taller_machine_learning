import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

url = 'diabetes.csv'
data = pd.read_csv(url)
rangos = [0,6,12,17]
nombres = ['1','2','3']
data.Pregnancies = pd.cut(data.Pregnancies, rangos, labels=nombres)
data.Pregnancies.replace(np.nan, 1, inplace=True)

rangos2 = [0,50,100,150,200]
nombres2 = ['1','2','3','4']
data.Glucose = pd.cut(data.Glucose, rangos2, labels=nombres2)

rangos3 = [0,50,100,150]
nombres3 = ['1','2','3']
data.BloodPressure = pd.cut(data.BloodPressure, rangos3, labels=nombres3)

rangos4 = [-1,200,500,700,900]
nombres4 = ['1','2','3','4']
data.Insulin = pd.cut(data.Insulin, rangos4, labels=nombres4)

rangos5 = [-1,20,40,60]
nombres5 = ['1','2','3']
data.BMI = pd.cut(data.BMI, rangos5, labels=nombres5)

rangos6 = [20,40,60,90]
nombres6 = ['1','2','3']
data.Age = pd.cut(data.Age, rangos6, labels=nombres6)

data.dropna(axis=0,how='any', inplace=True)


#Columnas Innecesarias
data.drop(['SkinThickness','DiabetesPedigreeFunction'], axis= 1, inplace = True)


# dividir la data
data_train = data[:450]
data_test = data[450:]
x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome) 

#REGRESION
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)
logreg.fit(x_train,y_train)
print('*'*50)
print('Regresión Logística')
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# SOPORTE VECTORIAL
svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
print('*'*50)
print('Maquina de soporte vectorial')
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# ARBOL DE DECISIO
arbol = DecisionTreeClassifier()
arbol.fit(x_train, y_train)
print('*'*50)
print('Decisión Tree')
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')

# RANDOM FOREST
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
print('*'*50)
print('RANDOM FOREST')
print(f'accuracy de Entrenamiento de Entrenamiento: {forest.score(x_train, y_train)}')
print(f'accuracy de Test de Entrenamiento: {forest.score(x_test, y_test)}')
print(f'accuracy de Validación: {forest.score(x_test_out, y_test_out)}')


# NAIVE BAYES
nayve = GaussianNB()
nayve.fit(x_train, y_train)
print('*'*50)
print('NAYVE BAYES')
print(f'accuracy de Entrenamiento de Entrenamiento: {nayve.score(x_train, y_train)}')
print(f'accuracy de Test de Entrenamiento: {nayve.score(x_test, y_test)}')
print(f'accuracy de Validación: {nayve.score(x_test_out, y_test_out)}')