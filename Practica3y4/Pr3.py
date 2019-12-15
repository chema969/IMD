from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from scipy.stats import wilcoxon,friedmanchisquare,chi2

def crossval(train_input,train_output,modelo):
  scores = cross_val_score(modelo, train_input, np.ravel(train_output), cv=10)
  return scores

def leer_csv(fichero):
  import sklearn.preprocessing
  train=pd.read_csv(fichero,header=0)
  train=np.array(train)
  train = train.astype(np.float64)
  train_inputs=train[:,:-1]
  train_outputs=train[:,-1:]
  scaler = sklearn.preprocessing.StandardScaler()
  train_inputs = scaler.fit_transform(train_inputs)
  return train_inputs, train_outputs


def conseguir_modelos():
  knn=KNeighborsClassifier(10)
  
  svm=SVC(C=10, kernel='rbf',gamma='scale')
  tree=DecisionTreeClassifier()
  return [knn,svm,tree]
 
 
 
bases_de_datos=['diabetes.csv','glass.csv','ionosphere.csv','nomnist.csv','segment_challenge.csv','spam.csv','vote.csv','waves.csv','wine.csv','winequality.csv']

modelos=conseguir_modelos()

salida_CCR=np.zeros((np.size(bases_de_datos),np.size(modelos)),dtype=float)

aux1=0
for i in bases_de_datos:
  train_i,train_o=leer_csv(i)
  aux2=0
  for j in modelos:
    score=crossval(train_i,train_o,j)
    salida_CCR[aux1,aux2]=np.mean(score)
    aux2=aux2+1
  aux1=aux1+1
  
nombres_modelos=['KNN','SVM','Tree']
plt.rcParams["figure.figsize"] = (20,25)

fig, axs = plt.subplots(5, 2)
aux1=0
aux2=0
for i in salida_CCR:
  axs[aux1,aux2].bar(nombres_modelos,i,color=['b','g','r'])
  axs[aux1, aux2].set_title(bases_de_datos[aux1+aux2])
  axs[aux1, aux2].set_ylim([0,1])
  aux1=aux1+1
  if aux1==5:
    aux1=0
    aux2=aux2+1

plt.show()

w, p = wilcoxon(salida_CCR[:,1],salida_CCR[:,0])

print(w)
print(p)

f,pg=friedmanchisquare(salida_CCR[:,0],salida_CCR[:,1],salida_CCR[:,2])

print(f)
print(pg)

p = 0.95
df = 1
# retrieve value <= probability
value = chi2.ppf(p, df)
print(value)
iman=(9*f)/((10*2)-f)
import scipy.stats
scor=scipy.stats.f.ppf(0.95,2,18)
print(iman)
print(scor)