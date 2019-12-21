# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1626UIWFAMAosBRahDKqVl_xwBJWC7y2M
"""

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from scipy.spatial import distance_matrix
from scipy.stats import wilcoxon,friedmanchisquare,chi2
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.metrics import silhouette_score
import math

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def fittingScoreKmeans(train_input,modelo):
  modelo.fit(train_input)
  dist=distance_matrix(train_input,train_input)
  pred=modelo.predict(train_input)
  adyacencias=np.zeros((pred.size,pred.size))
  for i in range(0,pred.size):
    for j in range(0,pred.size):
      if pred[i]==pred[j]:
        adyacencias[i,j]=1
  scores=corr2(adyacencias,dist)  
  return scores

def fittingScoreAgglomerative(train_input,modelo):
  pred=modelo.fit_predict(train_input)
  score=silhouette_score(train_input,pred)
  return score

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

def conseguir_modelos(n):
  kmeansn1=KMeans(n-1)
  kmeans=KMeans(n)
  kmeansn2=KMeans(n+1)

  return [kmeansn1,kmeans,kmeansn2]

def conseguir_modelos_agglomerative(n):
  simp=AgglomerativeClustering(n,linkage='single')
  comp=AgglomerativeClustering(n,linkage='complete')
  avrg=AgglomerativeClustering(n,linkage='average')

  return [simp,comp,avrg]

bases_de_datos=['flowmeter.csv','glass.csv','sat.csv','segment_challenge.csv','waves.csv']

salida_CCR=np.zeros((np.size(bases_de_datos),6),dtype=float)

aux1=0
for i in bases_de_datos:
  train_i,train_o=leer_csv(i)
  modelos=conseguir_modelos(np.unique(train_o).size)
  aux2=0
  for j in modelos:
    score=fittingScoreKmeans(train_i,j)
    salida_CCR[aux1,aux2]=score
    aux2=aux2+1
  modelos=conseguir_modelos_agglomerative(np.unique(train_o).size)
  for j in modelos:
    score=fittingScoreAgglomerative(train_i,j)
    salida_CCR[aux1,aux2]=score
    aux2=aux2+1
  aux1=aux1+1

print(salida_CCR)