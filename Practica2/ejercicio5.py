import numpy as np
import pandas as pd 
from scipy.io import arff
import matplotlib.pyplot as plt
nombre_fichero="basesDeDatos/aux.csv"
data = pd.read_csv(nombre_fichero)
fig,axes=plt.subplots(len(data.columns)-1,len(data.columns)-1,constrained_layout=True)
figsize=(9,6)
i=0
for col in data.columns:
  j=0
  for col2 in data.columns: 
    if(col!='class'and col2!='class'):
      data.plot.scatter(x=col,y=col2,c='class',colormap='viridis',ax=axes[i,j])
      j=j+1
  i=i+1

plt.show()
