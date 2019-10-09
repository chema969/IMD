import pandas as pd 
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt

#nombre_fichero = input("Introduce el nombre del fichero: ")
nombre_fichero="basesDeDatos/diabetes.arff"
#data = pd.read_csv(nombre_fichero) 
data,meta = arff.loadarff(nombre_fichero)
data=np.array(data);
print(data[1:3]);
plt.hist(data[1:][2:7],density=True, stacked=True,bins=10)
plt.show()
