import numpy as np

print("Introduce el numero de filas:")
rows=input()
print("Introduce el numero de columnas:")
cols=input()
randmat=np.random.rand(rows, cols)
print(randmat)
print("El maximo valor de la matriz es ", randmat.max())
print("El minimo valor de la matriz es {randmat.min()}")


