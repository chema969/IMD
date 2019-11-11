import pandas as pd
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
nombre_variables = ['viento', 'velocidad_viento', 'periodo_ola_dominante', 'media_ola_dominante','presion','temperatura','clase']
olas = pd.read_csv('train.csv')
le.fit(olas['clase'])
clases_numeros = le.transform(olas['clase'])

olas_df = pd.DataFrame(olas[nombre_variables], columns=nombre_variables)

pd.plotting.scatter_matrix(olas_df, c=clases_numeros, figsize=(8, 8));
