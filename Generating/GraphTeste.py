import pandas as pd
import numpy as np
from numpy.random import RandomState

N = 60000

pares = np.arange(0,1000,2) 
impares = np.arange(1,1000,2)

data = []
for i in range(N):
    opcao = np.random.randint(0,2,size=1)
    # invertido por conveção
    if (opcao==1):
        par = list(np.random.choice(pares,  size=2,replace=False))
    else:
        par = list(np.random.choice(impares,size=2,replace=False))
    par.append(opcao[0])
    data.append(par)

data = np.reshape(data,(len(data),3))

dic_data = {
    'C1': data[:,0],
    'C2': data[:,1],
    'Cls':data[:,2]
}

df_data = pd.DataFrame(dic_data)
df_data.to_csv('graf_test.csv')    