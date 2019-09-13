import pandas as pd
import numpy as np
import math as mt

#Importer les accélérations
Raw_data = pd.read_csv('Linear Acceleration.csv')
Raw_data.columns = [[u'temps', u'a_x', u'a_y', u'a_z']]

# Enlever les données de premiers 5 secondes
A = Raw_data.values
A_0 = A[A[:,0] > 5]

T = A_0[:,0]
A_x = A_0[:,1]
A_y = A_0[:,2]
A_z = A_0[:,3]

# Etude  d'une periode de 0.7s
p = 0.7

#tt note les instants que nous faisons un pas
tt = np.array([])
#P note les valeurs de accélérations quand nous faisons un pas
P = np.array([])

N = int(p/0.03)
A_1 = A[0:N]
A_Y = A_1[:,2]
s = np.size(A_Y)
S = np.size(A_y)
Key = 0
for t in range(s,S):
    A_Y = np.append(A_Y, A_y[t])
    A_Y = np.delete(A_Y, 0)
    M = np.max(A_Y)
    m = np.min(A_Y)
    p = abs( M - m )
    if (p > 1):
        if (A_Y[int(mt.floor(s/2)) - 1] == np.min(A_Y)):
            Key = Key +1
            tt = np.append(tt, T[t])
            P = np.append(P, p)

print(Key)
print(tt)
print(P)