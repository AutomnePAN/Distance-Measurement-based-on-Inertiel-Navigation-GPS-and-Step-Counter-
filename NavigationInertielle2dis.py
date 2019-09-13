import pandas as pd
import numpy as np
import math as mt

#Importer the dataset 
accelerometer = pd.read_csv('Accelerometer.csv')
linear_acceleration = pd.read_csv('Linear Acceleration.csv')
gyrope = pd.read_csv('Gyroscope.csv')
accelerometer.columns = [[0,1,2,3]];
linear_acceleration.columns = [[0,1,2,3]];
gyrope.columns = [[0,1,2,3]];

#Combiner les données à une matrice
A = pd.merge(accelerometer, linear_acceleration[[1,2,3]],left_index=True,right_index=True,how='outer')
A_g = pd.merge(A, gyrope[[1,2,3]],left_index=True,right_index=True,how='outer')

# Temps: le temps; 
# a_(x,y,z) : les accélérometre ;
#a_(x,y,z)_w : les accélérations linéaires ; 
# o_(x,y,z): les vitesses angulaires
A_0 = A_g.values
A_0 = A_0[A_0[:,0] > 5]
print(A_0)
A_g = pd.DataFrame(A_0,columns=['temps', 'a_x_w', 'a_y_w','a_z_w','a_x','a_y','a_z','o_x','o_y','o_z'])
print(A_g)
#Enlever les données de premiers 5 secondes

#Changement de referentiel de chaque instant a instant t = 0
# Le referntiel a l'instant t = 0 est le referentiel que l'on transorme a .
# Establishment of Matrix de change ment de referentiel

T = A_g['temps'].values
A_x_w = A_g['a_x_w'].values
A_y_w = A_g['a_y_w'].values
A_z_w = A_g['a_z_w'].values
o_x = A_g['o_x'].values
o_y = A_g['o_y'].values
o_z = A_g['o_z'].values
A_X = np.zeros(A_g.shape[0])
A_Y = np.zeros(A_g.shape[0])
A_Z = np.zeros(A_g.shape[0])

# M : la matrice de 'changement de référentiel'
M = np.array([[1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
for t in range(1, A_g.shape[0]):
    M  = M + np.array([[0, -o_z[t]*(T[t]- T[t-1]), -o_y[t]*(T[t]- T[t-1])],[o_z[t]*(T[t]- T[t-1]), 0 , o_x[t]*(T[t]- T[t-1])],[o_y[t]*(T[t]- T[t-1]) , -o_x[t]*(T[t]- T[t-1]),0]])
    I = np.array([A_x_w[t] ,  A_y_w[t], A_z_w[t]])
    F = M.dot(I)
    A_X[t] = F[0]
    A_Y[t] = F[1]
    A_Z[t] = F[2]

# a_(X,Y,Z) les composantes de accélération dans le référentiel fixée
A_g['a_X'] = A_X
A_g['a_Y'] = A_Y
A_g['a_Z'] = A_Z

accelerometer.columns = [[0,1,2,3]]
linear_acceleration.columns = [[0,1,2,3]]

#Calculs de "g
g_raw = accelerometer - linear_acceleration
g = g_raw[[1,2,3]]
g.columns  = [u'g_x', u'g_y', u'g_z']
#Calcul de module de 'g'
g['module_g'] = None
g['module_g'] = np.sqrt( np.square(g['g_x']) + np.square(g['g_y']) + np.square(g['g_z']) ) 

A_g = pd.DataFrame.merge(A_g,g,left_index=True,right_index=True,how='outer')


#Déterminer l'angle de référentiel fixé and le plan horizontal
G_x_0  = g['g_x'][0]
G_y_0  = g['g_y'][0]
G_z_0  = g['g_z'][0]
G_module = g['module_g'][0]

#Calculs de Vitesse a chaque instants
V_x = np.zeros(np.size(A_X))
V_y = np.zeros(np.size(A_X))
V_z = np.zeros(np.size(A_X))
V_m = np.zeros(np.size(A_X))
V_x_h = np.zeros(np.size(A_X))
V_y_h = np.zeros(np.size(A_X))
V_z_h = np.zeros(np.size(A_X))
V_m_h = np.zeros(np.size(A_X))

for t in range(1, np.size(A_X)):
    V_x[t] = V_x[ t-1 ] + (T[t] - T[t-1])*(A_X[t]+ A_X[t-1])/2
    V_y[t] = V_y[ t-1 ] + (T[t] - T[t-1])*(A_Y[t]+ A_Y[t-1])/2
    V_z[t] = V_z[ t-1 ] + (T[t] - T[t-1])*(A_Z[t]+ A_Z[t-1])/2
    # Projeter V_x,V_y, V_z sur le plan horizontal
    V_x_h[t] = V_x[t] - ( V_x[t]*G_x_0 + V_y[t]*G_y_0 + V_z[t]*G_z_0 )*V_x[t] /(G_module* G_module)
    V_y_h[t] = V_y[t] - ( V_x[t]*G_x_0 + V_y[t]*G_y_0 + V_z[t]*G_z_0 )*V_y[t] /(G_module* G_module)
    V_z_h[t] = V_z[t] - ( V_x[t]*G_x_0 + V_y[t]*G_y_0 + V_z[t]*G_z_0 )*V_z[t] /(G_module* G_module)
    V_m_h[t] = mt.sqrt( V_x_h[t]*V_x_h[t] + V_y_h[t]*V_y_h[t] + V_z_h[t]*V_z_h[t] )


#Calcul le distance 
D = np.zeros(np.size(A_X))
for t in range(1,np.size(A_X)):
    D[t] = D[t-1]  +  (T[t] - T[t-1])*(V_m_h[t]+ V_m_h[t-1])/2

#Exporter le résultat
distance = pd.DataFrame(D,columns = ["distance"])
distance_t = pd.merge(A_g[['temps']],distance,left_index=True,right_index=True,how='outer')

#Le résultat est présenté danss le document 'Distance.csv'
pd.DataFrame(distance_t).to_csv('Distance.csv')