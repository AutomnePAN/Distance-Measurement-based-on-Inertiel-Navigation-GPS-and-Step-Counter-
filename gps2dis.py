#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd             # Vous devez installer pandas et numpy
import numpy as np
import math

file_name = 'gpslente3.csv'     # Nom du fichier csv
data = pd.read_csv(file_name)   # lire les données
R = 6371000                     # rayon de la terre
batch = 1                     # fréquence pour lire les données

latitude = data['Latitude (°)'] 
longitude = data['Longitude (°)'] 


def haversin(theta):            # formule de Haversin
    return math.sin(theta/2)**2

def distance(lon1,lon2,lat1,lat2):  # Calculs des distances selon les équations dans le rapport
    res = haversin(lat2-lat1) + math.cos(lat1) * math.cos(lat2) * haversin(lon1-lon2)
    sin2dr = math.sqrt(res)
    d = 2 * R * math.asin(sin2dr)
    assert d>=0
    return d

def main():
    dis_a_chaque_instant = 0
    dis = np.zeros(latitude.shape)
    data_length = latitude.shape[0]
    count = 0
    while count + batch <= data_length-1: #calcul de la distance à chaque instant
        lon1 = math.radians(longitude[count]) ; lat1 = math.radians(latitude[count])
        lon2 = math.radians(longitude[count+batch]) ; lat2 = math.radians(latitude[count+batch])
        dis_a_chaque_instant += distance(lon1,lon2,lat1,lat2)
        dis[count+batch] = dis_a_chaque_instant  
        count += batch
    return pd.DataFrame(dis)

if __name__=='__main__':
    data['Our_distance'] = main()
    print(data['Our_distance'].tail())
    data.to_csv(file_name,index=None)
    print('Fini')
    