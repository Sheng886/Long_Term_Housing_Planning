import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import math
import csv

###### Distacne Matrix ######

def distance_matrix(df1,df2):

    geom1 = [Point(xy) for xy in zip(df1.latitude, df1.longitude)]
    geom2 = [Point(xy) for xy in zip(df2.latitude, df2.longitude)]
    gdf1 = gpd.GeoDataFrame(df1,geometry=geom1)
    gdf2 = gpd.GeoDataFrame(df2,geometry=geom2)

    n1 = df1.shape[0]
    n2 = df2.shape[0]

    distance = np.zeros((n1,n2))

    for i in range(n1):
        for j in range(n2):
            distance[i][j] = geodesic((gdf1['latitude'][i],gdf1['longitude'][i]), (gdf2['latitude'][j],gdf2['longitude'][j])).km

    return distance

###### Scenario Generation ######

def scenario_generation(args, k):

    scenarios = k

    # import pdb;pdb.set_trace()

    k_neighbor = 4

    mean_1 = 0.5037691
    mean_2 = 0.5036693
    min_1 = -0.3640831
    st_1 = 0.01035242
    st_2 = 0.008703158
    min_2 = -0.4216035

    lambda_1 = 0.71134
    lambda_2 = 0.75294

    sigma_1 = 0.20684
    sigma_2 = 0.20695

    fit_ian = pd.read_csv('data/Scenario/fit_ian.csv')
    neighbors= pd.read_csv('data/Scenario/adjacent_county_ian.csv')

    fit_1 = fit_ian["fit_trailer"].to_numpy()
    fit_2 = fit_ian["fit_MHU"].to_numpy()

    adjacent_matrix = np.zeros((args.J,args.J))
    for i in range(0,args.J):
        adjacent_matrix[i][neighbors["n1"][i]-1] = 1/k_neighbor
        adjacent_matrix[i][neighbors["n2"][i]-1] = 1/k_neighbor
        adjacent_matrix[i][neighbors["n3"][i]-1] = 1/k_neighbor
        adjacent_matrix[i][neighbors["n4"][i]-1] = 1/k_neighbor


    output_1_demand = np.zeros((scenarios,args.J))
    output_2_demand = np.zeros((scenarios,args.J))

    for k in range(0,scenarios):
        error_1 = np.random.normal(0, sigma_1, args.J)
        error_2 = np.random.normal(0, sigma_2, args.J)

        y_1 = fit_1 + np.matmul((np.identity(args.J) - lambda_1*adjacent_matrix),error_1)
        y_2 = fit_2 + np.matmul((np.identity(args.J) - lambda_2*adjacent_matrix),error_2)

        demand_1 = ((np.sign(y_1))*(np.abs(y_1))**(1/0.3030303) - 1e-9 + min_1)*st_1 + mean_1
        demand_2 = ((np.sign(y_2))*(np.abs(y_2))**(1/0.222222) - 1e-9 + min_2)*st_2 + mean_2

        demand_1 = np.log(demand_1) - np.log(1 - demand_1)
        demand_2 = np.log(demand_2) - np.log(1 - demand_2)

        for i in range(0,args.J):
            if(demand_1[i] <= 0):
                demand_1[i] = 0
            if(demand_2[i] <= 0):
                demand_2[i] = 0

        for j in range(0,args.J):
            output_1_demand[k][j] = demand_1[j]
            output_2_demand[k][j] = demand_2[j]


    trailer_name = "data/Scenario/trailer_" + str(scenarios) +".txt"
    MHU_name = "data/Scenario/MHU_" + str(scenarios) + ".txt"

    np.savetxt(trailer_name,output_1_demand)
    np.savetxt(MHU_name,output_2_demand)

    return 0







