import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from scipy.stats import poisson
from scipy.stats import gaussian_kde
import math
import csv
import pdb

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


def distribute_hurricane_month(n_hurricane, month_probabilities):

    month_boxes = len(month_probabilities)
    month_distribution = np.random.multinomial(n_hurricane, month_probabilities)
    
    return month_distribution


def main_generator(args):

    # Data input ------------------------------------------------------------------


    Frequency = pd.read_excel(args.Frequency)
    Frequency = Frequency.to_dict()

    Atlantic_month_dis = pd.read_excel(args.Atlantic_month_dis)
    Atlantic_month_dis = Atlantic_month_dis.to_numpy()[0]

    Atlantic_SS_dis = pd.read_excel(args.Atlantic_SS_dis)
    Atlantic_SS_dis = Atlantic_SS_dis.to_numpy()[0]

    Gulf_month_dis = pd.read_excel(args.Gulf_month_dis)
    Gulf_month_dis = Gulf_month_dis.to_numpy()[0]
    
    Gulf_SS_dis = pd.read_excel(args.Gulf_SS_dis)
    Gulf_SS_dis = Gulf_SS_dis.to_numpy()[0]

    # Study Region Data
    Data_Atlantic_df = pd.read_csv(args.Data_Atlantic)
    Data_Gulf_df = pd.read_csv(args.Data_Gulf)

    # For 2kde
    Hurricane_landfall_Atlantic_df = pd.read_csv(args.Hurricane_landfall_Atlantic)

    longitude = Hurricane_landfall_Atlantic_df['Longitude'].values
    latitude = Hurricane_landfall_Atlantic_df['Latitude'].values
    coords_Atlantic = np.vstack([longitude, latitude])
    kde_Atlantic = gaussian_kde(coords_Atlantic)

    Hurricane_landfall_Gulf_df = pd.read_csv(args.Hurricane_landfall_Gulf)
    longitude = Hurricane_landfall_Gulf_df['Longitude'].values
    latitude = Hurricane_landfall_Gulf_df['Latitude'].values
    coords_Gulf = np.vstack([longitude, latitude])
    kde_Gulf = gaussian_kde(coords_Gulf)

    Regression_par = pd.read_excel(args.Regression_par)
    Regression_par = Regression_par.to_dict()

    # MC Trans Matrix --------------------------------------------------------------
    # State (A,G) ------------------------------------------------------------------

    states = []
    temp = []

    for n1 in range(args.N_A):
        for n2 in range(args.N_G):
            states.append([n1, n2])

    args.N = args.N_A * args.N_G


    MC_trans = []
    temp = []
    for n in range(args.N):
        if(states[n][0] == args.N_A-1):
            pro_A_temp = 1 - sum( poisson.pmf(n_, Frequency["Atlantic"][0]) for n_ in range(args.N_A-1))
        else:
            pro_A_temp = poisson.pmf(states[n][0], Frequency["Atlantic"][0])

        if(states[n][1] == args.N_G-1):
            pro_G_temp = 1 - sum( poisson.pmf(n_, Frequency["Gulf"][0]) for n_ in range(args.N_G-1))
        else:
            pro_G_temp =poisson.pmf(states[n][1], Frequency["Gulf"][0])
        temp_ = pro_A_temp*pro_G_temp
        temp.append(temp_)

    for n in range(args.N):
        MC_trans.append(temp)
    
    # Demand Generator--------------------------------------------------------------
    # Step 1 distribute the numbers to the month by month distribution -------------

    demand = np.zeros((args.T,args.N,args.K,args.J,args.G,args.M+1))
    demand_root = np.zeros((args.K,args.J,args.G,args.M+1))

    for T in range(args.T):
        for n in range(args.N):
            for k in range(args.T):
                A_month_freq = distribute_hurricane_month(states[n][0], Atlantic_month_dis)
                G_month_freq = distribute_hurricane_month(states[n][1], Gulf_month_dis)
                for M in range(args.M):
                    A_num_hurricane = A_month_freq[M]
                    G_num_hurricane = G_month_freq[M]

                    for h in range(A_num_hurricane):
                        Landfall = kde_Atlantic.resample(1)
                        SS = np.random.choice([1,2,3,4,5], size=1, p=Atlantic_SS_dis)

                        


                    for h in range(G_num_hurricane):
                        Landfall = kde_Gulf.resample(1)
                        SS = np.random.choice([1,2,3,4,5], size=1, p=Gulf_SS_dis)

           
    pdb.set_trace()














