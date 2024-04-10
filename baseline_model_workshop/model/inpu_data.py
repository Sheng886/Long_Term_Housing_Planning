from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import time
import math
import pdb
import os
import os.path


class input_data_class:
    def __init__(self, args):

        ### ------------------ Demand Distribution parameter ------------------------- ###

        df_demand_mean_std = pd.read_excel("data/demand_all_orginal.xlsx")

        demand_mean = np.zeros((args.J))
        demand_std = np.zeros((args.J))

        for j in range(args.J):
            demand_mean[j] = df_demand_mean_std.iloc[j][1]
            demand_std[j] = df_demand_mean_std.iloc[j][2]

        ### ------------------ Hurricane Count Distibution Parameter ------------------------- ###

        df_hurricane_count_mean = pd.read_excel("data/Storm_impact_1851.xlsx")

        count_mean = np.zeros((args.T))

        for t in range(args.T):
            count_mean[t] = df_hurricane_count_mean.iloc[t][2]


        ### ------------------ Percentage Group ------------------------- ###

        df_group_per = pd.read_excel("data/household_type_percentage.xlsx")

        group_percentage = np.zeros((args.J,args.G))

        for j in range(args.J):
            for g in range(args.G):
                group_percentage[j][g] = df_group_per.iloc[j][g+1]

        
        self.demand = np.zeros((args.K,args.J,args.G,args.T))
        for k in range(args.K):
            for t in range(args.T):
                count_temp = np.random.poisson(count_mean[t])
                for j in range(args.J):
                    demand_total = 0
                    for count in range(count_temp):
                        temp = int(np.random.normal(demand_mean[j], demand_std[j]))
                        if(temp >= 0):
                            demand_total = demand_total + temp
                    for g in range(args.G):
                        self.demand[k][j][g][t] = demand_total*group_percentage[j][g]

        # np.save('demand.npy', self.demand)

        self.demand = np.load('demand.npy')


        ### ------------------ Distance matrix ------------ ###

        df_Staging_Area_loc = pd.read_excel("data/Staging_Area_loc.xlsx")
        df_Study_Region_loc = pd.read_excel("data/Study_Region_loc.xlsx")
        df_Suppy_node_loc = pd.read_excel("data/Suppy_node_loc.xlsx")

        name_column_loc = list(df_Staging_Area_loc.columns)
        df_Staging_Area_loc = df_Staging_Area_loc[['latitude','longitude']]
        df_Study_Region_loc = df_Study_Region_loc[['latitude','longitude']]
        df_Suppy_node_loc = df_Suppy_node_loc[['latitude','longitude']]

        self.wj_dis = func.distance_matrix(df_Staging_Area_loc,df_Study_Region_loc)
        self.iw_dis = func.distance_matrix(df_Suppy_node_loc,df_Staging_Area_loc)


        # ### ------------------ Transportation price ------------------ ### 
        self.t_cost = args.t_cost

        # ### ------------------ House Information ------------------ ###

        self.I_p = np.zeros((args.P))
        self.CH_p = np.zeros((args.P))
        self.O_p = np.zeros((args.P))
        self.R_p = np.zeros((args.P))
        self.u_p = np.zeros((args.P))

        df_House_info = pd.read_excel("data/House_Info.xlsx")

        for p in range(args.P):
            self.I_p[p] = df_House_info.iloc[0][p+1]
            self.CH_p[p] = df_House_info.iloc[1][p+1]
            self.O_p[p] = df_House_info.iloc[2][p+1]
            self.R_p[p] = df_House_info.iloc[3][p+1]
            self.u_p[p] = df_House_info.iloc[4][p+1]


        # ### ------------------ House_Attribute ------------------ ###

        self.A_H_flood = np.zeros((args.A,args.P))
        self.A_H_wind = np.zeros((args.A,args.P))

        df_A_H_flood = pd.read_excel("data/House_Attribute_flood.xlsx")
        df_A_H_wind = pd.read_excel("data/House_Attribute_wind.xlsx")

        for a in range(args.A):
            for p in range(args.P):
                self.A_H_flood[a][p] = df_A_H_flood.iloc[a][p+1]
                self.A_H_wind[a][p] = df_A_H_wind.iloc[a][p+1]


        # ### ------------------Household weight ------------------ ###

        self.Hd_weight = np.zeros((args.A,args.G))

        df_Hd_weight = pd.read_excel("data/Victim_Weight.xlsx")

        for a in range(args.A):
            for g in range(args.G):
                self.Hd_weight[a][g] = df_Hd_weight.iloc[a][g+1]

        

        # ### ------------------Unmet Penalty Parameter ------------------ ###
        self.CU_g = np.zeros((args.G))

        df_CU_g = pd.read_excel("data/Victim_Info.xlsx")

        for g in range(args.G):
            self.CU_g[g] = df_CU_g.iloc[0][g+1]


        # ### ------------------Staging Area Capacity ------------------ ###
        self.Cap_w = np.zeros((args.W))

        df_Cap_w = pd.read_excel("data/Staging_Area_info.xlsx")

        for w in range(args.W):
            self.Cap_w[w]  = df_Cap_w["Capacity"][w]


        # pdb.set_trace()