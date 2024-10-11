from model import func, scenariotree
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


    
        ### ------------------ MC & Poisson --------------- ###



        self.MC_tran_matrix = np.load(args.MC_trans_path)
        self.demand = np.load(args.demand_path)
        self.demand_root = np.load(args.demand_root_path)

        sum_result = np.sum(self.demand, axis=(3,4,5))

        # Print the shape and the result
        # print("Shape of the result after summing first three dimensions:", sum_result.shape)
        # print("Summation result:", sum_result)

        name_without_extension = args.MC_trans_path.split('.')[0]
        info = name_without_extension.split('_')

        args.T = int(info[4])
        args.N = int(info[6])
        args.J = int(info[8])
        args.M = int(info[10])
        args.K = int(info[12])

        temp = 1
        for t in range(args.T):
            temp += args.N**(t+1)
        args.TN = temp

        # pdb.set_trace()

        if(args.Model == "2SSP" or args.Model == "Extend"):
            self.tree = scenariotree.ScenarioTree(args, args.TN, self.demand_root)
            self.tree._build_tree_red(self.MC_tran_matrix, self.demand)

        # self.tree.print_tree_sce()
        # self.tree.print_tree_red()
        
        # pdb.set_trace()


        ### ------------------ Distance matrix ------------ ###

        # df_Staging_Area_loc = pd.read_excel("data/Staging_Area_loc.xlsx")
        # df_Study_Region_loc = pd.read_excel("data/Study_Region_loc.xlsx")
        # df_Suppy_node_loc = pd.read_excel("data/Suppy_node_loc.xlsx")

        # name_column_loc = list(df_Staging_Area_loc.columns)
        # df_Staging_Area_loc = df_Staging_Area_loc[['latitude','longitude']]
        # df_Study_Region_loc = df_Study_Region_loc[['latitude','longitude']]
        # df_Suppy_node_loc = df_Suppy_node_loc[['latitude','longitude']]

        # self.wj_dis = func.distance_matrix(df_Staging_Area_loc,df_Study_Region_loc)
        # self.iw_dis = func.distance_matrix(df_Suppy_node_loc,df_Staging_Area_loc)


        # ### ------------------ Transportation price ------------------ ### 
        self.t_cost = args.t_cost

        # ### ------------------ House Information ------------------ ###

        self.P_p = np.zeros((args.P))
        self.O_p = np.zeros((args.P))
        self.R_p = np.zeros((args.P))
        self.H_p = np.zeros((args.P))

        


        df_House_info = pd.read_excel("data/House_Info.xlsx")

        for p in range(args.P):
            # self.P_p[p] = args.P_p_factor*df_House_info.iloc[0][p+1]
            self.P_p[p] = args.P_p_factor*2
            # self.O_p[p] = args.O_p_factor*df_House_info.iloc[1][p+1]
            self.O_p[p] = args.O_p_factor*1000
            self.R_p[p] = df_House_info.iloc[2][p+1]
            self.H_p[p] =  self.O_p[p]*args.H_p_factor*df_House_info.iloc[2][p+1]

            print("R:", self.O_p[p])
            print("R:", self.H_p[p])

        # ### ------------------Supply ------------------ ###
        self.B_i = np.zeros((args.I))

        df_I = pd.read_excel("data/Supply_Info.xlsx")

        for i in range(args.I):
            self.B_i[i]  = df_I["Production"][i]

        # ### ------------------Unmet Penalty Parameter ------------------ ###
        self.CU_g = np.zeros((args.G))

        df_CU_g = pd.read_excel("data/Victim_Info.xlsx")

        for g in range(args.G):
            # self.CU_g[g] = args.C_u_factor*df_CU_g.iloc[0][g+1]
            self.CU_g[g] = args.C_u_factor*100*self.O_p[g]

            print(self.CU_g[g])


        # ### ------------------Staging Area Capacity ------------------ ###
        self.Cap_w = np.zeros((args.W))
        self.E_w = np.zeros((args.W))

        df_Cap_w = pd.read_excel("data/Staging_Area_info.xlsx")

        for w in range(args.W):
            self.Cap_w[w]  = df_Cap_w["Capacity"][w]
            self.E_w[w] = df_Cap_w["Etend_price"][w]

            print("Cap_w:", df_Cap_w["Capacity"][w])
            print("E_w:", df_Cap_w["Etend_price"][w])

            




        # pdb.set_trace()