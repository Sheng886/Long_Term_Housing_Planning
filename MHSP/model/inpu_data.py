from model import func, scenariotree
from arguments import Arguments
import numpy as np
import pandas as pd
import time
import math
import pdb
import os
import os.path
import time
import sys


class input_data_class:
    def __init__(self, args):

        ### ------------------ MC & Poisson --------------- ###

        # Read the CSV file
        start_time = time.time()
        data = np.loadtxt(args.demand_path, delimiter=",", skiprows=1) 

        # Extract indices and values
        indices = data[:, :-1].astype(int)  # All columns except the last are indices (convert to int)
        values = data[:, -1]                # Last column is the value

        # Determine the shape of the original array
        shape = tuple(np.max(indices, axis=0) + 1)  # Add 1 because indices are zero-based

        # Create an empty array and fill it with the values
        self.demand = np.zeros(shape)
        self.demand[tuple(indices.T)] = values  # Use advanced indexing to map values back

        
        end_time = time.time()
        time_taken = end_time - start_time
        print("Demand data loaded. ",time_taken,"secs.")

        name_without_extension = args.demand_path.split('.')[0]
        info = name_without_extension.split('_')

        args.T = int(info[3])
        args.N = int(info[5])
        args.J = int(info[7])
        args.M = int(info[9])
        args.K = int(info[11])

        start_time = time.time()
        data = np.loadtxt(args.MC_trans_path, delimiter=",", skiprows=1) 

        indices = data[:, :-1].astype(int)  # All columns except the last are indices (convert to int)
        values = data[:, -1]                # Last column is the value

        # Determine the shape of the original array
        shape = tuple(np.max(indices, axis=0) + 1)  # Add 1 because indices are zero-based

        # Create an empty array and fill it with the values
        self.MC_tran_matrix  = np.zeros(shape)
        self.MC_tran_matrix [tuple(indices.T)] = values  # Use advanced indexing to map values back

        # pdb.set_trace()

        # sum_result = np.sum(self.demand, axis=(3,4,5))
        # Print the shape and the result
        # print("Shape of the result after summing first three dimensions:", sum_result.shape)
        # print("Summation result:", sum_result)

        end_time = time.time()
        time_taken = end_time - start_time
        print("MC trans matrix loaded.",time_taken,"secs.")

        temp = 1
        for t in range(args.T):
            temp += args.N**(t+1)
        args.TN = temp

        # pdb.set_trace()

        if(args.Model == "2SSP" or args.Model == "Extend"):
            start_time = time.time()
            self.tree = scenariotree.ScenarioTree(args, args.TN, self.demand)
            self.tree._build_tree_red(self.MC_tran_matrix, self.demand)
            end_time = time.time()
            time_taken = end_time - start_time
            print("ScenarioTree generated.",time_taken,"secs.")
            print("Memory Used.",sys.getsizeof(self.tree)) 

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

        start_time = time.time()
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

            # print("R:", self.O_p[p])
            # print("R:", self.H_p[p])

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

            # print(self.CU_g[g])


        # ### ------------------Staging Area Capacity ------------------ ###
        self.Cap_w = np.zeros((args.W))
        self.E_w = np.zeros((args.W))

        df_Cap_w = pd.read_excel("data/Staging_Area_info.xlsx")

        for w in range(args.W):
            self.Cap_w[w]  = df_Cap_w["Capacity"][w]
            self.E_w[w] = df_Cap_w["Etend_price"][w]

            # print("Cap_w:", df_Cap_w["Capacity"][w])
            # print("E_w:", df_Cap_w["Etend_price"][w])

            
        end_time = time.time()
        time_taken = end_time - start_time
        print("Parameters loaded.",time_taken,"secs.")


        # pdb.set_trace()