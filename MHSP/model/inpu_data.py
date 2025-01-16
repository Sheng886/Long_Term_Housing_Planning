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

        ### ------------------ Demand --------------- ###

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

        ### ------------------ MC Matrix --------------- ###

        start_time = time.time()
        data = np.loadtxt(args.MC_trans_path, delimiter=",", skiprows=1) 

        indices = data[:, :-1].astype(int)  # All columns except the last are indices (convert to int)
        values = data[:, -1]                # Last column is the value

        # Determine the shape of the original array
        shape = tuple(np.max(indices, axis=0) + 1)  # Add 1 because indices are zero-based

        # Create an empty array and fill it with the values
        self.MC_tran_matrix  = np.zeros(shape)
        self.MC_tran_matrix [tuple(indices.T)] = values  # Use advanced indexing to map values back

        for stage in range(args.T):
            for state in range(args.N):
                self.MC_tran_matrix[stage][state] = self.MC_tran_matrix[stage][state]/sum(self.MC_tran_matrix[stage][state])

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

        start_time = time.time()

        # ### ------------------Supply ------------------ ###

        self.B_i = np.zeros((args.I))

        df_I = pd.read_excel("data/Supply_Info.xlsx")

        for i in range(args.I):
            self.B_i[i]  = df_I["Production"][i]

        print("-------------Supply-----------------------")
        print("Production Capacity:", self.B_i)

        # ### ------------------ House Information ------------------ ###

        self.P_p = np.zeros((args.P))
        self.O_p = np.zeros((args.P))
        self.R_p = np.zeros((args.P))
        self.H_p = np.zeros((args.P))


        df_House_info = pd.read_excel("data/House_Info.xlsx")

        for p in range(args.P):
            self.P_p[p] = args.P_p_factor*df_House_info.iloc[0][p+1]
            self.O_p[p] = args.O_p_factor*df_House_info.iloc[1][p+1]*args.sc
            self.R_p[p] = args.R_p_factor
            self.H_p[p] = args.H_p_factor*self.O_p[p]

        print("-------------House Info-----------------------")
        print("Production time:", self.P_p)
        print("Acquire Cost:", self.O_p)
        print("Recycle Cost:", args.R_p_factor*self.O_p)
        print("Holding Cost:", self.H_p)



        # ### ------------------Unmet Penalty Parameter ------------------ ###
        self.CU_g = np.zeros((args.G))
        for g in range(args.G):
            self.CU_g[g] = args.C_u_factor*self.O_p[g]

        print("-------------Penalty -----------------------")
        print("Unmet penalty:", self.CU_g)



        # ### ------------------Staging Area Capacity ------------------ ###
        self.Cap_w = np.zeros((args.W))
        self.E_w = np.zeros((args.W))
        self.II_w = np.zeros((args.W,args.P))

        df_Cap_w = pd.read_excel("data/Staging_Area_info.xlsx")

        for w in range(args.W):
            self.Cap_w[w]  = args.Cp_w_factor*df_Cap_w["Capacity"][w]
            self.E_w[w] = (sum(self.O_p)/args.P)*args.E_w_factor
            for p in range(args.P):
                self.II_w[w][p] = self.Cap_w[w]*args.II_factor

        print("-------------Penalty -----------------------")
        print("Inital Capacity:", self.Cap_w)
        print("Inital Inventory:", self.II_w)
        print("Increasing Capacity Cost:", self.E_w)
            
        end_time = time.time()
        time_taken = end_time - start_time
        print("Parameters loaded.",time_taken,"secs.")

        print("-------------Info -----------------------")
        print("# of Stage:", args.T, "# of state:", args.N)
        if(args.Model == "SDDP" or "RS_SDDP"):
            print("Sample Method of SDDP", args.sample_path)

        # pdb.set_trace()