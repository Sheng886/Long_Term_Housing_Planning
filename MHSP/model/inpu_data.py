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

        temp = 1
        for t in range(args.T):
            temp += args.N**(t+1)
        args.TN = temp

        # ### ------------------ demand ------------------------- ###
        
        # df_tree = pd.read_excel("scen_tree/tree_distmatrix.xlsx")
        # df_tree = df_tree.iloc[: , 1:]
        # tree_adj_matrix = df_tree.values.tolist()

        # df_scen1 = pd.read_excel("scen_tree/tree_scen1.xlsx")
        # df_scen2 = pd.read_excel("scen_tree/tree_scen2.xlsx")

        # tree_pr = df_scen1["Pr"].values.tolist()
        # temp = np.array(tree_pr)
        # reshaped_temp = temp.reshape(args.TN, args.K)
        # tree_pr = reshaped_temp.tolist()


        # tree_demand1 = df_scen1.iloc[: , 3:].values.tolist()
        # temp = np.array(tree_demand1)
        # reshaped_temp = temp.reshape(7, args.K, args.M)
        # tree_demand1 = reshaped_temp.tolist()

        # tree_demand2 = df_scen2.iloc[: , 3:].values.tolist()
        # temp = np.array(tree_demand2)
        # reshaped_temp = temp.reshape(args.TN, args.K, args.M)
        # tree_demand2 = reshaped_temp.tolist()

        # self.tree = scenariotree.ScenarioTree(args.TN)
        # self.tree._build_tree_red(tree_adj_matrix)
        # self.tree._build_tree_black(tree_demand1,tree_demand2,tree_pr)
        # # self.tree.print_tree_sce()
        # # self.tree.print_tree_red()
        

        # pdb.set_trace()

        ### ------------------ MC & Poisson --------------- ###

        df_MC = pd.read_excel("scen_tree/MC.xlsx")
        df_MC = df_MC.iloc[: , 1:]
        MC_tran_matrix = df_MC.values.tolist()

        df_month_par = pd.read_excel("scen_tree/Hurricane_month.xlsx")
        df_month_par = df_month_par.iloc[: , 1:]
        temp = np.array(df_month_par)
        reshaped_temp = temp.reshape(args.N, args.M)
        month_par = reshaped_temp.tolist()

        self.demand = np.zeros((args.T,args.N,args.K,args.P,args.M))
        self.demand_root = np.zeros((args.K,args.P,args.M))

        for t in range(args.T):
            for n in range(args.N):
                for m in range(args.M):
                    for k in range(args.K):
                        self.demand[t][n][k][0][m] = np.random.poisson(month_par[n][m], 1)*args.DTrailer
                        self.demand[t][n][k][1][m] = np.random.poisson(month_par[n][m], 1)*args.DMHU

        for n in range(args.N):
            for m in range(args.M):
                for k in range(args.K):
                    self.demand_root[k][0][m] = np.random.poisson(month_par[n][m], 1)*args.DTrailer
                    self.demand_root[k][1][m] = np.random.poisson(month_par[n][m], 1)*args.DMHU


        self.tree = scenariotree.ScenarioTree(args.TN, self.demand_root)
        self.tree._build_tree_red(args, MC_tran_matrix, self.demand)

        self.tree.print_tree_sce()
        self.tree.print_tree_red()
        


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

        self.P_p = np.zeros((args.P))
        self.O_p = np.zeros((args.P))
        self.R_p = np.zeros((args.P))


        df_House_info = pd.read_excel("data/House_Info.xlsx")

        for p in range(args.P):
            self.P_p[p] = df_House_info.iloc[0][p+1]
            self.O_p[p] = df_House_info.iloc[1][p+1]
            self.R_p[p] = df_House_info.iloc[2][p+1]



        

        # ### ------------------Supply ------------------ ###
        self.B_i = np.zeros((args.I))

        df_I = pd.read_excel("data/Supply_Info.xlsx")

        for i in range(args.I):
            self.B_i[i]  = df_I["Production"][i]

        # ### ------------------Unmet Penalty Parameter ------------------ ###
        self.CU_g = np.zeros((args.G))

        df_CU_g = pd.read_excel("data/Victim_Info.xlsx")

        for g in range(args.G):
            self.CU_g[g] = df_CU_g.iloc[0][g+1]


        # ### ------------------Staging Area Capacity ------------------ ###
        self.Cap_w = np.zeros((args.W))
        self.E_w = np.zeros((args.W))

        df_Cap_w = pd.read_excel("data/Staging_Area_info.xlsx")

        for w in range(args.W):
            self.Cap_w[w]  = df_Cap_w["Capacity"][w]
            self.E_w[w] = df_Cap_w["Etend_price"][w]


        # ### ------------------Study Region ------------------ ###
        self.J_pro = np.zeros((args.J))

        df_j_pro = pd.read_excel("scen_tree/region_prob.xlsx")

        for j in range(args.J):
            self.J_pro[j] = df_j_pro.iloc[0][j]


        # pdb.set_trace()