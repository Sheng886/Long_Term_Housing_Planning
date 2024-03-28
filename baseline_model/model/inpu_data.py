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

        ### ------------------ demand ------------------------- ###
        
        self.demand = np.zeros((args.K,args.J,args.G,args.T))
        for k in range(args.K):
            for j in range(args.J):
                a = 0
                b = 0
                for g in range(args.G):
                    if(g == 5):
                        a = 1200
                        b = 1000
                    elif(g == 3):
                        a = 1000
                        b = 800
                    elif(g == 4):
                        a = 800
                        b = 600
                    elif(g == 2):
                        a = 600
                        b = 400
                    elif(g == 0):
                        a = 400
                        b = 200
                    elif(g == 1):
                        a = 200
                        b = 0
                    for t in range(args.T):
                        self.demand[k][j][g][t] = np.random.randint(b,a)


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

        df_House_info = pd.read_excel("data/House_Info.xlsx")

        for p in range(args.P):
            self.I_p[p] = df_House_info.iloc[0][p+1]
            self.CH_p[p] = df_House_info.iloc[1][p+1]
            self.O_p[p] = df_House_info.iloc[2][p+1]
            self.R_p[p] = df_House_info.iloc[3][p+1]



        # ### ------------------ House_Attribute ------------------ ###

        self.A_H_flood = np.zeros((args.A,args.P))
        self.A_H_wind = np.zeros((args.A,args.P))

        # df = pd.read_excel(args.Deprivation_Penalty_path)
        # self.deprivation_a0 = df["a_0"]*args.TIU
        # self.deprivation_a1 = df["a_1"]*args.TIU

        # ### ------------------ Mismatch Penalty ------------------ ###

        # df = pd.read_excel(args.Mismatch_Penalty_path)
        # self.group_name = list(df.columns)
        # self.group_name = self.group_name[1:]
        # self.mismatch = np.zeros((args.P-1,args.G))
        # for i in range (0,args.P-1):
        #     for j in range(0,args.G):
        #         self.mismatch[i][j] = df[self.group_name[j]][i]

        # ### ------------------ Unmet Penalty ------------------ ###

        # df = pd.read_excel(args.Unmet_Penalty_path)
        # unmet_column = list(df.columns)
        # self.unmet = np.zeros(args.G)
        # for g in range(0,args.G):
        #     self.unmet[g] = df[unmet_column[g]][0]

        # ### ------------------ Unused Inventory Penalty ------------------ ###

        # df = pd.read_excel(args.Unused_Inventory_Penalty_path)
        # unused_column = list(df.columns)
        # self.unused = np.zeros(args.P)
        # for p in range(0,args.P):
        #     self.unused[p] = df[unused_column[p]][0]

        # ### ------------------ Staging Area ------------------ ###
        
        # df = pd.read_excel(args.Staging_Area_path)
        # staging_area_column = list(df.columns)
        # staging_area_location = df[['latitude','longitude']]
        # self.staging_area_capacity = np.zeros(args.W)
        # self.high_staging_area_flow = np.zeros(args.W)
        # self.mid_staging_area_flow = np.zeros(args.W)
        # self.low_staging_area_flow = np.zeros(args.W)
        # for i in range(0,args.W):
        #     self.staging_area_capacity[i] = df[staging_area_column[2]][i]*args.TIU
        #     self.high_staging_area_flow[i] = df[staging_area_column[3]][i]*args.TIU
        #     self.mid_staging_area_flow[i] = df[staging_area_column[4]][i]*args.TIU
        #     self.low_staging_area_flow[i] = df[staging_area_column[5]][i]*args.TIU

        # ### ------------------ Scenario Generation ------------------ ###
        
        

        # func.scenario_generation(args,200)

        # trailer_name = args.Demand_Trailer_path
        # MHU_name = args.Demand_MHU_path
        # if(args.random_generate == 1):
        #     if(args.Large_Scenario_data == 1):
        #         args.K = args.Num_Large_Scenario
        #         trailer_name = "data/Scenario/trailer_" + str(args.K) +".txt"
        #         MHU_name = "data/Scenario/MHU_" + str(args.K) + ".txt"
        #         self.demand_per_trailer = np.loadtxt("data/Scenario/trailer_1000.txt")
        #         self.demand_per_MHU = np.loadtxt("data/Scenario/MHU_1000.txt")
        #     else:
        #         func.scenario_generation(args, args.K)
        #         trailer_name = "data/Scenario/trailer_" + str(args.K) +".txt"
        #         MHU_name = "data/Scenario/MHU_" + str(args.K) + ".txt"

        

        # ### ------------------ Study Region ------------------ ###

        # df = pd.read_excel(args.Study_Region_path)
        # study_region_column = list(df.columns)
        # study_region_location = df[['latitude','longitude']]
        # self.homeowner_occupied = np.zeros(args.J)

        # if(args.Large_Scenario_data == 1):
        #     trailer_name = "data/Scenario/trailer_" + str(args.K) +".txt"
        #     MHU_name = "data/Scenario/MHU_" + str(args.K) + ".txt"

        # self.demand_per_trailer = np.loadtxt(trailer_name)
        # self.demand_per_MHU = np.loadtxt(MHU_name)
        
        # for j in range(0,args.J):
        #     self.homeowner_occupied[j] = df[study_region_column[5]][j]
        #     for k in range(0,args.K):
        #         self.demand_per_trailer[k][j] = self.homeowner_occupied[j]*self.demand_per_trailer[k][j]
        #         self.demand_per_MHU[k][j] = self.homeowner_occupied[j]*self.demand_per_MHU[k][j]
        
        # # np.savetxt("trailer_50_K50_3.txt",self.demand_per_trailer)
        # # np.savetxt("MHU_50_K50_3.txt",self.demand_per_MHU)

        

        # # self.demand_per_trailer = np.loadtxt(args.Data_input_path_trailer)
        # # self.demand_per_MHU = np.loadtxt(args.Data_input_path_MHU)


        # # self.demand_per_trailer = np.loadtxt("sc_50_type1.txt").transpose()
        # # self.demand_per_MHU = np.loadtxt("sc_50_type2.txt").transpose()

        # self.demand = np.stack((self.demand_per_trailer,self.demand_per_MHU),axis=1)
        # self.demand = np.round(self.demand)


        # # temp_ = np.zeros((args.J*args.G*args.K,4))
        # # for k in range(args.K):
        # #     for j in range(args.J):
        # #         for g in range(args.G):
        # #             temp_[k*args.J*args.G+j*args.G+g,0] = k
        # #             temp_[k*args.J*args.G+j*args.G+g,1] = j
        # #             temp_[k*args.J*args.G+j*args.G+g,2] = g
        # #             temp_[k*args.J*args.G+j*args.G+g,3] = self.demand[k][g][j]

        # # df_name = ["k","j","g",'d']      
        # # df = pd.DataFrame(temp_, columns=[df_name])
        # # df.to_csv("demand.csv")

        # # pdb.set_trace()



        # ### ------------------ Supply ------------------ ###

        # df = pd.read_excel(args.Supplier_path)
        # supplier_column = list(df.columns)
        # supplier_location = df[['latitude','longitude']]
        # self.supplier_price = np.zeros((args.I,args.P))
        # self.high_supplier_inventory = np.zeros((args.I,args.P))
        # self.mid_supplier_inventory = np.zeros((args.I,args.P))
        # self.low_supplier_inventory = np.zeros((args.I,args.P))
        # self.high_supplier_prod_time = np.zeros((args.I,args.P))
        # self.mid_supplier_prod_time = np.zeros((args.I,args.P))
        # self.low_supplier_prod_time = np.zeros((args.I,args.P))
        # self.num_production_line = np.zeros(args.I)
        # self.extra_production_line = np.zeros(args.I)
        # self.high_supplier_flow = np.zeros(args.I)
        # self.mid_supplier_flow = np.zeros(args.I)
        # self.low_supplier_flow = np.zeros(args.I)
        # self.high_supplier_prod_time_TIU = np.zeros((args.I,args.P))
        # self.mid_supplier_prod_time_TIU = np.zeros((args.I,args.P))
        # self.low_supplier_prod_time_TIU = np.zeros((args.I,args.P))
        # for i in range(0,args.I):
        #     self.num_production_line[i] = df[supplier_column[24]][i]
        #     self.extra_production_line[i] = df[supplier_column[25]][i]
        #     self.high_supplier_flow[i] = df[supplier_column[26]][i]*args.TIU
        #     self.mid_supplier_flow[i] = df[supplier_column[27]][i]*args.TIU
        #     self.low_supplier_flow[i] = df[supplier_column[28]][i]*args.TIU
        #     for p in range(0, args.P):
        #         self.supplier_price[i][p] = df[supplier_column[3+p]][i]
        #         self.mid_supplier_inventory[i][p] = df[supplier_column[6+p]][i]
        #         self.high_supplier_inventory[i][p] = df[supplier_column[12+p]][i]
        #         self.low_supplier_inventory[i][p] = df[supplier_column[9+p]][i]
        #         self.high_supplier_prod_time[i][p] = df[supplier_column[15+p]][i]
        #         self.mid_supplier_prod_time[i][p] = df[supplier_column[18+p]][i]
        #         self.low_supplier_prod_time[i][p] = df[supplier_column[21+p]][i]
        #         self.high_supplier_prod_time[i][p] = int(math.ceil(df[supplier_column[15+p]][i]/args.TIU))
        #         self.mid_supplier_prod_time_TIU[i][p] = int(math.ceil(df[supplier_column[18+p]][i]/args.TIU))
        #         self.low_supplier_prod_time_TIU[i][p] = int(math.ceil(df[supplier_column[21+p]][i]/args.TIU))

        
        # ### ------------------ distance matrix ------------------ ###

        # self.supplier_area_distance = func.distance_matrix(supplier_location,staging_area_location)
        # self.area_region_distance = func.distance_matrix(staging_area_location,study_region_location)
        # self.supplier_region_distance = func.distance_matrix(supplier_location,study_region_location)

        # ### ------------------ travel days matrix ------------------ ###

        # # extra input file need to be added, here we assume all 1 days 

        # self.iw_t = np.full((args.I, args.W),1)
        # self.wj_t = np.full((args.W, args.J),1)
        # self.ij_t = np.full((args.I, args.J),1)

        # ### ------------------ Production TIU Ratio ---------------- ###
        # self.production_ratio_N = np.zeros((args.I,args.P))
        # for i in range(args.I):
        #     for p in range(args.P):
        #         self.production_ratio_N[i][p] = np.round((1/self.mid_supplier_prod_time[i][p])*(self.mid_supplier_prod_time_TIU[i][p])*args.TIU)
        
        # self.production_ratio_E = np.zeros((args.I,args.P))
        # for i in range(args.I):
        #     for p in range(args.P):
        #         self.production_ratio_E[i][p] = np.round((1/self.low_supplier_prod_time[i][p])*(self.low_supplier_prod_time_TIU[i][p])*args.TIU)
       
        # ### ------------------ cost scale down ------------------ ###

        # self.house_price = self.house_price/scale_down_cost
        # self.supplier_price = self.supplier_price/scale_down_cost
        # self.trans_price = self.trans_price/scale_down_cost

        # pdb.set_trace()