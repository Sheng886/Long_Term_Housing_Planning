from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import time
import math
import pdb
import os
import os.path


generate_bigM_cut_time = 0
generate_special_cut_time = 0
solving_sub_Normal_problem = 0
solving_sub_Emergency_problem = 0
solving_master_problem = 0

big_M_cut = 0
spe_cut = 0

cut_vio_thred = 1e-5
nonint_ratio = 1e-5
dual_primal_ratio = 1e-2

node = 0

dual = 0

scale_down_cost = 10000

mip_gap = 1e-8
time_limit = 3600*4

# data input

class input_data:
    def __init__(self, args):

        ### ------------------ time point ----------------------- ###

        df = pd.read_excel(args.Timepoint_path)
        self.first_end_time_point = df["First_End"][0]
        self.deprivation_time_point = df["Deprivation_Start"][0]
        self.second_end_time_point = df["Second_End"][0]


        # self.first_end_time_point = 1
        # self.deprivation_time_point = 3
        # self.second_end_time_point = 4
        ### ------------------ time-index preprocess ------------ ###


        first_end_time_point_TIU = int(math.ceil(self.first_end_time_point/args.TIU))
        second_end_time_point_TIU = first_end_time_point_TIU + int(math.ceil((self.second_end_time_point-self.first_end_time_point)/args.TIU))
        deprivation_time_point_TIU = first_end_time_point_TIU + int(math.ceil((self.deprivation_time_point-self.first_end_time_point)/args.TIU))
        
        self.first_end_time_point = first_end_time_point_TIU
        self.second_end_time_point = second_end_time_point_TIU
        self.deprivation_time_point = deprivation_time_point_TIU
        args.T = self.second_end_time_point

        ### ------------------ Transportation price ------------------ ### 
        self.trans_price = args.trans_cost

        ### ------------------ House Information ------------------ ###

        df = pd.read_excel(args.House_Info_path)

        self.house_name = list(df.columns)
        self.house_name = self.house_name[1:]

        self.house_price = np.zeros(args.P)
        self.house_volumn_factor = np.zeros(args.P)
        self.house_install_time = np.zeros(args.P)
        

        for i in range(0,args.P):
            self.house_price[i] = df[self.house_name[i]][1]
            self.house_volumn_factor[i] = df[self.house_name[i]][2]
            self.house_install_time[i] = int(math.ceil(df[self.house_name[i]][3]/args.TIU))

        df = pd.read_excel(args.House_Match)

        self.house_0_match = np.zeros(args.P)

        for i in range(1,args.P):
            self.house_0_match[i] = df[self.house_name[i]][0]


        ### ------------------ Deprivation Cost Function ------------------ ###

        df = pd.read_excel(args.Deprivation_Penalty_path)
        self.deprivation_a0 = df["a_0"]*args.TIU
        self.deprivation_a1 = df["a_1"]*args.TIU

        ### ------------------ Mismatch Penalty ------------------ ###

        df = pd.read_excel(args.Mismatch_Penalty_path)
        self.group_name = list(df.columns)
        self.group_name = self.group_name[1:]
        self.mismatch = np.zeros((args.P-1,args.G))
        for i in range (0,args.P-1):
            for j in range(0,args.G):
                self.mismatch[i][j] = df[self.group_name[j]][i]

        ### ------------------ Unmet Penalty ------------------ ###

        df = pd.read_excel(args.Unmet_Penalty_path)
        unmet_column = list(df.columns)
        self.unmet = np.zeros(args.G)
        for g in range(0,args.G):
            self.unmet[g] = df[unmet_column[g]][0]

        ### ------------------ Unused Inventory Penalty ------------------ ###

        df = pd.read_excel(args.Unused_Inventory_Penalty_path)
        unused_column = list(df.columns)
        self.unused = np.zeros(args.P)
        for p in range(0,args.P):
            self.unused[p] = df[unused_column[p]][0]

        ### ------------------ Staging Area ------------------ ###
        
        df = pd.read_excel(args.Staging_Area_path)
        staging_area_column = list(df.columns)
        staging_area_location = df[['latitude','longitude']]
        self.staging_area_capacity = np.zeros(args.W)
        self.high_staging_area_flow = np.zeros(args.W)
        self.mid_staging_area_flow = np.zeros(args.W)
        self.low_staging_area_flow = np.zeros(args.W)
        for i in range(0,args.W):
            self.staging_area_capacity[i] = df[staging_area_column[2]][i]*args.TIU
            self.high_staging_area_flow[i] = df[staging_area_column[3]][i]*args.TIU
            self.mid_staging_area_flow[i] = df[staging_area_column[4]][i]*args.TIU
            self.low_staging_area_flow[i] = df[staging_area_column[5]][i]*args.TIU

        ### ------------------ Scenario Generation ------------------ ###
        
        

        func.scenario_generation(args,200)

        trailer_name = args.Demand_Trailer_path
        MHU_name = args.Demand_MHU_path
        if(args.random_generate == 1):
            if(args.Large_Scenario_data == 1):
                args.K = args.Num_Large_Scenario
                trailer_name = "data/Scenario/trailer_" + str(args.K) +".txt"
                MHU_name = "data/Scenario/MHU_" + str(args.K) + ".txt"
                self.demand_per_trailer = np.loadtxt("data/Scenario/trailer_1000.txt")
                self.demand_per_MHU = np.loadtxt("data/Scenario/MHU_1000.txt")
            else:
                func.scenario_generation(args, args.K)
                trailer_name = "data/Scenario/trailer_" + str(args.K) +".txt"
                MHU_name = "data/Scenario/MHU_" + str(args.K) + ".txt"

        

        ### ------------------ Study Region ------------------ ###

        df = pd.read_excel(args.Study_Region_path)
        study_region_column = list(df.columns)
        study_region_location = df[['latitude','longitude']]
        self.homeowner_occupied = np.zeros(args.J)

        if(args.Large_Scenario_data == 1):
            trailer_name = "data/Scenario/trailer_" + str(args.K) +".txt"
            MHU_name = "data/Scenario/MHU_" + str(args.K) + ".txt"

        self.demand_per_trailer = np.loadtxt(trailer_name)
        self.demand_per_MHU = np.loadtxt(MHU_name)
        
        for j in range(0,args.J):
            self.homeowner_occupied[j] = df[study_region_column[5]][j]
            for k in range(0,args.K):
                self.demand_per_trailer[k][j] = self.homeowner_occupied[j]*self.demand_per_trailer[k][j]
                self.demand_per_MHU[k][j] = self.homeowner_occupied[j]*self.demand_per_MHU[k][j]
        
        # np.savetxt("trailer_50_K50_3.txt",self.demand_per_trailer)
        # np.savetxt("MHU_50_K50_3.txt",self.demand_per_MHU)

        

        # self.demand_per_trailer = np.loadtxt(args.Data_input_path_trailer)
        # self.demand_per_MHU = np.loadtxt(args.Data_input_path_MHU)


        # self.demand_per_trailer = np.loadtxt("sc_50_type1.txt").transpose()
        # self.demand_per_MHU = np.loadtxt("sc_50_type2.txt").transpose()

        self.demand = np.stack((self.demand_per_trailer,self.demand_per_MHU),axis=1)
        self.demand = np.round(self.demand)


        # temp_ = np.zeros((args.J*args.G*args.K,4))
        # for k in range(args.K):
        #     for j in range(args.J):
        #         for g in range(args.G):
        #             temp_[k*args.J*args.G+j*args.G+g,0] = k
        #             temp_[k*args.J*args.G+j*args.G+g,1] = j
        #             temp_[k*args.J*args.G+j*args.G+g,2] = g
        #             temp_[k*args.J*args.G+j*args.G+g,3] = self.demand[k][g][j]

        # df_name = ["k","j","g",'d']      
        # df = pd.DataFrame(temp_, columns=[df_name])
        # df.to_csv("demand.csv")

        # pdb.set_trace()



        ### ------------------ Supply ------------------ ###

        df = pd.read_excel(args.Supplier_path)
        supplier_column = list(df.columns)
        supplier_location = df[['latitude','longitude']]
        self.supplier_price = np.zeros((args.I,args.P))
        self.high_supplier_inventory = np.zeros((args.I,args.P))
        self.mid_supplier_inventory = np.zeros((args.I,args.P))
        self.low_supplier_inventory = np.zeros((args.I,args.P))
        self.high_supplier_prod_time = np.zeros((args.I,args.P))
        self.mid_supplier_prod_time = np.zeros((args.I,args.P))
        self.low_supplier_prod_time = np.zeros((args.I,args.P))
        self.num_production_line = np.zeros(args.I)
        self.extra_production_line = np.zeros(args.I)
        self.high_supplier_flow = np.zeros(args.I)
        self.mid_supplier_flow = np.zeros(args.I)
        self.low_supplier_flow = np.zeros(args.I)
        self.high_supplier_prod_time_TIU = np.zeros((args.I,args.P))
        self.mid_supplier_prod_time_TIU = np.zeros((args.I,args.P))
        self.low_supplier_prod_time_TIU = np.zeros((args.I,args.P))
        for i in range(0,args.I):
            self.num_production_line[i] = df[supplier_column[24]][i]
            self.extra_production_line[i] = df[supplier_column[25]][i]
            self.high_supplier_flow[i] = df[supplier_column[26]][i]*args.TIU
            self.mid_supplier_flow[i] = df[supplier_column[27]][i]*args.TIU
            self.low_supplier_flow[i] = df[supplier_column[28]][i]*args.TIU
            for p in range(0, args.P):
                self.supplier_price[i][p] = df[supplier_column[3+p]][i]
                self.mid_supplier_inventory[i][p] = df[supplier_column[6+p]][i]
                self.high_supplier_inventory[i][p] = df[supplier_column[12+p]][i]
                self.low_supplier_inventory[i][p] = df[supplier_column[9+p]][i]
                self.high_supplier_prod_time[i][p] = df[supplier_column[15+p]][i]
                self.mid_supplier_prod_time[i][p] = df[supplier_column[18+p]][i]
                self.low_supplier_prod_time[i][p] = df[supplier_column[21+p]][i]
                self.high_supplier_prod_time[i][p] = int(math.ceil(df[supplier_column[15+p]][i]/args.TIU))
                self.mid_supplier_prod_time_TIU[i][p] = int(math.ceil(df[supplier_column[18+p]][i]/args.TIU))
                self.low_supplier_prod_time_TIU[i][p] = int(math.ceil(df[supplier_column[21+p]][i]/args.TIU))

        
        ### ------------------ distance matrix ------------------ ###

        self.supplier_area_distance = func.distance_matrix(supplier_location,staging_area_location)
        self.area_region_distance = func.distance_matrix(staging_area_location,study_region_location)
        self.supplier_region_distance = func.distance_matrix(supplier_location,study_region_location)

        ### ------------------ travel days matrix ------------------ ###

        # extra input file need to be added, here we assume all 1 days 

        self.iw_t = np.full((args.I, args.W),1)
        self.wj_t = np.full((args.W, args.J),1)
        self.ij_t = np.full((args.I, args.J),1)

        ### ------------------ Production TIU Ratio ---------------- ###
        self.production_ratio_N = np.zeros((args.I,args.P))
        for i in range(args.I):
            for p in range(args.P):
                self.production_ratio_N[i][p] = np.round((1/self.mid_supplier_prod_time[i][p])*(self.mid_supplier_prod_time_TIU[i][p])*args.TIU)
        
        self.production_ratio_E = np.zeros((args.I,args.P))
        for i in range(args.I):
            for p in range(args.P):
                self.production_ratio_E[i][p] = np.round((1/self.low_supplier_prod_time[i][p])*(self.low_supplier_prod_time_TIU[i][p])*args.TIU)
       
        ### ------------------ cost scale down ------------------ ###

        self.house_price = self.house_price/scale_down_cost
        self.supplier_price = self.supplier_price/scale_down_cost
        self.trans_price = self.trans_price/scale_down_cost

        # pdb.set_trace()




class master():
    def __init__(self,args,input_data):


        ### ------------------ Model ------------------ ###

        self.input_data = input_data
        self.temp = []


        self.model = gp.Model("Master")

        # First-stage
        self.x = self.model.addVars(args.I, args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Xiwpt')
        self.s = self.model.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Sipt')
        self.m = self.model.addVars(args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Mwpt')
        self.v = self.model.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Vipt')
        self.a = self.model.addVars(args.I, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Awpt')
        self.z = self.model.addVars(args.K, vtype=GRB.BINARY, name='Zk')
        self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, name='theta')

        # Objective
        self.model.setObjective(quicksum((self.input_data.supplier_price[i][p] + self.input_data.supplier_area_distance[i][w]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*self.x[i,w,p,t] for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T)) + self.theta,GRB.MINIMIZE);


        if(args.UB != None):
            self.model.addConstr(quicksum((self.input_data.supplier_price[i][p] + self.input_data.supplier_area_distance[i][w]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*self.x[i,w,p,t] for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T)) + self.theta <= args.UB)

        if(args.modular == 0):
            for w in range(args.W):
                for t in range(args.T):
                    self.model.addConstr(self.m[w,0,t] == 0)

            for w in range(args.W):
                for i in range(args.I):
                    for t in range(args.T):
                        self.model.addConstr(self.x[i,w,0,t] == 0)

        for w in range(args.W):
            for p in range(args.P):
                self.model.addConstr(self.m[w,p,0] == 0)

        for w in range(args.W):
            for p in range(args.P):
                for i in range(args.I):
                    self.model.addConstr(self.x[i,w,p,0] == 0)

        # Initial Machine Capacity && Machine used flow && Every time unit Machine Capacity
        for i in range(args.I):
            # 3i
            self.model.addConstr(self.a[i,0] == 0)
            # 3j
            for t in range(args.T-1):
                self.model.addConstr(self.a[i,t+1] + quicksum(self.input_data.production_ratio_N[i][p]*self.s[i,p,t+1-self.input_data.mid_supplier_prod_time_TIU[i][p]] for p in range(args.P) if t+1-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0) ==  quicksum(self.input_data.production_ratio_N[i][p]*self.s[i,p,t] for p in range(args.P) if t+self.input_data.mid_supplier_prod_time_TIU[i][p] <= args.T-1) + self.a[i,t])
                # 3k
                self.model.addConstr(self.a[i,t] <= self.input_data.num_production_line[i])

        # Initial Inventory && Inventory flow
        for i in range(args.I):
            for p in range(args.P):
                # 3d
                self.model.addConstr(self.v[i,p,0] == self.input_data.mid_supplier_inventory[i][p])
                self.model.addConstr(self.s[i,p,args.T-1] == 0)
                # 3e
                for t in range(args.T-1):
                    if(t-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0):
                        self.model.addConstr(self.v[i,p,t+1] + quicksum(self.x[i,w,p,t+1] for w in range(args.W) if t+1+self.input_data.iw_t[i][w] <= args.T-1) ==  self.v[i,p,t] + self.input_data.production_ratio_N[i][p]*self.s[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]])
                    else:
                        self.model.addConstr(self.v[i,p,t+1] + quicksum(self.x[i,w,p,t+1] for w in range(args.W)) ==  self.v[i,p,t] )
                                        
        # 3f
        # Supply sending flow
        for t in range(args.T):
            for i in range(args.I):
                self.model.addConstr(quicksum(self.input_data.house_volumn_factor[p]*self.x[i,w,p,t] for w in range(args.W) for p in range(args.P)) <= self.input_data.mid_supplier_flow[i])

        # 3g
        # First-stage Staging area inventory flow limitation
        for w in range(args.W): 
            for t in range(args.T-1):
                for p in range(args.P):
                    self.model.addConstr(self.m[w,p,t+1] == quicksum(self.x[i,w,p,t-self.input_data.iw_t[i][w]] for i in range(args.I) if t-self.input_data.iw_t[i][w] >= 0) + self.m[w,p,t])

        # 3h
        # Statgin area Capacity
        for w in range(args.W):
            for t in range(args.T):
                self.model.addConstr(quicksum(self.input_data.house_volumn_factor[p]*self.m[w,p,t] for p in range(args.P)) <= self.input_data.staging_area_capacity[w])
        

        self.vk = self.model.addVars(args.I,args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Vk_ipt')
        self.nuk = self.model.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Nuk_ipt')
        self.fk_ij = self.model.addVars(args.I, args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_ijpt')
        self.fk_wj = self.model.addVars(args.W, args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_wjpt')
        self.fk0_j = self.model.addVars(args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk0_jpt')
        self.mk = self.model.addVars(args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Mk_wpt')
        self.yk = self.model.addVars(args.J, args.P, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='yk_jpgt')
        self.bk = self.model.addVars(args.I, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='bk_it')
        self.dk = self.model.addVars(args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='dk_jgt')
        self.qk = self.model.addVars(args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='qk_jgt')


        self.model.addConstr(self.theta == quicksum(self.input_data.area_region_distance[w][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price*self.fk_wj[w,j,p,t] for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(self.input_data.first_end_time_point,args.T))
                + quicksum(args.short_factor*self.input_data.unmet[g]*self.input_data.house_price[g+1]*(self.qk[j,g,args.T-1]) for j in range(args.J) for g in range(args.G))
                + quicksum(self.input_data.unused[p]*self.input_data.house_price[p]*self.mk[w,p,args.T-1] for w in range(args.W) for p in range(args.P))
                + quicksum((self.input_data.deprivation_a0[0] + self.input_data.deprivation_a1[0]*(args.TIU)*(args.T-t))*(self.input_data.house_price[g+1])*(self.qk[j,g,t]) for j in range(args.J) for t in range(self.input_data.deprivation_time_point,args.T) for g in range(args.G))
                + quicksum((args.emergency_price_factor*self.input_data.house_price[p] + self.input_data.supplier_region_distance[i][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*self.fk_ij[i,j,p,t] for i in range(args.I) for j in range(args.J) for t in range(self.input_data.first_end_time_point,args.T) for p in range(args.P)))



        for i in range(args.I):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.model.addConstr(self.nuk[i,p,t] == 0)
                    if(t != self.input_data.first_end_time_point):
                        self.model.addConstr(self.vk[i,p,t] == 0)

        for w in range(args.W):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(0,self.input_data.first_end_time_point+1):
                        self.model.addConstr(self.fk_wj[w,j,p,t] == 0)


        for j in range(args.J):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.model.addConstr(self.fk0_j[j,p,t] == 0)

        for i in range(args.I):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(0,self.input_data.first_end_time_point+1):
                        self.model.addConstr(self.fk_ij[i,j,p,t] == 0)

        for w in range(args.W):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point):
                    self.model.addConstr(self.mk[w,p,t] == 0)

        for i in range(args.I):
            for t in range(0,self.input_data.first_end_time_point+1):
                self.model.addConstr(self.bk[i,t] == 0)

        for j in range(args.J):
            for g in range(args.G):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.model.addConstr(self.dk[j,g,t] == 0)
                    if(t != self.input_data.first_end_time_point):
                        self.model.addConstr(self.qk[j,g,t] == 0)

        for t in range(args.T):
            for j in range(args.J):
                for p in range(args.P):
                    for g in range(args.G):
                        if(p-1 != g):
                            self.model.addConstr(self.yk[j,p,g,t] == 0)

        for j in range(args.J):
            for p in range(args.P):
                for g in range(args.G):
                    for t in range(0,self.input_data.first_end_time_point+1):
                        self.model.addConstr(self.yk[j,p,g,t] == 0)

        for i in range(args.I):
            for p in range(args.P):
                self.model.addConstr(self.vk[i,p,self.input_data.first_end_time_point] == self.v[i,p,self.input_data.first_end_time_point])

        # 1b
        for w in range(args.W):
            for p in range(args.P):
                self.model.addConstr(self.mk[w,p,self.input_data.first_end_time_point] == self.m[w,p,self.input_data.first_end_time_point])


        for w in range(args.W):
            for t in range(self.input_data.first_end_time_point,args.T):
                self.model.addConstr(quicksum(self.input_data.house_volumn_factor[p]*self.mk[w,p,t] for p in range(args.P)) <= self.input_data.staging_area_capacity[w])

        for w in range(args.W):
                for p in range(args.P):
                    for t in range(self.input_data.first_end_time_point,args.T-1):
                        self.model.addConstr(self.mk[w,p,t+1] + quicksum(self.fk_wj[w,j,p,t+1] for j in range(args.J) if t+1+self.input_data.wj_t[w][j] <= args.T-1) - self.mk[w,p,t] == quicksum(self.x[i,w,p,t-self.input_data.iw_t[i][w]] for i in range(args.I) if (t-self.input_data.iw_t[i][w]) >= 0))
        
        for j in range(args.J):
            for t in range(args.T-1):
                for g in range(args.G):
                    self.model.addConstr(self.dk[j,g,t+1] == self.dk[j,g,t] + self.yk[j,p,p-1,t+1])




    def run_constraint(self,args):

        # 1h
        for j in range(args.J):
            for g in range(args.G):
                for t in range(self.input_data.first_end_time_point,args.T):
                    temp = self.model.addConstr(self.dk[j,g,t] + self.qk[j,g,t] == self.input_data.demand[0][g][j])
                    self.temp.append(temp)

        if(args.error == 0):
            # 1f
            # match flow
            for j in range(args.J):
                for t in range(self.input_data.first_end_time_point,args.T):
                    for p in range(1,args.P):
                        if(t - self.input_data.house_install_time[0] >= self.input_data.first_end_time_point + 1):
                            temp = self.model.addConstr(1/self.input_data.house_0_match[p]*self.fk0_j[j,p,t-self.input_data.house_install_time[0]] + quicksum(self.fk_wj[w,j,p,t-self.input_data.wj_t[w][j]] for w in range(args.W) if (t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point + 1)) + quicksum(self.fk_ij[i,j,p,t] for i in range(args.I)) == self.yk[j,p,p-1,t]) 
                        else:
                            temp = self.model.addConstr(quicksum(self.fk_wj[w,j,p,t-self.input_data.wj_t[w][j]] for w in range(args.W) if (t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point + 1)) + quicksum(self.fk_ij[i,j,p,t] for i in range(args.I)) == self.yk[j,p,p-1,t]) 
                        self.temp.append(temp)

            for i in range(args.I):
                for j in range(args.J):
                    for p in range(args.P):
                        for t in range(self.input_data.first_end_time_point,args.T):
                            temp = self.model.addConstr(self.fk_ij[i,j,p,t] == 0)
                            self.temp.append(temp)


            # 1d
            # second-stage staging area delivery flow limit
            for w in range(args.W):
                for t in range(self.input_data.first_end_time_point,args.T):
                    temp = self.model.addConstr(quicksum(self.input_data.house_volumn_factor[p]*self.fk_wj[w,j,p,t] for p in range(args.P) for j in range(args.J)) <= self.input_data.mid_staging_area_flow[w]) 
                    self.temp.append(temp)
            # 1e
            # novel house match
            for j in range(args.J):
                for t in range(args.T):
                    temp = self.model.addConstr(quicksum(self.fk_wj[w,j,0,t] for w in range(args.W)) - quicksum(self.fk0_j[j,p,t+self.input_data.house_install_time[0]] for p in range(1,args.P) if (t + self.input_data.house_install_time[0]) <= args.T-1) == 0) 
                    self.temp.append(temp)
        else:
            if(args.modular == 0):
                for i in range(args.I):
                    for j in range(args.J):
                        for t in range(args.T):
                            temp = self.model.addConstr(self.fk_ij[i,j,0,t] == 0)
                            self.temp.append(temp)
                            # for p in range(args.P):
                            #     self.sub_Recovery.addConstr(fk_ij[i,j,p,t] == 0)
            # 2f
            # match flow
            for j in range(args.J):
                for t in range(self.input_data.first_end_time_point,args.T):
                    for p in range(1,args.P):
                        if(t - self.input_data.house_install_time[0] >= self.input_data.first_end_time_point + 1):
                            temp = self.model.addConstr(1/self.input_data.house_0_match[p]*self.fk0_j[j,p,t-self.input_data.house_install_time[0]] + quicksum(self.fk_wj[w,j,p,t-self.input_data.wj_t[w][j]] for w in range(args.W) if t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point+1) + quicksum(self.fk_ij[i,j,p,t-self.input_data.ij_t[i][j]] for i in range(args.I) if t-self.input_data.ij_t[i][j]>= self.input_data.first_end_time_point+1) == self.yk[j,p,p-1,t]) 
                        else:
                            temp = self.model.addConstr(quicksum(self.fk_wj[w,j,p,t-self.input_data.wj_t[w][j]] for w in range(args.W) if t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point+1) + quicksum(self.fk_ij[i,j,p,t-self.input_data.ij_t[i][j]] for i in range(args.I) if t-self.input_data.ij_t[i][j]>= self.input_data.first_end_time_point+1) == self.yk[j,p,p-1,t]) 
                        self.temp.append(temp)
            # 2h
            # emergency production
            for i in range(args.I):
                for t in range(self.input_data.first_end_time_point,args.T-1):
                        temp = self.model.addConstr(self.bk[i,t+1] + quicksum(self.input_data.production_ratio_E[i][p]*self.nuk[i,p,t+1-self.input_data.low_supplier_prod_time_TIU[i][p]] for p in range(args.P) if t+1-self.input_data.low_supplier_prod_time_TIU[i][p] >= self.input_data.first_end_time_point+1) == quicksum(self.input_data.production_ratio_E[i][p]*self.nuk[i,p,t] for p in range(args.P) if t+self.input_data.low_supplier_prod_time_TIU[i][p] <= args.T-1) + self.bk[i,t]) 
                        self.temp.append(temp)

            # 2i
            for i in range(args.I):
                for t in range(self.input_data.first_end_time_point,args.T):
                    temp = self.model.addConstr(self.bk[i,t] <= self.input_data.extra_production_line[i])
                    self.temp.append(temp)

            # 2d
            # Second-stage Supplier Inventory flow
            self.supplier_inventory_cons = []
            for i in range(args.I):
                for p in range(args.P):
                    for t in range(self.input_data.first_end_time_point,args.T-1):
                        if(t-self.input_data.low_supplier_prod_time_TIU[i][p] >= self.input_data.first_end_time_point + 1):
                            if(t-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0):
                                temp = self.model.addConstr(self.vk[i,p,t+1]  + quicksum(self.fk_ij[i,j,p,t+1] for j in range(args.J) if t+1+self.input_data.ij_t[i][j] <= args.T-1) - self.vk[i,p,t] - self.input_data.production_ratio_E[i][p]*self.nuk[i,p,t-self.input_data.low_supplier_prod_time_TIU[i][p]] == self.input_data.production_ratio_N[i][p]*self.s[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]] - sum(self.x[i,w,p,t+1] for w in range(args.W)))
                            else:
                                temp = self.model.addConstr(self.vk[i,p,t+1]  + quicksum(self.fk_ij[i,j,p,t+1] for j in range(args.J) if t+1+self.input_data.ij_t[i][j] <= args.T-1) - self.vk[i,p,t] - self.input_data.production_ratio_E[i][p]*self.nuk[i,p,t-self.input_data.low_supplier_prod_time_TIU[i][p]] == - quicksum(self.x[i,w,p,t+1] for w in range(args.W)))
                        else:
                            if(t-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0):
                                temp = self.model.addConstr(self.vk[i,p,t+1]  + quicksum(self.fk_ij[i,j,p,t+1] for j in range(args.J) if t+1+self.input_data.ij_t[i][j] <= args.T-1) - self.vk[i,p,t]  == self.input_data.production_ratio_N[i][p]*self.s[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point] - quicksum(self.x[i,w,p,t+1] for w in range(args.W)))
                            else:
                                temp = self.model.addConstr(self.vk[i,p,t+1]  + quicksum(self.fk_ij[i,j,p,t+1] for j in range(args.J) if t+1+self.input_data.ij_t[i][j] <= args.T-1) - self.vk[i,p,t]  == - quicksum(self.x[i,w,p,t+1] for w in range(args.W)))
                        self.temp.append(temp)

            # 2e
            # novel house match
            for j in range(args.J):
                for t in range(args.T):
                    temp = self.model.addConstr(quicksum(self.fk_ij[i,j,0,t-self.input_data.ij_t[i][j]] for i in range(args.I) if t-self.input_data.ij_t[i][j] >= self.input_data.first_end_time_point+1) + quicksum(self.fk_wj[w,j,0,t-self.input_data.wj_t[w][j]] for w in range(args.W) if t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point+1) - quicksum(self.fk0_j[j,p,t] for p in range(1,args.P) if t + self.input_data.house_install_time[0] <= args.T-1) == 0) 
                    self.temp.append(temp)

        

    def run(self,args):
        start = time.time()
        
        # self.model.write("model.lp")
        self.model.update()
        self.model.Params.lazyConstraints = 1
        self.model._xvars = self.x
        self.model._svars = self.s
        self.model._mvars = self.m
        self.model._vvars = self.v
        self.model._avars = self.a
        self.model._zvars = self.z
        self.model._thetavar = self.theta
        self.model.setParam('TimeLimit', time_limit)


        o_argsK = args.K
        o_demand = self.input_data.demand
        o_error = args.error

        demand_temp = np.zeros((1,args.G,args.J))
        
        obj_diff = np.zeros((o_argsK))

        args.K = 1
        for k in range(o_argsK):
            for e in range(2):
                args.error = e

                for g in range(args.G):
                    for j in range(args.J):
                        demand_temp[0][g][j] = o_demand[k][g][j]

                self.input_data.demand = demand_temp
                self.run_constraint(args)
                self.model.setParam("OutputFlag", 0)
                self.model.optimize()
                # self.model.write("{}_{}_{}_{}_{}.sol".format(k,e,int(args.short_factor*10)),int(args.dep_factor*10),int(args.unused_factor*10))
                obj_diff[k] = self.model.ObjBound - obj_diff[k]

                for con_temp in self.temp:
                    self.model.remove(con_temp)
                self.temp = []
                self.model.reset()
                
                

        index = np.flip(np.argsort(obj_diff))
        self.model.dispose()

        args.K = o_argsK
        args.error = o_error

        return index


        
