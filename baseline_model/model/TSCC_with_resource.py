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

node = 0

big_M_cut = 0
spe_cut = 0

cut_vio_thred = 1e-5
nonint_ratio = 1e-5
dual_primal_ratio = 1e-2

dual = 0

scale_down_cost = 10000

mip_gap = 1e-8
time_limit = 3600*4

cut_save = 0
cut_count = 0

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



class sub_N():
    def __init__(self,args,input_data):

        self.input_data = input_data
        self.sub_Normal = gp.Model("subprob_Normal");
        vk = self.sub_Normal.addVars(args.I,args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Vk_ipt')
        nuk = self.sub_Normal.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Nuk_ipt')
        fk_ij = self.sub_Normal.addVars(args.I, args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_ijpt')
        fk_wj = self.sub_Normal.addVars(args.W, args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_wjpt')
        fk0_j = self.sub_Normal.addVars(args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk0_jpt')
        mk = self.sub_Normal.addVars(args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Mk_wpt')
        yk = self.sub_Normal.addVars(args.J, args.P, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='yk_jpgt')
        bk = self.sub_Normal.addVars(args.I, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='bk_it')
        dk = self.sub_Normal.addVars(args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='dk_jgt')
        qk = self.sub_Normal.addVars(args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='qk_jgt')

        self.vk = vk
        self.nuk = nuk
        self.fk_ij = fk_ij
        self.fk_wj = fk_wj
        self.fk0_j = fk0_j
        self.yk = yk
        self.mk = mk
        self.bk = bk
        self.dk = dk
        self.qk = qk

        self.sub_Normal.setObjective(quicksum(self.input_data.area_region_distance[w][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price*fk_wj[w,j,p,t] for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(self.input_data.first_end_time_point,args.T))
                + quicksum(args.short_factor*self.input_data.unmet[g]*self.input_data.house_price[g+1]*(qk[j,g,args.T-1]) for j in range(args.J) for g in range(args.G))
                + quicksum(self.input_data.unused[p]*self.input_data.house_price[p]*mk[w,p,args.T-1] for w in range(args.W) for p in range(args.P))
                + quicksum(args.dep_factor*(self.input_data.deprivation_a0[0] + self.input_data.deprivation_a1[0]*(args.TIU)*(args.T-t))*(self.input_data.house_price[g+1])*(qk[j,g,t]) for j in range(args.J) for t in range(self.input_data.deprivation_time_point,args.T) for g in range(args.G))
                + quicksum((args.emergency_price_factor*self.input_data.supplier_price[i][p]  + self.input_data.supplier_region_distance[i][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*fk_ij[i,j,p,t] for i in range(args.I) for j in range(args.J) for t in range(self.input_data.first_end_time_point,args.T) for p in range(args.P))
                ,GRB.MINIMIZE);

        for i in range(args.I):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.sub_Normal.addConstr(nuk[i,p,t] == 0)
                    if(t != self.input_data.first_end_time_point):
                        self.sub_Normal.addConstr(vk[i,p,t] == 0)

        for w in range(args.W):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(0,self.input_data.first_end_time_point+1):
                        self.sub_Normal.addConstr(fk_wj[w,j,p,t] == 0)

        for j in range(args.J):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.sub_Normal.addConstr(fk0_j[j,p,t] == 0)

        for i in range(args.I):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(0,self.input_data.first_end_time_point+1):
                        self.sub_Normal.addConstr(fk_ij[i,j,p,t] == 0)

        for w in range(args.W):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point):
                    self.sub_Normal.addConstr(mk[w,p,t] == 0)

        for i in range(args.I):
            for t in range(0,self.input_data.first_end_time_point+1):
                self.sub_Normal.addConstr(bk[i,t] == 0)

        for j in range(args.J):
            for g in range(args.G):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.sub_Normal.addConstr(dk[j,g,t] == 0)
                    if(t != self.input_data.first_end_time_point):
                        self.sub_Normal.addConstr(qk[j,g,t] == 0)

        for t in range(args.T):
            for j in range(args.J):
                for p in range(args.P):
                    for g in range(args.G):
                        if(p-1 != g):
                            self.sub_Normal.addConstr(yk[j,p,g,t] == 0)

        
        
        self.S_initial_inventory_cons = []
        # Second-stage Supplier initial Inventory
        for i in range(args.I):
            for p in range(args.P):
                temp = self.sub_Normal.addConstr(vk[i,p,self.input_data.first_end_time_point] == 0)
                self.S_initial_inventory_cons.append(temp)

        # 1b
        self.W_initial_inventory_cons = []
        # Second-stage staging area initail inventory
        for w in range(args.W):
            for p in range(args.P):
                temp = self.sub_Normal.addConstr(mk[w,p,self.input_data.first_end_time_point] == 0)
                self.W_initial_inventory_cons.append(temp)

        # 3f in sub
        self.W_Capacity_cons = []
        # Statgin area Capacity (second-stage)
        for w in range(args.W):
            for t in range(self.input_data.first_end_time_point,args.T):
                temp = self.sub_Normal.addConstr(quicksum(self.input_data.house_volumn_factor[p]*mk[w,p,t] for p in range(args.P)) <= self.input_data.staging_area_capacity[w])
                self.W_Capacity_cons.append(temp)

        # 1c
        self.W_inventory_flow_cons = []
        #Second-stage stating area inventory flow
        for w in range(args.W):
            for p in range(args.P):
                for t in range(self.input_data.first_end_time_point,args.T-1):
                    temp = self.sub_Normal.addConstr(mk[w,p,t+1] + quicksum(fk_wj[w,j,p,t+1] for j in range(args.J) if t+1+self.input_data.wj_t[w][j] <= args.T-1) - mk[w,p,t] == 0)
                    self.W_inventory_flow_cons.append(temp)

        # 1f
        # match flow
        for j in range(args.J):
            for t in range(self.input_data.first_end_time_point,args.T):
                for p in range(1,args.P):
                    if(t - self.input_data.house_install_time[0] >= self.input_data.first_end_time_point + 1):
                        self.sub_Normal.addConstr(1/self.input_data.house_0_match[p]*fk0_j[j,p,t-self.input_data.house_install_time[0]] + quicksum(fk_wj[w,j,p,t-self.input_data.wj_t[w][j]] for w in range(args.W) if (t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point + 1)) + quicksum(fk_ij[i,j,p,t] for i in range(args.I)) == yk[j,p,p-1,t]) 
                    else:
                        self.sub_Normal.addConstr(quicksum(fk_wj[w,j,p,t-self.input_data.wj_t[w][j]] for w in range(args.W) if (t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point + 1)) + quicksum(fk_ij[i,j,p,t] for i in range(args.I)) == yk[j,p,p-1,t]) 
                    

        # 1g
        # demand flow
        for j in range(args.J):
            for t in range(args.T-1):
                for g in range(args.G):
                    self.sub_Normal.addConstr(dk[j,g,t+1] == dk[j,g,t] + yk[j,g+1,g,t+1])

        # 1h
        self.demand_cons = []
        for j in range(args.J):
            for g in range(args.G):
                for t in range(self.input_data.first_end_time_point,args.T):
                    temp = self.sub_Normal.addConstr(dk[j,g,t] + qk[j,g,t] == 0)
                    self.demand_cons.append(temp)

        # add the same constriant with sub_R 
        self.Emergency_flow_cons = []
        for i in range(args.I):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(self.input_data.first_end_time_point,args.T):
                        temp = self.sub_Normal.addConstr(fk_ij[i,j,p,t] == 0)
                        self.Emergency_flow_cons.append(temp)

        # 1d
        # second-stage staging area delivery flow limit
        self.staging_area_flow_cons = []
        for w in range(args.W):
            for t in range(self.input_data.first_end_time_point,args.T):
                temp = self.sub_Normal.addConstr(quicksum(self.input_data.house_volumn_factor[p]*fk_wj[w,j,p,t] for p in range(args.P) for j in range(args.J)) <= self.input_data.mid_staging_area_flow[w]) 
                self.staging_area_flow_cons.append(temp)
    
        # 1e
        # novel house match
        self.novel_house_match_cons = []
        for j in range(args.J):
            for t in range(args.T):
                temp = self.sub_Normal.addConstr(quicksum(fk_wj[w,j,0,t] for w in range(args.W)) - quicksum(fk0_j[j,p,t+self.input_data.house_install_time[0]] for p in range(1,args.P) if (t + self.input_data.house_install_time[0]) <= args.T-1) == 0) 
                self.novel_house_match_cons.append(temp)

    def run(self,args,k,x_vals,m_vals,v_vals):

        # add first-stage decision to second-stage model
        # 2c

        for i in range(args.I):
            for p in range(args.P):
                if(abs(v_vals[i,p,self.input_data.first_end_time_point]) < nonint_ratio):
                    # self.S_initial_inventory_cons[i*args.P+p].rhs = 0
                    self.S_initial_inventory_cons[i*args.P+p].setAttr(GRB.Attr.RHS, 0)
                else:
                    # self.S_initial_inventory_cons[i*args.P+p].rhs = v_vals[i,p,self.input_data.first_end_time_point]
                    self.S_initial_inventory_cons[i*args.P+p].setAttr(GRB.Attr.RHS, v_vals[i,p,self.input_data.first_end_time_point])

        # 1b
        for w in range(args.W):
            for p in range(args.P):
                if(abs(m_vals[w,p,self.input_data.first_end_time_point]) < nonint_ratio):
                    # self.W_initial_inventory_cons[w*args.P+p].rhs = 0
                    self.W_initial_inventory_cons[w*args.P+p].setAttr(GRB.Attr.RHS, 0)
                else:
                    # self.W_initial_inventory_cons[w*args.P+p].rhs = m_vals[w,p,self.input_data.first_end_time_point]
                    self.W_initial_inventory_cons[w*args.P+p].setAttr(GRB.Attr.RHS, m_vals[w,p,self.input_data.first_end_time_point])

        # 1c
        for w in range(args.W):
            for p in range(args.P):
                for t in range(args.T-self.input_data.first_end_time_point-1):
                    if(abs(sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t
                            [i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0)) < nonint_ratio):
                        # self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].rhs = 0
                        self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].setAttr(GRB.Attr.RHS,0)
                    else:
                        # self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].rhs = sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for i in range(args.I))
                        self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].setAttr(GRB.Attr.RHS,sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t
                            [i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0))
        
        
        # 1h
        for j in range(args.J):
            for g in range(args.G):
                for t in range(args.T-self.input_data.first_end_time_point):
                    # self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].rhs = self.input_data.demand[k][g][j]
                    self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].setAttr(GRB.Attr.RHS,self.input_data.demand[k][g][j])

        if(args.UR == "reset"):
            self.sub_Normal.reset()
        else:
            self.sub_Normal.update()
        self.sub_Normal.setParam("OutputFlag", 0)
        # self.sub_Normal.setParam('FeasibilityTol', 1e-09)
        # self.sub_Normal.setParam('OptimalityTol', 1e-09)
        self.sub_Normal.optimize()

        ##

        if self.sub_Normal.status == GRB.OPTIMAL:
            0
            # print(self.sub_Normal.ObjVal)
        else:
            print(k, "Normal infesible or unbounded")
            for w in range(args.W):
                for t in range(args.T):
                    print("x:",w,t, sum(self.input_data.house_volumn_factor[p]*x_vals[i,w,p,t] for p in range(args.P) for i in range(args.I)))

            for w in range(args.W):
                for t in range(args.T):
                    print("w:",w,t, sum(self.input_data.house_volumn_factor[p]*m_vals[w,p,t] for p in range(args.P)))
            pdb.set_trace()


        pi_3 = np.zeros((args.W,args.T-self.input_data.first_end_time_point))
        pi_2b = np.zeros((args.W, args.P))
        pi_2c = np.zeros((args.W,args.P,args.T-self.input_data.first_end_time_point-1))
        pi_2d = np.zeros((args.W,args.T-self.input_data.first_end_time_point))
        pi_4c = np.zeros((args.I,args.P))
        pi_2h = np.zeros((args.J,args.G,args.T-self.input_data.first_end_time_point))

        Gk = 0

        # print(k,":",sub.ObjVal)

        sol_temp = 0

        # dual solution
        for j in range(args.J):
            for g in range(args.G):
                for t in range(args.T-self.input_data.first_end_time_point):
                    pi_2h[j][g][t] = self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].pi
                    if(dual == 1):
                        sol_temp = sol_temp + pi_2h[j][g][t]*self.input_data.demand[k][g][j]
                    Gk = Gk + pi_2h[j][g][t]*self.input_data.demand[k][g][j] 

        for w in range(args.W):
            for t in range(args.T-self.input_data.first_end_time_point):
                pi_3[w][t] = self.W_Capacity_cons[w*(args.T-self.input_data.first_end_time_point)+t].pi
                if(dual == 1):
                    sol_temp = sol_temp + pi_3[w][t]*self.input_data.staging_area_capacity[w]
                Gk = Gk + pi_3[w][t]*self.input_data.staging_area_capacity[w]

        for w in range(args.W):
            for p in range(args.P):
                pi_2b[w][p] = self.W_initial_inventory_cons[w*args.P+p].pi
                if(dual == 1):
                    sol_temp = sol_temp + pi_2b[w][p]*m_vals[w,p,self.input_data.first_end_time_point]

        
        for w in range(args.W):
            for p in range(args.P):
                for t in range(args.T-self.input_data.first_end_time_point-1):
                    pi_2c[w][p][t] = self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].pi
                    if(dual == 1):
                        sol_temp = sol_temp + pi_2c[w][p][t]*sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0)        


        for i in range(args.I):
            for p in range(args.P):
                pi_4c[i][p] = self.S_initial_inventory_cons[i*args.P+p].pi
                if(dual == 1):
                    sol_temp = sol_temp + pi_4c[i][p]*v_vals[i,p,self.input_data.first_end_time_point]

            
        for w in range(args.W):
            for t in range(args.T-self.input_data.first_end_time_point):
                pi_2d[w][t] = self.staging_area_flow_cons[w*(args.T-self.input_data.first_end_time_point)+t].pi
                if(dual == 1):
                    sol_temp = sol_temp + self.input_data.mid_staging_area_flow[w]*pi_2d[w][t]
                Gk = Gk + self.input_data.mid_staging_area_flow[w]*pi_2d[w][t]

        if(dual == 1):
           if(abs(sol_temp-self.sub_Normal.ObjVal) >= dual_primal_ratio):
                print("Primal and Dual are not match Normal")
                print("dula:",sol_temp)
                print("primal:",self.sub_Normal.ObjVal)
                pdb.set_trace()

        return pi_3,pi_2b,pi_2c,pi_2d,pi_4c,pi_2h,Gk,self.sub_Normal.ObjVal
    def run_evaluation(self,args,k,x_vals,m_vals,v_vals,new_dir_path):

        # add first-stage decision to second-stage model
        for i in range(args.I):
            for p in range(args.P):
                if(abs(v_vals[i,p,self.input_data.first_end_time_point].x) < nonint_ratio):
                    # self.S_initial_inventory_cons[i*args.P+p].rhs = 0
                    self.S_initial_inventory_cons[i*args.P+p].setAttr(GRB.Attr.RHS, 0)
                else:
                    # self.S_initial_inventory_cons[i*args.P+p].rhs = v_vals[i,p,self.input_data.first_end_time_point]
                    self.S_initial_inventory_cons[i*args.P+p].setAttr(GRB.Attr.RHS, v_vals[i,p,self.input_data.first_end_time_point].x)

        # 1b
        for w in range(args.W):
            for p in range(args.P):
                if(abs(m_vals[w,p,self.input_data.first_end_time_point].x) < nonint_ratio):
                    # self.W_initial_inventory_cons[w*args.P+p].rhs = 0
                    self.W_initial_inventory_cons[w*args.P+p].setAttr(GRB.Attr.RHS, 0)
                else:
                    # self.W_initial_inventory_cons[w*args.P+p].rhs = m_vals[w,p,self.input_data.first_end_time_point]
                    self.W_initial_inventory_cons[w*args.P+p].setAttr(GRB.Attr.RHS, m_vals[w,p,self.input_data.first_end_time_point].x)

        # 1c
        for w in range(args.W):
            for p in range(args.P):
                for t in range(args.T-self.input_data.first_end_time_point-1):
                    if(abs(sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t
                            [i][w]].x for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0)) < nonint_ratio):
                        # self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].rhs = 0
                        self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].setAttr(GRB.Attr.RHS,0)
                    else:
                        # self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].rhs = sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for i in range(args.I))
                        self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].setAttr(GRB.Attr.RHS,sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t
                            [i][w]].x for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0))
        
        
        # 1h
        for j in range(args.J):
            for g in range(args.G):
                for t in range(args.T-self.input_data.first_end_time_point):
                    # self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].rhs = self.input_data.demand[k][g][j]
                    self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].setAttr(GRB.Attr.RHS,self.input_data.demand[k][g][j])

        if(args.UR == "reset"):
            self.sub_Normal.reset()
        else:
            self.sub_Normal.update()

        self.sub_Normal.setParam("OutputFlag", 0)
        # self.sub_Normal.setParam('FeasibilityTol', 1e-09)
        # self.sub_Normal.setParam('OptimalityTol', 1e-09)
        self.sub_Normal.optimize()


        if self.sub_Normal.status == GRB.OPTIMAL:
            0
            # print(self.sub_Normal.ObjVal)
        else:
            print(k, k, "Eval: Normal infesible or unbounded")
            pdb.set_trace()

        logistc_cost = sum(self.input_data.area_region_distance[w][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price*self.fk_wj[w,j,p,t].x for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(self.input_data.first_end_time_point,args.T))
        unmet = sum(args.short_factor*self.input_data.unmet[g]*self.input_data.house_price[g+1]*(self.qk[j,g,args.T-1].x) for j in range(args.J) for g in range(args.G))
        unmet1 = sum((self.qk[j,0,args.T-1].x) for j in range(args.J))
        unmet2 = sum((self.qk[j,1,args.T-1].x) for j in range(args.J))
        mismatch = sum((self.input_data.mismatch[p-1][g])*self.input_data.house_price[p]*self.yk[j,p,g,t].x for j in range(args.J) for p in range(1,args.P) for g in range(args.G) for t in range(self.input_data.first_end_time_point,args.T))
        unused = sum(self.input_data.unused[p]*self.input_data.house_price[p]*self.mk[w,p,args.T-1].x for w in range(args.W) for p in range(args.P))
        unused1 = sum((self.mk[w,0,args.T-1].x) for w in range(args.W))
        unused2 = sum((self.mk[w,1,args.T-1].x) for w in range(args.W))
        unused3 = sum((self.mk[w,2,args.T-1].x) for w in range(args.W))
        deprivation = sum(args.dep_factor*(self.input_data.deprivation_a0[0] + self.input_data.deprivation_a1[0]*(args.TIU)*(args.T-t))*self.input_data.house_price[g+1]*(self.qk[j,g,t].x) for j in range(args.J) for t in range(self.input_data.deprivation_time_point,args.T) for g in range(args.G))
        emergency_cost = sum((args.emergency_price_factor*self.input_data.house_price[p] + self.input_data.supplier_region_distance[i][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*self.fk_ij[i,j,p,t].x for i in range(args.I) for j in range(args.J) for t in range(self.input_data.first_end_time_point,args.T) for p in range(args.P))
        

        if(args.save_demand_flow == 1):
            new_dir_path = new_dir_path + "/" + str(k) + "_Normal_" + ".csv"
            temp_ = np.zeros((args.J*args.G*args.T,5))
            for j in range(args.J):
                for g in range(args.G):
                    for t in range(args.T):
                        temp_[j*args.T*args.G+g*args.T+t,0] = j
                        temp_[j*args.T*args.G+g*args.T+t,1] = g
                        temp_[j*args.T*args.G+g*args.T+t,2] = t
                        temp_[j*args.T*args.G+g*args.T+t,3] = self.dk[j,g,t].x
                        temp_[j*args.T*args.G+g*args.T+t,4] = self.input_data.demand[k][g][j]

            df_name = ["j","g","t",'d',"dd"]
            # data = np.concatenate((temp_J,temp_G), axis=0)
            # data = np.concatenate((data,temp_T), axis=0)
            # data = np.concatenate((data,temp_D), axis=0)
            # pdb.set_trace()
            df = pd.DataFrame(temp_, columns=[df_name])
            df.to_csv(new_dir_path)


        # print("Normal,",k,":")
        # print("logistc_cost:",logistc_cost)
        # print("unmet:",unmet)
        # print("unmet1:",unmet1)
        # print("unmet1:",unmet2)
        # print("mismatch:",mismatch)
        # print("unused:",unused)
        # print("unused1:",unused1)
        # print("unused2:",unused2)
        # print("unused3:",unused3)
        # print("deprivation:",deprivation)
        # print("emergency_cost:",emergency_cost)
        # import pdb;pdb.set_trace()



        # pdb.set_trace()        
        
        return self.sub_Normal.ObjVal,logistc_cost,unmet,mismatch,unused,deprivation,emergency_cost,unmet1,unmet2,unused1,unused2,unused3

class sub_R():
    def __init__(self,args,input_data):

        self.input_data = input_data

        self.sub_Recovery = gp.Model("subprob_Recovery");
        vk = self.sub_Recovery.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Vk_ipt')
        nuk = self.sub_Recovery.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Nuk_ipt')
        fk_ij = self.sub_Recovery.addVars(args.I, args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_ijpt')
        fk_wj = self.sub_Recovery.addVars(args.W, args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_wjpt')
        fk0_j = self.sub_Recovery.addVars(args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk0_jpt')
        yk = self.sub_Recovery.addVars(args.J, args.P, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='yk_jpgt')
        mk = self.sub_Recovery.addVars(args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Mk_wpt')
        bk = self.sub_Recovery.addVars(args.I, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='bk_it')
        dk = self.sub_Recovery.addVars(args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='dk_jgt')
        qk = self.sub_Recovery.addVars(args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='qk_jgt')
        
        self.vk = vk
        self.nuk = nuk
        self.fk_ij = fk_ij
        self.fk_wj = fk_wj
        self.fk0_j = fk0_j
        self.yk = yk
        self.mk = mk
        self.bk = bk
        self.dk = dk
        self.qk = qk



        self.sub_Recovery.setObjective(quicksum(self.input_data.area_region_distance[w][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price*fk_wj[w,j,p,t] for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(self.input_data.first_end_time_point,args.T))
                + quicksum(args.short_factor*self.input_data.unmet[g]*self.input_data.house_price[g+1]*(qk[j,g,args.T-1]) for j in range(args.J) for g in range(args.G))
                + quicksum(self.input_data.unused[p]*self.input_data.house_price[p]*mk[w,p,args.T-1] for w in range(args.W) for p in range(args.P))
                + quicksum(args.dep_factor*(self.input_data.deprivation_a0[0] + self.input_data.deprivation_a1[0]*(args.TIU)*(args.T-t))*self.input_data.house_price[g+1]*(qk[j,g,t]) for j in range(args.J) for t in range(self.input_data.deprivation_time_point,args.T) for g in range(args.G))
                + quicksum((args.emergency_price_factor*self.input_data.supplier_price[i][p]  + self.input_data.supplier_region_distance[i][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*fk_ij[i,j,p,t] for i in range(args.I) for j in range(args.J) for t in range(self.input_data.first_end_time_point,args.T) for p in range(args.P))
                ,GRB.MINIMIZE);


        # self.sub_Recovery.setObjective(quicksum(self.input_data.area_region_distance[w][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price*fk_wj[w,j,p,t] for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(self.input_data.first_end_time_point,args.T))
        #         + quicksum(args.short_factor*self.input_data.unmet[g]*self.input_data.house_price[g+1]*(qk[j,g,args.T-1]) for j in range(args.J) for g in range(args.G))
        #         + quicksum(self.input_data.unused[p]*self.input_data.house_price[p]*mk[w,p,args.T-1] for w in range(args.W) for p in range(args.P))
        #         + quicksum((self.input_data.deprivation_a0[0] + self.input_data.deprivation_a1[0]*(args.TIU)*(args.T-t))*self.input_data.house_price[g+1]*(qk[j,g,t]) for j in range(args.J) for t in range(self.input_data.deprivation_time_point,args.T) for g in range(args.G))
        #         ,GRB.MINIMIZE);

        
        if(args.modular == 0):
            for i in range(args.I):
                for j in range(args.J):
                    for t in range(args.T):
                        self.sub_Recovery.addConstr(fk_ij[i,j,0,t] == 0)
                        # for p in range(args.P):
                        #     self.sub_Recovery.addConstr(fk_ij[i,j,p,t] == 0)

        
        for i in range(args.I):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.sub_Recovery.addConstr(nuk[i,p,t] == 0)
                    if(t != self.input_data.first_end_time_point):
                        self.sub_Recovery.addConstr(vk[i,p,t] == 0)

        for w in range(args.W):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(0,self.input_data.first_end_time_point+1):
                        self.sub_Recovery.addConstr(fk_wj[w,j,p,t] == 0)

        for j in range(args.J):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.sub_Recovery.addConstr(fk0_j[j,p,t] == 0)

        for i in range(args.I):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(0,self.input_data.first_end_time_point+1):
                        self.sub_Recovery.addConstr(fk_ij[i,j,p,t] == 0)

        for j in range(args.J):
            for p in range(args.P):
                for g in range(args.G):
                    for t in range(0,self.input_data.first_end_time_point+1):
                        self.sub_Recovery.addConstr(yk[j,p,g,t] == 0)

        for w in range(args.W):
            for p in range(args.P):
                for t in range(0,self.input_data.first_end_time_point):
                    self.sub_Recovery.addConstr(mk[w,p,t] == 0)

        # 2g
        for i in range(args.I):
            for t in range(0,self.input_data.first_end_time_point+2):
                self.sub_Recovery.addConstr(bk[i,t] == 0)

        for j in range(args.J):
            for g in range(args.G):
                for t in range(0,self.input_data.first_end_time_point+1):
                    self.sub_Recovery.addConstr(dk[j,g,t] == 0)
                    if(t != self.input_data.first_end_time_point):
                        self.sub_Recovery.addConstr(qk[j,g,t] == 0)

        for t in range(args.T):
            for j in range(args.J):
                for p in range(args.P):
                    for g in range(args.G):
                        if(g != p-1):
                            self.sub_Recovery.addConstr(yk[j,p,g,t] == 0)

        
        # 2c
        self.S_initial_inventory_cons = []
        # Second-stage Supplier initial Inventory
        for i in range(args.I):
            for p in range(args.P):
                temp = self.sub_Recovery.addConstr(vk[i,p,self.input_data.first_end_time_point] == 0)
                self.S_initial_inventory_cons.append(temp)

        # 1b
        self.W_initial_inventory_cons = []
        # Second-stage staging area initail inventory
        for w in range(args.W):
            for p in range(args.P):
                temp = self.sub_Recovery.addConstr(mk[w,p,self.input_data.first_end_time_point] == 0)
                self.W_initial_inventory_cons.append(temp)

        
        # master -> sub : capacity
        self.W_Capacity_cons = []
        # Statgin area Capacity (second-stage)
        for w in range(args.W):
            for t in range(self.input_data.first_end_time_point,args.T):
                temp = self.sub_Recovery.addConstr(quicksum(self.input_data.house_volumn_factor[p]*mk[w,p,t] for p in range(args.P)) <= self.input_data.staging_area_capacity[w])
                self.W_Capacity_cons.append(temp)

        # 1c
        self.W_inventory_flow_cons = []
        #Second-stage stating area inventory flow
        for w in range(args.W):
                for p in range(args.P):
                    for t in range(self.input_data.first_end_time_point,args.T-1):
                        temp = self.sub_Recovery.addConstr(mk[w,p,t+1] + quicksum(fk_wj[w,j,p,t+1] for j in range(args.J) if t+1+self.input_data.wj_t[w][j] <= args.T-1) - mk[w,p,t] == 0)
                        self.W_inventory_flow_cons.append(temp)

        # 2f
        # match flow
        for j in range(args.J):
            for t in range(self.input_data.first_end_time_point,args.T):
                for p in range(1,args.P):
                    if(t - self.input_data.house_install_time[0] >= self.input_data.first_end_time_point + 1):
                        self.sub_Recovery.addConstr(1/self.input_data.house_0_match[p]*fk0_j[j,p,t-self.input_data.house_install_time[0]] + quicksum(fk_wj[w,j,p,t-self.input_data.wj_t[w][j]] for w in range(args.W) if t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point+1) + quicksum(fk_ij[i,j,p,t-self.input_data.ij_t[i][j]] for i in range(args.I) if t-self.input_data.ij_t[i][j]>= self.input_data.first_end_time_point+1) == yk[j,p,p-1,t]) 
                    else:
                       self.sub_Recovery.addConstr(quicksum(fk_wj[w,j,p,t-self.input_data.wj_t[w][j]] for w in range(args.W) if t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point+1) + quicksum(fk_ij[i,j,p,t-self.input_data.ij_t[i][j]] for i in range(args.I) if t-self.input_data.ij_t[i][j]>= self.input_data.first_end_time_point+1) == yk[j,p,p-1,t]) 
                    
        # 1g
        # demand flow
        for j in range(args.J):
            for t in range(args.T-1):
                for g in range(args.G):
                    self.sub_Recovery.addConstr(dk[j,g,t+1] == dk[j,g,t] + yk[j,g+1,g,t+1])
        
        # 1h
        self.demand_cons = []
        for j in range(args.J):
            for g in range(args.G):
                for t in range(self.input_data.first_end_time_point,args.T):
                    temp = self.sub_Recovery.addConstr(dk[j,g,t] + qk[j,g,t] == 0)
                    self.demand_cons.append(temp)


        # 2h
        # emergency production
        self.emergency_product_cons = []
        for i in range(args.I):
            for t in range(self.input_data.first_end_time_point,args.T-1):
                temp = self.sub_Recovery.addConstr(bk[i,t+1] + quicksum(self.input_data.production_ratio_E[i][p]*nuk[i,p,t+1-self.input_data.low_supplier_prod_time_TIU[i][p]] for p in range(args.P) if t+1-self.input_data.low_supplier_prod_time_TIU[i][p] >= self.input_data.first_end_time_point+1) == quicksum(self.input_data.production_ratio_E[i][p]*nuk[i,p,t] for p in range(args.P) if t+self.input_data.low_supplier_prod_time_TIU[i][p] <= args.T-1) + bk[i,t]) 
                self.emergency_product_cons.append(temp)

        # 2i
        # extra production line in emergency modality
        self.extra_production_cons = []
        for i in range(args.I):
            for t in range(self.input_data.first_end_time_point,args.T):
                temp = self.sub_Recovery.addConstr(bk[i,t] <= self.input_data.extra_production_line[i])
                self.extra_production_cons.append(temp)

        # 2d
        # Second-stage Supplier Inventory flow
        self.supplier_inventory_cons = []
        for i in range(args.I):
            for p in range(args.P):
                for t in range(self.input_data.first_end_time_point,args.T-1):
                    if(t-self.input_data.low_supplier_prod_time_TIU[i][p] >= self.input_data.first_end_time_point + 1):
                        temp = self.sub_Recovery.addConstr(vk[i,p,t+1]  + quicksum(fk_ij[i,j,p,t+1] for j in range(args.J) if t+1+self.input_data.ij_t[i][j] <= args.T-1) - vk[i,p,t] - self.input_data.production_ratio_E[i][p]*nuk[i,p,t-self.input_data.low_supplier_prod_time_TIU[i][p]] == 0)
                    else:
                        temp = self.sub_Recovery.addConstr(vk[i,p,t+1]  + quicksum(fk_ij[i,j,p,t+1] for j in range(args.J) if t+1+self.input_data.ij_t[i][j] <= args.T-1) - vk[i,p,t]  == 0)
                    self.supplier_inventory_cons.append(temp)

        # 2e
        # novel house match
        self.novel_house_match_cons = []
        for j in range(args.J):
            for t in range(args.T):
                temp = self.sub_Recovery.addConstr(quicksum(fk_ij[i,j,0,t-self.input_data.ij_t[i][j]] for i in range(args.I) if t-self.input_data.ij_t[i][j] >= self.input_data.first_end_time_point+1) + quicksum(fk_wj[w,j,0,t-self.input_data.wj_t[w][j]] for w in range(args.W) if t-self.input_data.wj_t[w][j] >= self.input_data.first_end_time_point+1) - quicksum(fk0_j[j,p,t] for p in range(1,args.P) if t + self.input_data.house_install_time[0] <= args.T-1) == 0) 
                self.novel_house_match_cons.append(temp)
                

    def run(self,args,k,s_vals,x_vals,m_vals,v_vals):

        # add first-stage decision to second-stage model

        # 2c
        for i in range(args.I):
            for p in range(args.P):
                if(abs(v_vals[i,p,self.input_data.first_end_time_point]) < nonint_ratio):
                    self.S_initial_inventory_cons[i*args.P+p].setAttr(GRB.Attr.RHS, 0)
                # self.S_initial_inventory_cons[i*args.P+p].rhs = v_vals[i,p,self.input_data.first_end_time_point]
                else:
                    self.S_initial_inventory_cons[i*args.P+p].setAttr(GRB.Attr.RHS, v_vals[i,p,self.input_data.first_end_time_point])

        # 1b
        for w in range(args.W):
            for p in range(args.P):
                if(abs(m_vals[w,p,self.input_data.first_end_time_point]) < nonint_ratio):
                    # self.W_initial_inventory_cons[w*args.P+p].rhs = 0
                    self.W_initial_inventory_cons[w*args.P+p].setAttr(GRB.Attr.RHS, 0)
                else:
                    # self.W_initial_inventory_cons[w*args.P+p].rhs = m_vals[w,p,self.input_data.first_end_time_point]
                    self.W_initial_inventory_cons[w*args.P+p].setAttr(GRB.Attr.RHS, m_vals[w,p,self.input_data.first_end_time_point])

        # 1c
        for w in range(args.W):
            for p in range(args.P):
                for t in range(args.T-self.input_data.first_end_time_point-1):
                    if(abs(sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t
                            [i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0)) < nonint_ratio):
                        # self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].rhs = 0
                        self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].setAttr(GRB.Attr.RHS,0)
                    else:
                        # self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].rhs = sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for i in range(args.I))
                        self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].setAttr(GRB.Attr.RHS,sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t
                            [i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0))
         # 1h
        for j in range(args.J):
            for g in range(args.G):
                for t in range(args.T-self.input_data.first_end_time_point):
                    # self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].rhs = self.input_data.demand[k][g][j]
                    self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].setAttr(GRB.Attr.RHS,self.input_data.demand[k][g][j])

        # 2d
        for i in range(args.I):
            for p in range(args.P):
                for t in range(args.T-1-self.input_data.first_end_time_point):
                    if(t+self.input_data.first_end_time_point-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0):
                        if(abs(self.input_data.production_ratio_N[i][p]*s_vals[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point] - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1] for w in range(args.W))) < nonint_ratio):
                            self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].setAttr(GRB.Attr.RHS, 0)
                        else:
                        # self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].rhs = s_vals[i,p,t-self.input_data.mid_supplier_prod_time[i][p]+self.input_data.first_end_time_point] - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for w in range(args.W))
                            self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].setAttr(GRB.Attr.RHS, self.input_data.production_ratio_N[i][p]*s_vals[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point] - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1] for w in range(args.W)))
                    else:
                        if(abs(sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1] for w in range(args.W))) < nonint_ratio):
                            self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].setAttr(GRB.Attr.RHS, 0)
                        else:
                            self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].setAttr(GRB.Attr.RHS, - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1] for w in range(args.W)))
                   
                        
        

        if(args.UR == "reset"):
            self.sub_Recovery.reset()
        else:
            self.sub_Recovery.update()
        self.sub_Recovery.setParam("OutputFlag", 0)
        # self.sub_Recovery.setParam('FeasibilityTol', 1e-09)
        # self.sub_Recovery.setParam('OptimalityTol', 1e-09)
        self.sub_Recovery.optimize()
        
        #np.savetxt("update_VBasis.txt",self.sub_Recovery.VBasis)
        #np.savetxt("update_CBasis.txt",self.sub_Recovery.CBasis)
        

        # solution_pool_size = self.sub_Recovery.getAttr('SolCount')
        # print ("Solution pool contains {0} solutions".format(solution_pool_size))
        

        if self.sub_Recovery.status == GRB.OPTIMAL:
            0
            # print(self.sub_Recovery.ObjVal)
        else:
            print(k, "Recovery infesible or unbounded")
            for i in range(args.I):
                for p in range(args.P):
                    for t in range(args.T-1):
                        if(t-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0):
                            print(t,i,p, "v+1 / x+1 = v / s" , v_vals[i,p,t+1], sum(x_vals[i,w,p,t+1] for w in range(args.W)), v_vals[i,p,t], self.input_data.production_ratio_N[i][p]*s_vals[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]])
                        else:
                            print(t,i,p, "v+1 / x+1 = v / s" , v_vals[i,p,t+1], sum(x_vals[i,w,p,t+1] for w in range(args.W)), v_vals[i,p,t],0)
            pdb.set_trace()


        pi_3 = np.zeros((args.W,args.T-self.input_data.first_end_time_point))
        pi_2b = np.zeros((args.W, args.P))
        pi_2c = np.zeros((args.W,args.P,args.T-self.input_data.first_end_time_point-1))
        pi_4c = np.zeros((args.I,args.P))
        pi_2h = np.zeros((args.J,args.G,args.T-self.input_data.first_end_time_point))
        pi_4d = np.zeros((args.I,args.P,args.T-1-self.input_data.first_end_time_point))
        pi_4i = np.zeros((args.I,args.T-self.input_data.first_end_time_point))

        # print(k,":",sub.ObjVal)

        Gk=0

        sol_temp = 0

        # dual solution
        for j in range(args.J):
            for g in range(args.G):
                for t in range(args.T-self.input_data.first_end_time_point):
                    pi_2h[j][g][t] = self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].pi
                    if(dual == 1):
                        sol_temp = sol_temp + pi_2h[j][g][t]*self.input_data.demand[k][g][j]
                    Gk = Gk + pi_2h[j][g][t]*self.input_data.demand[k][g][j]

        for w in range(args.W):
            for t in range(args.T-self.input_data.first_end_time_point):
                pi_3[w][t] = self.W_Capacity_cons[w*(args.T-self.input_data.first_end_time_point)+t].pi
                if(dual == 1):    
                    sol_temp = sol_temp + pi_3[w][t]*self.input_data.staging_area_capacity[w]
                Gk = Gk + pi_3[w][t]*self.input_data.staging_area_capacity[w]

        for w in range(args.W):
            for p in range(args.P):
                pi_2b[w][p] = self.W_initial_inventory_cons[w*args.P+p].pi
                if(dual == 1):
                    sol_temp = sol_temp + pi_2b[w][p]*m_vals[w,p,self.input_data.first_end_time_point]

        
        for w in range(args.W):
            for p in range(args.P):
                for t in range(args.T-self.input_data.first_end_time_point-1):
                    pi_2c[w][p][t] = self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].pi
                    if(dual == 1):
                        sol_temp = sol_temp + pi_2c[w][p][t]*sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0)


        for i in range(args.I):
            for p in range(args.P):
                pi_4c[i][p] = self.S_initial_inventory_cons[i*args.P+p].pi
                if(dual == 1):
                    sol_temp = sol_temp + pi_4c[i][p]*v_vals[i,p,self.input_data.first_end_time_point]

            
        for i in range(args.I):
            for p in range(args.P):
                for t in range(args.T-1-self.input_data.first_end_time_point):
                    pi_4d[i][p][t] = self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].pi
                    if(dual == 1):
                        sol_temp = sol_temp + (self.input_data.production_ratio_N[i][p]*sum(s_vals[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point] for temp in range(1) if(t+self.input_data.first_end_time_point-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0)) - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1] for w in range(args.W)))*pi_4d[i][p][t]

        for i in range(args.I):
            for t in range(args.T-self.input_data.first_end_time_point):
                pi_4i[i][t] = self.extra_production_cons[i*(args.T-self.input_data.first_end_time_point)+t].pi
                if(dual == 1):
                    sol_temp = sol_temp + self.input_data.extra_production_line[i]*pi_4i[i][t]
                Gk = Gk + self.input_data.extra_production_line[i]*pi_4i[i][t] 

        
        if(dual == 1):
            if(abs(sol_temp-self.sub_Recovery.ObjVal) >= dual_primal_ratio):
                print("Primal and Dual are not match Emergency")
                print("dula:",sol_temp)
                print("primal:",self.sub_Recovery.ObjVal)
                pdb.set_trace()
        
        
        
        # global temp_out
        # if(temp_out == 1):
        # temp_out = temp_out + 1
        # if(k == 0):
        #     for j in range(args.J):
        #         for g in range(args.G):
        #             for t in range(args.T):
        #                 print(j,g,t,self.qk[j,g,t].x)


        # for i in range(args.I):
        #     for p in range(args.P):
        #         for t in range(args.T-1):
        #             if(t-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0):
        #                 a = v_vals[i,p,t+1] + sum(x_vals[i,w,p,t] for w in range(args.W))
        #                 b = v_vals[i,p,t] + (1/self.input_data.mid_supplier_prod_time[i][p])*(self.input_data.mid_supplier_prod_time_TIU[i][p])*args.TIU*s_vals[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]]
        #                 if( a-b  >= nonint_ratio):
        #                     print(a,b)
        #             else:
        #                 a = v_vals[i,p,t+1] + sum(x_vals[i,w,p,t] for w in range(args.W))
        #                 b = v_vals[i,p,t]
        #                 if( a-b  >= nonint_ratio):
        #                     print(a,b)

        return pi_3,pi_2b,pi_2c,pi_4c,pi_2h,pi_4d,pi_4i,Gk,self.sub_Recovery.ObjVal

    def run_evaluation(self,args,k,s_vals,x_vals,m_vals,v_vals,new_dir_path):

        # add first-stage decision to second-stage model

        
        # 2c
        for i in range(args.I):
            for p in range(args.P):
                if(abs(v_vals[i,p,self.input_data.first_end_time_point].x) < nonint_ratio):
                    self.S_initial_inventory_cons[i*args.P+p].setAttr(GRB.Attr.RHS, 0)
                # self.S_initial_inventory_cons[i*args.P+p].rhs = v_vals[i,p,self.input_data.first_end_time_point]
                else:
                    self.S_initial_inventory_cons[i*args.P+p].setAttr(GRB.Attr.RHS, v_vals[i,p,self.input_data.first_end_time_point].x)

        # 1b
        for w in range(args.W):
            for p in range(args.P):
                if(abs(m_vals[w,p,self.input_data.first_end_time_point].x) < nonint_ratio):
                    # self.W_initial_inventory_cons[w*args.P+p].rhs = 0
                    self.W_initial_inventory_cons[w*args.P+p].setAttr(GRB.Attr.RHS, 0)
                else:
                    # self.W_initial_inventory_cons[w*args.P+p].rhs = m_vals[w,p,self.input_data.first_end_time_point]
                    self.W_initial_inventory_cons[w*args.P+p].setAttr(GRB.Attr.RHS, m_vals[w,p,self.input_data.first_end_time_point].x)

        # 1c
        for w in range(args.W):
            for p in range(args.P):
                for t in range(args.T-self.input_data.first_end_time_point-1):
                    if(abs(sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t
                            [i][w]].x for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0)) < nonint_ratio):
                        # self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].rhs = 0
                        self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].setAttr(GRB.Attr.RHS,0)
                    else:
                        # self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].rhs = sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for i in range(args.I))
                        self.W_inventory_flow_cons[w*args.P*(args.T-self.input_data.first_end_time_point-1)+p*(args.T-self.input_data.first_end_time_point-1)+t].setAttr(GRB.Attr.RHS,sum(x_vals[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t
                            [i][w]].x for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0))
         # 1h
        for j in range(args.J):
            for g in range(args.G):
                for t in range(args.T-self.input_data.first_end_time_point):
                    # self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].rhs = self.input_data.demand[k][g][j]
                    self.demand_cons[j*args.G*(args.T-self.input_data.first_end_time_point) + g*(args.T-self.input_data.first_end_time_point) + t].setAttr(GRB.Attr.RHS,self.input_data.demand[k][g][j])

        # 2d
        for i in range(args.I):
            for p in range(args.P):
                for t in range(args.T-1-self.input_data.first_end_time_point):
                    if(t+self.input_data.first_end_time_point-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0):
                        if(abs(self.input_data.production_ratio_N[i][p]*s_vals[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point].x - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1].x for w in range(args.W))) < nonint_ratio):
                            self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].setAttr(GRB.Attr.RHS, 0)
                        else:
                        # self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].rhs = s_vals[i,p,t-self.input_data.mid_supplier_prod_time[i][p]+self.input_data.first_end_time_point] - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for w in range(args.W))
                            self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].setAttr(GRB.Attr.RHS, self.input_data.production_ratio_N[i][p]*s_vals[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point].x - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1].x for w in range(args.W)))
                    else:
                        if(abs(sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1].x for w in range(args.W))) < nonint_ratio):
                            self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].setAttr(GRB.Attr.RHS, 0)
                        else:
                            self.supplier_inventory_cons[i*args.P*(args.T-1-self.input_data.first_end_time_point)+p*(args.T-1-self.input_data.first_end_time_point)+t].setAttr(GRB.Attr.RHS, - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point+1].x for w in range(args.W)))
            

        if(args.UR == "reset"):
            self.sub_Recovery.reset()
        else:
            self.sub_Recovery.update()
        self.sub_Recovery.setParam("OutputFlag", 0)
        self.sub_Recovery.optimize()

        # solution_pool_size = self.sub_Recovery.getAttr('SolCount')
        # print ("Solution pool contains {0} solutions".format(solution_pool_size))        

        if self.sub_Recovery.status == GRB.OPTIMAL:
            0
            # print(self.sub_Recovery.ObjVal)
        else:
            print(k, k, "Eval: Recovery infesible or unbounded")
            pdb.set_trace()


        logistc_cost = sum(self.input_data.area_region_distance[w][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price*self.fk_wj[w,j,p,t].x for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(self.input_data.first_end_time_point,args.T))
        unmet = sum(args.short_factor*self.input_data.unmet[g]*self.input_data.house_price[g+1]*(self.qk[j,g,args.T-1].x) for j in range(args.J) for g in range(args.G))
        unmet1 = sum((self.qk[j,0,args.T-1].x) for j in range(args.J))
        unmet2 = sum((self.qk[j,1,args.T-1].x) for j in range(args.J))
        mismatch = sum((self.input_data.mismatch[p-1][g])*self.input_data.house_price[p]*self.yk[j,p,g,t].x for j in range(args.J) for p in range(1,args.P) for g in range(args.G) for t in range(self.input_data.first_end_time_point,args.T))
        unused = sum(self.input_data.unused[p]*self.input_data.house_price[p]*self.mk[w,p,args.T-1].x for w in range(args.W) for p in range(args.P))
        unused1 = sum((self.mk[w,0,args.T-1].x) for w in range(args.W))
        unused2 = sum((self.mk[w,1,args.T-1].x) for w in range(args.W))
        unused3 = sum((self.mk[w,2,args.T-1].x) for w in range(args.W))
        deprivation = sum(args.dep_factor*(self.input_data.deprivation_a0[0] + self.input_data.deprivation_a1[0]*(args.TIU)*(args.T-t))*self.input_data.house_price[g+1]*(self.qk[j,g,t].x) for j in range(args.J) for t in range(self.input_data.deprivation_time_point,args.T) for g in range(args.G))
        emergency_cost = sum((args.emergency_price_factor*self.input_data.house_price[p] + self.input_data.supplier_region_distance[i][j]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*self.fk_ij[i,j,p,t].x for i in range(args.I) for j in range(args.J) for t in range(args.T) for p in range(args.P))
        emergency_acqurie0 = sum(self.fk_ij[i,j,0,t].x for i in range(args.I) for j in range(args.J) for t in range(args.T))
        emergency_acqurie1 = sum(self.fk_ij[i,j,1,t].x for i in range(args.I) for j in range(args.J) for t in range(args.T))
        emergency_acqurie2 = sum(self.fk_ij[i,j,2,t].x for i in range(args.I) for j in range(args.J) for t in range(args.T))
    

        if(args.save_demand_flow == 1):
            new_dir_path = new_dir_path + "/" + str(k) + "_Recovery_" + ".csv"
            temp_ = np.zeros((args.J*args.G*args.T,5))
            for j in range(args.J):
                for g in range(args.G):
                    for t in range(args.T):
                        temp_[j*args.T*args.G+g*args.T+t,0] = j
                        temp_[j*args.T*args.G+g*args.T+t,1] = g
                        temp_[j*args.T*args.G+g*args.T+t,2] = t
                        temp_[j*args.T*args.G+g*args.T+t,3] = self.dk[j,g,t].x
                        temp_[j*args.T*args.G+g*args.T+t,4] = self.input_data.demand[k][g][j]

            df_name = ["j","g","t",'d',"dd"]
            # data = np.concatenate((temp_J,temp_G), axis=0)
            # data = np.concatenate((data,temp_T), axis=0)
            # data = np.concatenate((data,temp_D), axis=0)
            # pdb.set_trace()
            df = pd.DataFrame(temp_, columns=[df_name])
            df.to_csv(new_dir_path)


        # print("Recovery,",k,":")
        # print("logistc_cost:",logistc_cost)
        # print("unmet:",unmet)
        # print("unmet1:",unmet1)
        # print("unmet1:",unmet2)
        # print("mismatch:",mismatch)
        # print("unused:",unused)
        # print("unused1:",unused1)
        # print("unused2:",unused2)
        # print("unused3:",unused3)
        # print("deprivation:",deprivation)
        # print("emergency_cost:",emergency_cost)



        return self.sub_Recovery.ObjVal,logistc_cost,unmet,mismatch,unused,deprivation,emergency_cost,unmet1,unmet2,unused1,unused2,unused3,emergency_acqurie0,emergency_acqurie1,emergency_acqurie2

class master():
    def __init__(self,args,input_data,sub_N,sub_R,index):


        ### ------------------ Model ------------------ ###

        self.input_data = input_data
        self.sub_N = sub_N
        self.sub_R = sub_R
        self.index = index

        self.heu_cut = []


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
        # self.model.setObjective(quicksum((self.input_data.house_price[p] + self.input_data.supplier_area_distance[i][w]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*self.x[i,w,p,t] for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T)) + self.theta,GRB.MINIMIZE);


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

        # 3c
        # Recovery plan constraint
        self.model.addConstr(quicksum(self.z[k] for k in range(args.K)) == args.error)

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
        

        # self.model.setParam("OutputFlag", 0)

    def check_MIPSOL(self,args,model,where):
        if where == GRB.Callback.MIPSOL:
            global node
            node = node + 1


            # Second-stage
            # print("master solution:",model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
            x_vals = model.cbGetSolution(model._xvars)
            s_vals = model.cbGetSolution(model._svars)
            m_vals = model.cbGetSolution(model._mvars)
            v_vals = model.cbGetSolution(model._vvars)
            z_vals = model.cbGetSolution(model._zvars)
            # z_vals = np.around(z_vals)
            theta_vals = model.cbGetSolution(model.getVars()[-1])

            pi_3 = np.zeros((2,args.K,args.W,args.T-self.input_data.first_end_time_point))
            pi_2b = np.zeros((2,args.K,args.W, args.P))
            pi_2c = np.zeros((2,args.K,args.W,args.P,args.T-self.input_data.first_end_time_point-1))
            pi_2d = np.zeros((2,args.K,args.W,args.T-self.input_data.first_end_time_point))
            pi_4c = np.zeros((2,args.K,args.I, args.P))
            pi_4d = np.zeros((2,args.K,args.I,args.P,args.T-1-self.input_data.first_end_time_point))
            pi_4i = np.zeros((2,args.K,args.I,args.T-self.input_data.first_end_time_point))
            pi_2h = np.zeros((2,args.K,args.J,args.G,args.T-self.input_data.first_end_time_point))
            Gk = np.zeros((2,args.K))

            M1 = np.zeros((args.K))
            M2 = np.zeros((args.K))
            zvars_round = np.zeros((args.K))
            for k in range(args.K):
                zvars_round[k] = np.round(z_vals[k])
            # zvars_round = int(zvars_round)

            dual_opt = np.zeros((2,args.K))

            global solving_sub_Normal_problem,solving_sub_Emergency_problem
            
            for k in range(args.K):
                if(abs(z_vals[k]) <= 1e-5):
                    start = time.time()
                    pi_3[0][k],pi_2b[0][k],pi_2c[0][k],pi_2d[0][k],pi_4c[0][k],pi_2h[0][k],Gk[0][k],dual_opt[0][k] = self.sub_N.run(args,k,x_vals,m_vals,v_vals)
                    end = time.time()
                    solving_sub_Normal_problem = solving_sub_Normal_problem + end - start
                else:
                    start = time.time()
                    if(args.model != "TSCC_without"):
                        pi_3[1][k],pi_2b[1][k],pi_2c[1][k],pi_4c[1][k],pi_2h[1][k],pi_4d[1][k],pi_4i[1][k],Gk[1][k],dual_opt[1][k] = self.sub_R.run(args,k,s_vals,x_vals,m_vals,v_vals)
                    end = time.time()
                    solving_sub_Emergency_problem = solving_sub_Emergency_problem + end - start

            # print("MM:",M1,M2)
            # zvars_round = np.round(z_vals)
            lower_bound = (1/args.K)*sum(dual_opt[int(np.round(z_vals[k]))][k] for k in range(args.K))
            # print("---------------------------:",lower_bound)

            if(args.cut == "bigM" or args.cut == "both"):

                start = time.time()
                if(theta_vals < lower_bound - cut_vio_thred and abs(theta_vals - lower_bound)/max(abs(theta_vals),1e-10) > cut_vio_thred):
                    for k in range(args.K):   
                        M1[k] = sum(self.input_data.staging_area_capacity[w]*pi_2b[0][k][w][p] for w in range(args.W) for p in range(args.P) if pi_2b[0][k][w][p] > 1e-5) 
                        M2[k] = sum(self.input_data.staging_area_capacity[w]*pi_2b[1][k][w][p] for w in range(args.W) for p in range(args.P) if pi_2b[1][k][w][p] > 1e-5)
                        # print("M1:",M1[k])
                        M1[k] = M1[k] + sum(pi_2h[0][k][j][g][t]*self.input_data.demand[k][g][j] for j in range(args.J) for g in range(args.G) for t in range(args.T-self.input_data.first_end_time_point) if pi_2h[0][k][j][g][t] > 1e-5) 
                        M2[k] = M2[k] + sum(pi_2h[1][k][j][g][t]*self.input_data.demand[k][g][j] for j in range(args.J) for g in range(args.G) for t in range(args.T-self.input_data.first_end_time_point) if pi_2h[1][k][j][g][t] > 1e-5) 
                        # print("M1:",M1[k])
                        M1[k] = M1[k] + sum(self.input_data.staging_area_capacity[w]*pi_2c[0][k][w][p][t] for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if pi_2c[0][k][w][p][t] > 1e-5) 
                        M2[k] = M2[k] + sum(self.input_data.staging_area_capacity[w]*pi_2c[1][k][w][p][t] for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if pi_2c[1][k][w][p][t] > 1e-5) 
                        # print("M1:",M1[k])
                        M1[k] = M1[k] + sum((self.input_data.mid_supplier_inventory[i][p] + (self.first_end_time_point)*self.input_data.num_production_line[i])*pi_4c[0][k][i][p] for i in range(args.I) for p in range(args.P) if pi_4c[0][k][i][p] > 1e-5)
                        M2[k] = M2[k] + sum((self.input_data.mid_supplier_inventory[i][p] + (self.first_end_time_point)*self.input_data.num_production_line[i])*pi_4c[1][k][i][p] for i in range(args.I) for p in range(args.P) if pi_4c[1][k][i][p] > 1e-5)
                        # print("M1:",M1[k])
                        M2[k] = M2[k] + sum((self.input_data.num_production_line[i])*pi_4d[1][k][i][p][t] for t in range(args.T-1-self.input_data.first_end_time_point) for p in range(args.P) for i in range(args.I) if pi_4d[1][k][i][p][t] > 1e-5)
              
                    #print("MM:",M1,M2)

                    global big_M_cut
                    big_M_cut = big_M_cut + 1

                    model.cbLazy(model.getVars()[-1] + quicksum(M1[k]*model._zvars[k] for k in range(args.K) if abs(z_vals[k]) < nonint_ratio) + quicksum(M2[k]*(1-model._zvars[k]) for k in range(args.K) if abs(z_vals[k]-1) < nonint_ratio) 
                        >= (1/args.K)*(quicksum(quicksum(pi_2h[0][k][j][g][t]*self.input_data.demand[k][g][j] for g in range(args.G) for j in range(args.J) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                    + quicksum(pi_3[0][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                    + quicksum(pi_2b[0][k][w][p]*model._mvars[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 0))
                                                    + quicksum(pi_2c[0][k][w][p][t]*quicksum(model._xvars[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 0))
                                                    + quicksum(pi_4c[0][k][i][p]*model._vvars[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 0))
                                                    + quicksum(self.input_data.mid_staging_area_flow[w]*pi_2d[0][k][w][t] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                    + quicksum(pi_2h[1][k][j][g][t]*self.input_data.demand[k][g][j] for j in range(args.J) for g in range(args.G) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                    + quicksum(pi_3[1][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                    + quicksum(pi_2b[1][k][w][p]*model._mvars[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 1))
                                                    + quicksum(pi_2c[1][k][w][p][t]*quicksum(model._xvars[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 1))
                                                    + quicksum(pi_4c[1][k][i][p]*model._vvars[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 1))
                                                    + quicksum((self.input_data.production_ratio_N[i][p]*quicksum(model._svars[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point] for temp in range(1) if(t+self.input_data.first_end_time_point-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0)) - quicksum(model._xvars[i,w,p,t+self.input_data.first_end_time_point+1] for w in range(args.W)))*pi_4d[1][k][i][p][t] for i in range(args.I) for p in range(args.P) for t in range(args.T-1-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                    + quicksum(self.input_data.extra_production_line[i]*pi_4i[1][k][i][t] for i in range(args.I) for t in range(args.T-self.input_data.first_end_time_point) if (zvars_round[k] == 1)) for k in range(args.K))))

                    # temp_dual_eva = ((1/args.K)*(sum(sum(pi_2h[0][k][j][g][t]*self.input_data.demand[k][g][j] for g in range(args.G) for j in range(args.J) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                    #                                 + sum(pi_3[0][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                    #                                 + sum(pi_2b[0][k][w][p]*m_vals[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 0))
                    #                                 + sum(pi_2c[0][k][w][p][t]*sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for i in range(args.I)) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 0))
                    #                                 + sum(pi_4c[0][k][i][p]*v_vals[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 0))
                    #                                 + sum(self.input_data.mid_staging_area_flow[w]*pi_2d[0][k][w][t] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                    #                                 + sum(pi_2h[1][k][j][g][t]*self.input_data.demand[k][g][j] for j in range(args.J) for g in range(args.G) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                    #                                 + sum(pi_3[1][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                    #                                 + sum(pi_2b[1][k][w][p]*m_vals[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 1))
                    #                                 + sum(pi_2c[1][k][w][p][t]*sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for i in range(args.I)) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 1))
                    #                                 + sum(pi_4c[1][k][i][p]*v_vals[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 1))
                    #                                 + sum((s_vals[i,p,t-self.input_data.mid_supplier_prod_time[i][p]+self.input_data.first_end_time_point] - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for w in range(args.W)))*pi_4d[1][k][i][p][t] for i in range(args.I) for p in range(args.P) for t in range(args.T-1-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                    #                                 + sum(self.input_data.extra_production_line[i]*pi_4i[1][k][i][t] for i in range(args.I) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1)) for k in range(args.K))))

                    # if (abs(temp_dual_eva-lower_bound)>nonint_ratio):
                    #     print("dual_eva:",temp_dual_eva)
                    #     print("dual_cal:",lower_bound)
                
                                    
                # print("bigM:")
                # print("theta:",theta_vals)
                # print("lower bound",lower_bound)
                # print(k,"M1:",M1[k],"M2:",M2[k])
                end = time.time()
                global generate_bigM_cut_time
                generate_bigM_cut_time = generate_bigM_cut_time + (end - start)

                # Special cut
                start = time.time()

            if(args.cut == "Spe" or args.cut == "both"):

                # print("M0_like:",sum((Gk_reg-Gk[int(z_vals[k])][k])*z_vals[k] for k in range(self.args.K) if z_vals[k] == 0))
                # print("M1_like:",sum((Gk_bar-Gk[int(z_vals[k])][k])*z_vals[k] for k in range(self.args.K) if z_vals[k] == 1))
                start = time.time()
                if(theta_vals < lower_bound - cut_vio_thred and abs(theta_vals - lower_bound)/max(abs(theta_vals),1e-10) > cut_vio_thred):

                    for k in range(args.K):
                        if(abs(z_vals[k]) <= 1e-5):
                            start = time.time()
                            if(args.model != "TSCC_without"):
                                pi_3[1][k],pi_2b[1][k],pi_2c[1][k],pi_4c[1][k],pi_2h[1][k],pi_4d[1][k],pi_4i[1][k],Gk[1][k],dual_opt[1][k] = self.sub_R.run(args,k,s_vals,x_vals,m_vals,v_vals)
                            end = time.time()
                            solving_sub_Emergency_problem = solving_sub_Emergency_problem + end - start
                        else:
                            start = time.time()
                            pi_3[0][k],pi_2b[0][k],pi_2c[0][k],pi_2d[0][k],pi_4c[0][k],pi_2h[0][k],Gk[0][k],dual_opt[0][k] = self.sub_N.run(args,k,x_vals,m_vals,v_vals)
                            end = time.time()
                            solving_sub_Normal_problem = solving_sub_Normal_problem + end - start

                    Gk_reg = -1
                    Gk_bar = -1
                    for k in range(args.K):
                        if(np.round(z_vals[k]) == 0):
                            if(Gk_bar == -1):
                                Gk_bar = Gk[1][k]
                                continue
                            if(Gk[1][k] < Gk_bar):
                                Gk_bar = Gk[1][k]
                        else:
                            if(Gk_reg == -1):
                                Gk_reg = Gk[0][k]
                                continue
                            if(Gk[0][k] < Gk_reg):
                                Gk_reg = Gk[0][k]
                    

                    global spe_cut
                    spe_cut = spe_cut + 1

                    model.cbLazy(model.getVars()[-1] - (1/args.K)*quicksum((Gk_reg-Gk[0][k])*model._zvars[k] for k in range(args.K) if(zvars_round[k] == 0)) - (1/args.K)*quicksum((Gk_bar-Gk[1][k])*(1-model._zvars[k]) for k in range(args.K) if(zvars_round[k] == 1))
                            >= (1/args.K)*(quicksum(quicksum(pi_2h[0][k][j][g][t]*self.input_data.demand[k][g][j] for g in range(args.G) for j in range(args.J) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                    + quicksum(pi_3[0][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                    + quicksum(pi_2b[0][k][w][p]*model._mvars[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 0))
                                                    + quicksum(pi_2c[0][k][w][p][t]*quicksum(model._xvars[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 0))
                                                    + quicksum(pi_4c[0][k][i][p]*model._vvars[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 0))
                                                    + quicksum(self.input_data.mid_staging_area_flow[w]*pi_2d[0][k][w][t] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                    + quicksum(pi_2h[1][k][j][g][t]*self.input_data.demand[k][g][j] for j in range(args.J) for g in range(args.G) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                    + quicksum(pi_3[1][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                    + quicksum(pi_2b[1][k][w][p]*model._mvars[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 1))
                                                    + quicksum(pi_2c[1][k][w][p][t]*quicksum(model._xvars[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 1))
                                                    + quicksum(pi_4c[1][k][i][p]*model._vvars[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 1))
                                                    + quicksum((self.input_data.production_ratio_N[i][p]*quicksum(model._svars[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point] for temp in range(1) if(t+self.input_data.first_end_time_point-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0)) - quicksum(model._xvars[i,w,p,t+self.input_data.first_end_time_point+1] for w in range(args.W)))*pi_4d[1][k][i][p][t] for i in range(args.I) for p in range(args.P) for t in range(args.T-1-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                    + quicksum(self.input_data.extra_production_line[i]*pi_4i[1][k][i][t] for i in range(args.I) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1)) for k in range(args.K))))
                    
                    
                    if(args.cut_save == 1):
                    
                        global cut_count
                        path_temp =  "cut/{}_{}_{}_cut_{}.npz".format(args.K,args.error,args.TIU,cut_count,cut_count)
                        cut_count = cut_count+1
                        np.savez(path_temp,zvars_round = zvars_round,
                                            M1 = M1,
                                            M2 = M2,
                                            Gk_reg = Gk_reg,
                                            Gk = Gk,
                                            Gk_bar = Gk_bar,
                                            pi_2b = pi_2b,
                                            pi_4c = pi_4c,
                                            pi_2d = pi_2d,
                                            pi_2h = pi_2h,
                                            pi_3 = pi_3,
                                            pi_2c = pi_2c,
                                            pi_4i = pi_4i,
                                            pi_4d = pi_4d)
                        

                    

                    # temp_dual_eva = ((1/args.K)*(sum(sum(pi_2h[0][k][j][g][t]*self.input_data.demand[k][g][j] for g in range(args.G) for j in range(args.J) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                    #             + sum(pi_3[0][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                    #             + sum(pi_2b[0][k][w][p]*m_vals[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 0))
                    #             + sum(pi_2c[0][k][w][p][t]*sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for i in range(args.I)) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 0))
                    #             + sum(pi_4c[0][k][i][p]*v_vals[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 0))
                    #             + sum(self.input_data.mid_staging_area_flow[w]*pi_2d[0][k][w][t] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                    #             + sum(pi_2h[1][k][j][g][t]*self.input_data.demand[k][g][j] for j in range(args.J) for g in range(args.G) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                    #             + sum(pi_3[1][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                    #             + sum(pi_2b[1][k][w][p]*m_vals[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 1))
                    #             + sum(pi_2c[1][k][w][p][t]*sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for i in range(args.I)) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 1))
                    #             + sum(pi_4c[1][k][i][p]*v_vals[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 1))
                    #             + sum((s_vals[i,p,t-self.input_data.mid_supplier_prod_time[i][p]+self.input_data.first_end_time_point] - sum(x_vals[i,w,p,t+self.input_data.first_end_time_point] for w in range(args.W)))*pi_4d[1][k][i][p][t] for i in range(args.I) for p in range(args.P) for t in range(args.T-1-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                    #             + sum(self.input_data.extra_production_line[i]*pi_4i[1][k][i][t] for i in range(args.I) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1)) for k in range(args.K))))

                    # if (abs(temp_dual_eva-lower_bound)>nonint_ratio):
                    #     print("dual_eva:",temp_dual_eva)
                    #     print("dual_cal:",lower_bound)


                # print("Spe:")
                # print("theta:",theta_vals)
                # print("lower bound",lower_bound)
                # print(k,"M1:",sum((Gk_reg-Gk[int(z_vals[k])][k])*z_vals[k] for k in range(self.args.K) if z_vals[k] == 0),"M2:",sum((Gk_bar-Gk[int(z_vals[k])][k])*(1-z_vals[k]) for k in range(self.args.K) if z_vals[k] == 1))
                end = time.time()
                global generate_special_cut_time
                generate_special_cut_time = generate_special_cut_time + (end-start)
            # print(z_vals)

    def run_eval(self,args,s,x,m,v):

        e_eval=args.error

        second_obj_total = 0
        logistc_cost_total = 0
        unmet_total = 0
        unmet1_total = 0
        unmet2_total = 0
        mismatch_total = 0
        unused_total = 0
        unused1_total = 0
        unused2_total = 0
        unused3_total = 0
        deprivation_total = 0
        emergency_cost_total = 0
        emergency_acqurie0_total = 0
        emergency_acqurie1_total = 0
        emergency_acqurie2_total = 0

        second_obj = np.zeros((2,args.K))
        logistc_cost = np.zeros((2,args.K))
        unmet = np.zeros((2,args.K))
        unmet1 = np.zeros((2,args.K))
        unmet2 = np.zeros((2,args.K))
        mismatch = np.zeros((2,args.K))
        unused = np.zeros((2,args.K))
        unused1 = np.zeros((2,args.K))
        unused2 = np.zeros((2,args.K))
        unused3 = np.zeros((2,args.K))
        deprivation = np.zeros((2,args.K))
        emergency_cost = np.zeros((2,args.K))
        emergency_acqurie0 = np.zeros((2,args.K))
        emergency_acqurie1 = np.zeros((2,args.K))
        emergency_acqurie2 = np.zeros((2,args.K))

        

        new_dir_path = args.result_path + "/" + str(args.model) + "_k_" + str(args.K) + "e_" + str(args.error) + "_" + str(args.cut) + "_modular_" + str(args.modular) + "_" + str(args.UR) + "_" + str(args.output_name) + "_TIU_" + str(args.TIU) + "_sf_" + str(int(args.short_factor*10)) + "_df_" + str(int(args.dep_factor*10)) + "_mis_" + str(int(args.mis_factor*10)) + "_un_" + str(int(args.unused_factor*10))
        if(os.path.exists(new_dir_path) == False):
            os.mkdir(new_dir_path)


        first_stage_cost = sum((self.input_data.supplier_price[i][p] + self.input_data.supplier_area_distance[i][w]*self.input_data.house_volumn_factor[p]*self.input_data.trans_price)*x[i,w,p,t].x for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T))
        type0 = sum(x[i,w,0,t].x for i in range(args.I) for w in range(args.W) for t in range(args.T))
        type1 = sum(x[i,w,1,t].x for i in range(args.I) for w in range(args.W) for t in range(args.T))
        type2 = sum(x[i,w,2,t].x for i in range(args.I) for w in range(args.W) for t in range(args.T))
        temp_zk = []
        index = []
        output_index = []
        second_obj = np.zeros((2,args.K))
        
        v_star = 0

        for k in range(args.K):
            second_obj[0][k],logistc_cost[0][k],unmet[0][k],mismatch[0][k],unused[0][k],deprivation[0][k],emergency_cost[0][k],unmet1[0][k],unmet2[0][k],unused1[0][k],unused2[0][k],unused3[0][k] = self.sub_N.run_evaluation(args,k,x,m,v,new_dir_path)
            temp_zk.append(second_obj[0][k])

            second_obj[1][k],logistc_cost[1][k],unmet[1][k],mismatch[1][k],unused[1][k],deprivation[1][k],emergency_cost[1][k],unmet1[1][k],unmet2[1][k],unused1[1][k],unused2[1][k],unused3[1][k],emergency_acqurie0[1][k],emergency_acqurie1[1][k],emergency_acqurie2[1][k] = self.sub_R.run_evaluation(args,k,s,x,m,v,new_dir_path)
            temp_zk[k] = temp_zk[k]-second_obj[1][k]
            
            # print("Normal/Emergency",second_obj[0][k],second_obj[1][k])
            # print("Normal/Emergency:unmet",unmet[0][k],unmet[1][k])
            # print("Normal/Emergency:unmet1",unmet1[0][k],unmet1[1][k])
            # print("Normal/Emergency:unmet2",unmet2[0][k],unmet2[1][k])
            # print("Normal/Emergency:emergency_acqurie0",emergency_acqurie0[0][k],emergency_acqurie0[1][k])
            # print("Normal/Emergency:emergency_acqurie1",emergency_acqurie1[0][k],emergency_acqurie1[1][k])
            # print("Normal/Emergency:emergency_acqurie2",emergency_acqurie2[0][k],emergency_acqurie2[1][k])
            # print("Normal/Emergency:unused",unused[0][k],unused[1][k])
            # print("Normal/Emergency:deprivation",deprivation[0][k],deprivation[1][k])

            # pdb.set_trace()

            if(args.model == "TSCC_De_Sp_class" and args.Large_Scenario_main == 0):
                if(abs(self.z[k].x-1) <= nonint_ratio):
                    output_index.append(k)
            
        index = np.flip(np.argsort(temp_zk))


        # pdb.set_trace()


        for k in range(args.K):
            if (k < e_eval):
                temp = 1
            else:
                temp = 0

            # print("--------------------------------------------------")
            # print([index[k]])
            # print(unused1[temp][index[k]])
            # print(unused2[temp][index[k]])
            # print(unused3[temp][index[k]])
            
            second_obj_total = second_obj_total + second_obj[temp][index[k]]
            logistc_cost_total = logistc_cost_total + logistc_cost[temp][index[k]]
            unmet_total = unmet_total + unmet[temp][index[k]]
            mismatch_total = mismatch_total + mismatch[temp][index[k]]
            unused_total = unused_total + unused[temp][index[k]]
            deprivation_total = deprivation_total + deprivation[temp][index[k]]
            emergency_cost_total = emergency_cost_total + emergency_cost[temp][index[k]]
            unmet1_total = unmet1_total + unmet1[temp][index[k]]
            unmet2_total = unmet2_total + unmet2[temp][index[k]]
            unused1_total = unused1_total + unused1[temp][index[k]]
            unused2_total = unused2_total + unused2[temp][index[k]]
            unused3_total = unused3_total + unused3[temp][index[k]]

            emergency_acqurie0_total = emergency_acqurie0_total + emergency_acqurie0[temp][index[k]]
            emergency_acqurie1_total = emergency_acqurie1_total + emergency_acqurie1[temp][index[k]]
            emergency_acqurie2_total = emergency_acqurie2_total + emergency_acqurie2[temp][index[k]]

            
            if(temp == 0):
                if(v_star <= second_obj[temp][index[k]]):
                    v_star = second_obj[temp][index[k]]


            # print("logistc_cost:",logistc_cost[temp][index[k]])
            # print("unmet:",unmet[temp][index[k]])
            # print("unmet1:",unmet1[temp][index[k]])
            # print("unmet2:",unmet2[temp][index[k]])
            # print("mismatch:",mismatch[temp][index[k]])
            # print("unused:",unused[temp][index[k]])
            # print("unused1:",unused1[temp][index[k]])
            # print("unused2:",unused2[temp][index[k]])
            # print("unused3:",unused3[temp][index[k]])
            # print("deprivation:",deprivation[temp][index[k]])
            # print("emergency_cost:",emergency_cost[temp][index[k]])

        # print("run_evaluation: ", first_stage_cost+second_obj_total/args.K)
        # print("second_obj_total: ", second_obj_total/args.K)
        # print("evaluation_index:", index[:e_eval])
        # if(args.model == "TSCC_De_Sp_class"):
        #     print("calculation_index:", output_index)
        

        # else:
        #     for k in range(args.K):
        #         if(abs(self.z[k].x) < nonint_ratio):
        #             second_obj,logistc_cost,unmet,mismatch,unused,deprivation,emergency_cost,unmet1,unmet2,unused1,unused2,unused3 = self.sub_N.run_evaluation(args,k,x,m,v)
        #         else:
        #             second_obj,logistc_cost,unmet,mismatch,unused,deprivation,emergency_cost,unmet1,unmet2,unused1,unused2,unused3 = self.sub_R.run_evaluation(args,k,s,x,m,v)

        #         second_obj_total = second_obj_total + second_obj
        #         logistc_cost_total = logistc_cost_total + logistc_cost
        #         unmet_total = unmet_total + unmet
        #         mismatch_total = mismatch_total + mismatch
        #         unused_total = unused_total + unused
        #         deprivation_total = deprivation_total + deprivation
        #         emergency_cost_total = emergency_cost_total + emergency_cost
        #         unmet1_total = unmet1_total + unmet1
        #         unmet2_total = unmet2_total + unmet2
        #         unused1_tota = unused1_total + unused1
        #         unused2_tota = unused2_total + unused2
        #         unused3_tota = unused3_total + unused3

        df_name = ["back up active","OPT","V_star","first_stage_trans_cost","type0","type1","type2",
                   "second_stage_cots","second_trans_cost","Emergency_cost","unused_cost","unused0","unused1","unused2","mismatch_cost","unmet_cost","shortage 1","shortage 2","deprivation","E_acquire0","E_acquire1","E_acquire2"]
        data = [[e_eval, 
                first_stage_cost+(second_obj_total/args.K),
                v_star,
                first_stage_cost,
                type0,
                type1,
                type2,
                second_obj_total/args.K, 
                logistc_cost_total/args.K, 
                emergency_cost_total/args.K,
                unused_total/args.K,
                unused1_total,
                unused2_total,
                unused3_total, 
                mismatch_total/args.K, 
                unmet_total/args.K,
                unmet1_total, 
                unmet2_total,
                deprivation_total/args.K,
                emergency_acqurie0_total,
                emergency_acqurie1_total,
                emergency_acqurie2_total]]
        df = pd.DataFrame(data, columns=[df_name])
        name = args.result_path + "/" + str(args.model) + "_k_" + str(args.K) + "e_" + str(args.error) + "_" + str(args.cut) + "_modular_" + str(args.modular) + "_" + str(args.UR) + "_" + str(args.output_name) + "_TIU_" + str(args.TIU) + "_sf_" + str(int(args.short_factor*10)) + "_df_" + str(int(args.dep_factor*10)) + "_mis_" + str(int(args.mis_factor*10)) + "_un_" + str(int(args.unused_factor*10)) + ".csv"
        df.to_csv(name)

        result_name =  new_dir_path + "/" + "z" + ".txt"
        f = open(result_name,"w+")
        f.write("z: %s" % output_index)
        opt =  first_stage_cost+second_obj_total/args.K
        if(args.Large_Scenario_main == 1):
            sample_test_name = "1000_eval" + "_e_" + str(args.error) + ".txt"
            f_generate = open(sample_test_name,"a")
            f_generate.write("\nopt: %s" % opt)
            
            
        threshold_policy_value = np.zeros(args.threshold_policy_replication)
        threshold_em_ac_num = 0
            
        if(args.threshold_policy == 1):
            func.scenario_generation(args, args.threshold_policy_replication)

            trailer_name = "data/Scenario/trailer_" + str(args.threshold_policy_replication) +".txt"
            MHU_name = "data/Scenario/MHU_" + str(args.threshold_policy_replication) + ".txt"

            self.demand_per_trailer = np.loadtxt(trailer_name)
            self.demand_per_MHU = np.loadtxt(MHU_name)

            for j in range(0,args.J):
                for k in range(0,args.threshold_policy_replication):
                    self.demand_per_trailer[k][j] = self.input_data.homeowner_occupied[j]*self.demand_per_trailer[k][j]
                    self.demand_per_MHU[k][j] = self.input_data.homeowner_occupied[j]*self.demand_per_MHU[k][j]

            self.input_data.demand = np.stack((self.demand_per_trailer,self.demand_per_MHU),axis=1)
            self.input_data.demand = np.round(self.input_data.demand)

            # pdb.set_trace()


            for k in range(0,args.threshold_policy_replication):
                v_temp,logistc_cost[0][0],unmet[0][0],mismatch[0][0],unused[0][0],deprivation[0][0],emergency_cost[0][0],unmet1[0][0],unmet2[0][0],unused1[0][0],unused2[0][0],unused3[0][0] = self.sub_N.run_evaluation(args,k,x,m,v,new_dir_path)

                if(v_temp >= v_star):
                    v_temp,logistc_cost[1][0],unmet[1][0],mismatch[1][0],unused[1][0],deprivation[1][0],emergency_cost[1][0],unmet1[1][0],unmet2[1][0],unused1[1][0],unused2[1][0],unused3[1][0] = self.sub_R.run_evaluation(args,k,s,x,m,v,new_dir_path)
                    threshold_em_ac_num = threshold_em_ac_num + 1
                
                #print(v_temp)
                threshold_policy_value[k] = v_temp

            df_v = pd.DataFrame(threshold_policy_value, columns=['cost'])
            name = args.result_path + "/" + "e_" + str(args.error) + ".csv"
            df_v.to_csv(name)
            print(threshold_em_ac_num)



        return first_stage_cost+second_obj_total/args.K
        

    def run(self,args):
        
        
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
        #self.model.setParam('NumericFocus', 2)
        #self.model.setParam('IntFeasTol', 1e-09)
        #self.model.setParam('FeasibilityTol', 1e-09)
        #self.model.setParam('OptimalityTol', 1e-09)


        if(args.Heu == 1):

            temp_cons = []
            for k in range(args.K):
                if (k < args.error):
                    temp = self.model.addConstr(self.z[self.index[k]] == 1)
                else:
                    temp = self.model.addConstr(self.z[self.index[k]] == 0)
                temp_cons.append(temp)

            self.model.update()
            start = time.time()
            self.model.optimize(lambda model, where: self.check_MIPSOL(args,model, where))
            self.model.write("heu/{}_{}_{}_heu_solution.sol".format(args.K,args.error,args.TIU))
            end = time.time()

            result_name = args.result_path + "/" + str(args.model) + "_k_" + str(args.K) + "e_" + str(args.error) + "_" + str(args.cut) + "_modular_" + str(args.modular) + "_" + str(args.UR) + "_" + str(args.output_name) + "_TIU_" + str(args.TIU) + ".txt"
            f = open(result_name,"a")
            print((end - start))
            f.write("Heu-TSCC total process time: %s \r" % (end - start))
            f.close()

            for temp in temp_cons:
                    self.model.remove(temp_cons)

            # Add Lazy constraint generated by 2ssP
            for count in range(cut_count):
                # print("---------cut_count---------------",count)
                path_temp =  "cut/{}_{}_{}_cut_{}.npz".format(args.K,args.error,args.TIU,count)
                loaded_arrays = np.load(path_temp, allow_pickle = True)
                zvars_round = loaded_arrays['zvars_round']
                M1 = loaded_arrays['M1']
                M2 = loaded_arrays['M2']
                Gk_reg = loaded_arrays['Gk_reg']
                Gk = loaded_arrays['Gk']
                Gk_bar = loaded_arrays['Gk_bar']
                pi_2b = loaded_arrays['pi_2b']
                pi_4c = loaded_arrays['pi_4c']
                pi_2d = loaded_arrays['pi_2d']
                pi_2h = loaded_arrays['pi_2h']
                pi_3 = loaded_arrays['pi_3']
                pi_2c = loaded_arrays['pi_2c']
                pi_4i = loaded_arrays['pi_4i']
                pi_4d = loaded_arrays['pi_4d']
                os.remove(path_temp)

                self.model.addConstr(self.theta - (1/args.K)*quicksum((Gk_reg-Gk[0][k])*self.z[k] for k in range(args.K) if(zvars_round[k] == 0)) - (1/args.K)*quicksum((Gk_bar-Gk[1][k])*(1-self.z[k]) for k in range(args.K) if(zvars_round[k] == 1))
                                >= (1/args.K)*(quicksum(quicksum(pi_2h[0][k][j][g][t]*self.input_data.demand[k][g][j] for g in range(args.G) for j in range(args.J) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                        + quicksum(pi_3[0][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                        + quicksum(pi_2b[0][k][w][p]*self.m[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 0))
                                                        + quicksum(pi_2c[0][k][w][p][t]*quicksum(self.x[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 0))
                                                        + quicksum(pi_4c[0][k][i][p]*self.v[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 0))
                                                        + quicksum(self.input_data.mid_staging_area_flow[w]*pi_2d[0][k][w][t] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 0))
                                                        + quicksum(pi_2h[1][k][j][g][t]*self.input_data.demand[k][g][j] for j in range(args.J) for g in range(args.G) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                        + quicksum(pi_3[1][k][w][t]*self.input_data.staging_area_capacity[w] for w in range(args.W) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                        + quicksum(pi_2b[1][k][w][p]*self.m[w,p,self.input_data.first_end_time_point] for w in range(args.W) for p in range(args.P) if(zvars_round[k] == 1))
                                                        + quicksum(pi_2c[1][k][w][p][t]*quicksum(self.x[i,w,p,t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]] for i in range(args.I) if (t+self.input_data.first_end_time_point-self.input_data.iw_t[i][w]) >= 0) for w in range(args.W) for p in range(args.P) for t in range(args.T-self.input_data.first_end_time_point-1) if(zvars_round[k] == 1))
                                                        + quicksum(pi_4c[1][k][i][p]*self.v[i,p,self.input_data.first_end_time_point] for i in range(args.I) for p in range(args.P) if(zvars_round[k] == 1))
                                                        + quicksum((self.input_data.production_ratio_N[i][p]*quicksum(self.s[i,p,t-self.input_data.mid_supplier_prod_time_TIU[i][p]+self.input_data.first_end_time_point] for temp in range(1) if(t+self.input_data.first_end_time_point-self.input_data.mid_supplier_prod_time_TIU[i][p] >= 0)) - quicksum(self.x[i,w,p,t+self.input_data.first_end_time_point+1] for w in range(args.W)))*pi_4d[1][k][i][p][t] for i in range(args.I) for p in range(args.P) for t in range(args.T-1-self.input_data.first_end_time_point) if(zvars_round[k] == 1))
                                                        + quicksum(self.input_data.extra_production_line[i]*pi_4i[1][k][i][t] for i in range(args.I) for t in range(args.T-self.input_data.first_end_time_point) if(zvars_round[k] == 1)) for k in range(args.K))))
            self.model.update()
            self.model.params.StartNumber = -1
            self.model.read("heu/{}_{}_{}_heu_solution.sol".format(args.K,args.error,args.TIU))


        result_name = args.result_path + "/" + str(args.model) + "_k_" + str(args.K) + "e_" + str(args.error) + "_" + str(args.cut) + "_modular_" + str(args.modular) + "_" + str(args.UR) + "_" + str(args.output_name) + "_TIU_" + str(args.TIU) + "_sf_" + str(int(args.short_factor*10)) + "_df_" + str(int(args.dep_factor*10)) + "_mis_" + str(int(args.mis_factor*10)) + "_un_" + str(int(args.unused_factor*10)) + "_Heu_" + str(args.Heu) + "_cut_save_" + str(args.cut_save) + ".txt"
        args.cut_save = 0
        
        start = time.time()
        if(args.model != "WS"):
            self.model.optimize(lambda model, where: self.check_MIPSOL(args,model, where))
        else:
            print(node)
            self.model.optimize()
        end = time.time()
        time_m = self.model.Runtime
        f = open(result_name,"w+")
        f.write("total process time: %s \r" % (end - start))
        f.write("solving_master_problem: %s \r" % time_m)
        f.write("generate bigM cut time: %s \r" % generate_bigM_cut_time)
        f.write("generate special cut time: %s \r" % generate_special_cut_time)
        f.write("solving sub Normal problem: %s \r" % solving_sub_Normal_problem)
        f.write("solving sub Emergency problem: %s \r" % solving_sub_Emergency_problem)
        f.write("# big M cut: %s \r" % big_M_cut)
        f.write("# special M cut: %s \r" % spe_cut)
        f.write("# node: %s \r" % node)
        f.write("ObjVal: %s \r" % self.model.ObjVal)
        f.write("ObjBound: %s \r" % self.model.ObjBound)

        run_evaluation = self.run_eval(args,self.s,self.x,self.m,self.v)
        print("----------------------------")
        print("z",self.z)
        print("----------------------------")
        

        # if(args.Large == 1):

        # sub_N_large = self.sub_N(args,self.input_data)
        # sub_R_large = self.sub_R(args,self.input_data)
        # model_TSCC_D_large = self.master(args,input_data,sub_N,sub_R)



        if self.model.status == GRB.OPTIMAL:
            reulst_m = self.model.ObjVal
            f.write("final opt: %s" % reulst_m)
            # print("caculate_total_cost:", reulst_m)
            # print("caculate_second_stage_cost:", self.theta.x)
            # print("ratio:", abs(reulst_m-run_evaluation)/run_evaluation)

            if(args.Large_Scenario_main == 1):
                sample_test_name = "50_sample" + "_e_" + str(args.error) + ".txt"
                f_generate = open(sample_test_name,"a")
                f_generate.write("\nopt: %s" % reulst_m)

            return self.x,self.s,self.m,self.v,self.z
        else:
            f.write("final opt: None")