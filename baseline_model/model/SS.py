from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import math

scale_down_cost = 10000

class single_stage_chance():
    def __init__(self, args):
        
        ### ------------------ time point ----------------------- ###

        df = pd.read_excel(args.Timepoint_path)
        self.first_end_time_point = df["First_End"][0]
        self.deprivation_time_point = df["Deprivation_Start"][0]
        self.second_end_time_point = df["Second_End"][0]

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
        # func.scenario_generation(args, args.K)



        ### ------------------ Study Region ------------------ ###

        df = pd.read_excel(args.Study_Region_path)
        study_region_column = list(df.columns)
        study_region_location = df[['latitude','longitude']]
        self.homeowner_occupied = np.zeros(args.J)
        self.demand_per_trailer = np.loadtxt(args.Demand_Trailer_path)
        self.demand_per_MHU = np.loadtxt(args.Demand_MHU_path)
        
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

        global test_array
        test_array = self.demand

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
        
        
        ### ------------------ cost scale down ------------------ ###

        self.house_price = self.house_price/scale_down_cost
        self.supplier_price = self.supplier_price/scale_down_cost
        self.trans_price = self.trans_price/scale_down_cost

        ### ------------------ Model ------------------ ###

        # self.demand[0][0][0] = 100000


        self.model = gp.Model("SS_model")

        # First-stage
        self.x = self.model.addVars(args.I, args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Xiwpt')
        self.s = self.model.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Sipt')
        self.m = self.model.addVars(args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Mwpt')
        self.v = self.model.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Vipt')
        self.a = self.model.addVars(args.I, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Awpt')
        self.z = self.model.addVars(args.K, vtype=GRB.BINARY, name='Zk')

        # Second-stage
        self.f_wj = self.model.addVars(args.K, args.W, args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_wjpt')
        self.f0_j = self.model.addVars(args.K, args.J, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk0_jpt')
        self.y = self.model.addVars(args.K, args.J, args.P, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='yk_jpgt')
        self.d = self.model.addVars(args.K, args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='dk_jgt')
        self.mk = self.model.addVars(args.K, args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Mkwpt')


        if(args.modular == 0):
            for w in range(args.W):
                for p in range(args.P):
                    for t in range(args.T):
                        self.model.addConstr(self.m[w,0,t] == 0)

        if(args.modular == 0):
            for i in range(args.I):
                for w in range(args.W):
                    for t in range(args.T):
                        self.model.addConstr(self.x[i,w,0,t] == 0)



        if(args.modular == 0):
            for k in range(args.K):
                for w in range(args.W):
                    for t in range(args.T):
                        self.model.addConstr(self.mk[k,w,0,t] == 0)



        if(args.modular == 0):
            for w in range(args.W):
                for t in range(args.T):
                    self.model.addConstr(self.m[w,0,t] == 0)



        # Objective
        self.model.setObjective(sum((self.supplier_price[i][p] + self.supplier_area_distance[i][w]*self.house_volumn_factor[p]*self.trans_price)*self.x[i,w,p,t] for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T))
                              ,GRB.MINIMIZE);

        for w in range(args.W):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(0,self.first_end_time_point):
                        for k in range(args.K):
                            self.model.addConstr(self.f_wj[k,w,j,p,t] == 0)

        for j in range(args.J):
            for p in range(args.P):
                for t in range(0,self.first_end_time_point):
                    for k in range(args.K):
                        self.model.addConstr(self.f0_j[k,j,p,t] == 0)

        for j in range(args.J):
            for p in range(args.P):
                for g in range(args.G):
                    for t in range(0,self.first_end_time_point):
                        for k in range(args.K):
                            self.model.addConstr(self.y[k,j,p,g,t] == 0)

        for j in range(args.J):
            for g in range(args.G):
                for t in range(0,self.first_end_time_point):
                    for k in range(args.K):
                        self.model.addConstr(self.d[k,j,g,t] == 0)

        for w in range(args.W):
            for p in range(args.P):
                self.model.addConstr(self.m[w,p,0] == 0)

        for w in range(args.W):
            for p in range(args.P):
                for t in range(0,self.first_end_time_point):
                    self.model.addConstr(self.mk[k,w,p,0] == 0)

        # Recovery plan constraint
        self.model.addConstr(sum(self.z[k] for k in range(args.K)) >= args.K - args.error)

            
        # Initial Machine Capacity && Machine used flow && Every time unit Machine Capacity
        for i in range(args.I):
            self.model.addConstr(self.a[i,0] == 0)
            self.model.addConstr(self.a[i,args.T-1] <= self.num_production_line[i])
            for t in range(args.T-1):
                self.model.addConstr(self.a[i,t+1] + sum(self.s[i,p,t-self.mid_supplier_prod_time[i][p]] for p in range(args.P) if t-self.mid_supplier_prod_time[i][p] >= 0) ==  sum(self.s[i,p,t] for p in range(args.P)) + self.a[i,t])
                self.model.addConstr(self.a[i,t] <= self.num_production_line[i])     

        # Initial Inventory && Inventory flow
        for i in range(args.I):
            for p in range(args.P):
                self.model.addConstr(self.v[i,p,0] == self.mid_supplier_inventory[i][p])
                for t in range(args.T-1):
                    if(t-self.mid_supplier_prod_time[i][p] >= 0):
                        self.model.addConstr(self.v[i,p,t+1] + sum(self.x[i,w,p,t] for w in range(args.W)) ==  self.v[i,p,t] + self.s[i,p,t-self.mid_supplier_prod_time[i][p]])
                    else:
                        self.model.addConstr(self.v[i,p,t+1] + sum(self.x[i,w,p,t] for w in range(args.W)) ==  self.v[i,p,t] )
                    
        # Supply sending flow
        for t in range(args.T):
            for i in range(args.I):
                self.model.addConstr(quicksum(self.house_volumn_factor[p]*self.x[i,w,p,t] for w in range(args.W) for p in range(args.P)) <= self.mid_supplier_flow[i])

        # First-stage Staging area inventory flow limitation
        for w in range(args.W): 
            for t in range(self.first_end_time_point):
                for p in range(args.P):
                    self.model.addConstr(self.m[w,p,t+1] == sum(self.x[i,w,p,t] for i in range(args.I)) + self.m[w,p,t])


        for w in range(args.W): 
            for p in range(args.P):
                for k in range(args.K):
                    self.model.addConstr(self.m[w,p,self.first_end_time_point] == self.mk[k,w,p,self.first_end_time_point])


        #Second-stage stating area inventory flow
        for w in range(args.W):
            for p in range(args.P):
                for t in range(self.first_end_time_point,args.T-1):
                    for k in range(args.K):
                        self.model.addConstr(self.mk[k,w,p,t+1] + sum(self.f_wj[k,w,j,p,t] for j in range(args.J)) == sum(self.x[i,w,p,t] for i in range(args.I)) + self.mk[k,w,p,t])

        # Statgin area Capacity
        for w in range(args.W):
            for t in range(args.T):
                self.model.addConstr(sum(self.house_volumn_factor[p]*self.m[w,p,t] for p in range(args.P)) <= self.staging_area_capacity[w])
                for k in range(args.K):
                    self.model.addConstr(sum(self.house_volumn_factor[p]*self.mk[k,w,p,t] for p in range(args.P)) <= self.staging_area_capacity[w])

        # second-stage staging area delivery flow
        for w in range(args.W):
            for t in range(self.first_end_time_point,args.T):
                for k in range(args.K):
                    self.model.addConstr(sum(self.house_volumn_factor[p]*self.f_wj[k,w,j,p,t] for p in range(args.P) for j in range(args.J)) <= self.mid_staging_area_flow[w]) 

        
        # novel house match
        for j in range(args.J):
            for t in range(args.T):
                for k in range(args.K):
                    self.model.addConstr(sum(self.f_wj[k,w,j,0,t] for w in range(args.W)) == sum(self.f0_j[k,j,p,t+p*self.house_install_time[0]] for p in range(1,args.P) if t + p*self.house_install_time[0] <= args.T-1)) 

        # match flow
        for j in range(args.J):
            for t in range(args.T):
                for p in range(1,args.P):
                    for k in range(args.K):
                        self.model.addConstr(1/self.house_0_match[p]*self.f0_j[k,j,p,t] + sum(self.f_wj[k,w,j,p,t] for w in range(args.W))  == sum(self.y[k,j,p,g,t] for g in range(args.G))) 

        # demand flow
        for j in range(args.J):
            for t in range(args.T-1):
                for g in range(args.G):
                    for k in range(args.K):
                        self.model.addConstr(self.d[k,j,g,t+1] == self.d[k,j,g,t] + sum(self.y[k,j,p,g,t] for p in range(1,args.P)))

        # demand limitation
        for j in range(args.J):
            for k in range(args.K):
                for g in range(args.G):
                    self.model.addConstr(self.d[k,j,g,args.T-1] >= self.demand[k][g][j]*self.z[k])

        self.model.update()

        # import pdb;pdb.set_trace()

    def run(self,args):
        self.model.setParam("OutputFlag", 0)
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print("fesible")
            return self.x,self.s,self.m,self.v
        else:
            print("infesible or unbounded")
            for k in range(args.K):
                for g in range(args.G):
                    print(sum(self.demand[k][g][j] for j in range(args.J)))

