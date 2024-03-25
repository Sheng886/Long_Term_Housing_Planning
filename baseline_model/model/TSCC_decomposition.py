from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class two_stage_chance():
    def __init__(self, args):

        self.args = args
        # import pdb;pdb.set_trace()

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
        	self.house_install_time[i] = df[self.house_name[i]][3]

        df = pd.read_excel(args.House_Match)

        self.house_0_match = np.zeros(args.P)

        for i in range(1,args.P):
            self.house_0_match[i] = df[self.house_name[i]][0]


        ### ------------------ Deprivation Cost Function ------------------ ###

        df = pd.read_excel(args.Deprivation_Penalty_path)
        self.deprivation_a0 = df["a_0"]
        self.deprivation_a1 = df["a_1"]

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
            self.staging_area_capacity[i] = df[staging_area_column[2]][i]
            self.high_staging_area_flow[i] = df[staging_area_column[3]][i]
            self.mid_staging_area_flow[i] = df[staging_area_column[4]][i]
            self.low_staging_area_flow[i] = df[staging_area_column[5]][i]

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
        
        # np.savetxt("trailer_50.txt",self.demand_per_trailer)
        # np.savetxt("MHU_50.txt",self.demand_per_MHU)

        self.demand_per_trailer = np.loadtxt("test_data/trailer_50.txt")
        self.demand_per_MHU = np.loadtxt("test_data/MHU_50.txt")

        # self.demand_per_trailer = np.loadtxt("sc_50_type1.txt").transpose()
        # self.demand_per_MHU = np.loadtxt("sc_50_type2.txt").transpose()

        # import pdb;pdb.set_trace()

        self. demand = np.stack((self.demand_per_trailer,self.demand_per_MHU),axis=1)


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
        for i in range(0,args.I):
            self.num_production_line[i] = df[supplier_column[24]][i]
            self.extra_production_line[i] = df[supplier_column[25]][i]
            self.high_supplier_flow[i] = df[supplier_column[26]][i]
            self.mid_supplier_flow[i] = df[supplier_column[27]][i]
            self.low_supplier_flow[i] = df[supplier_column[28]][i]
            for p in range(0, args.P):
            	self.supplier_price[i][p] = df[supplier_column[3+p]][i]
            	self.mid_supplier_inventory[i][p] = df[supplier_column[6+p]][i]
            	self.high_supplier_inventory[i][p] = df[supplier_column[12+p]][i]
            	self.low_supplier_inventory[i][p] = df[supplier_column[9+p]][i]
            	self.high_supplier_prod_time[i][p] = df[supplier_column[15+p]][i]
            	self.mid_supplier_prod_time[i][p] = df[supplier_column[18+p]][i]
            	self.low_supplier_prod_time[i][p] = df[supplier_column[21+p]][i]

        ### ------------------ time point ------------------ ###

        df = pd.read_excel(args.Timepoint_path)
        self.first_end_time_point = df["First_End"][0]
        self.deprivation_time_point = df["Deprivation_Start"][0]
        self.second_end_time_point = df["Second_End"][0]

        
        ### ------------------ distance matrix ------------------ ###

        self.supplier_area_distance = func.distance_matrix(supplier_location,staging_area_location)
        self.area_region_distance = func.distance_matrix(staging_area_location,study_region_location)
        self.supplier_region_distance = func.distance_matrix(supplier_location,study_region_location)

        ### ------------------ Model ------------------ ###

        # self.demand[0][0][0] = 100000

        self.model = gp.Model("TSCC_model")

        # First-stage
        self.x = self.model.addVars(args.I, args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Xiwpt')
        self.s = self.model.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Sipt')
        self.m = self.model.addVars(args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Mwpt')
        self.v = self.model.addVars(args.I, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Vipt')
        self.a = self.model.addVars(args.I, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Awpt')
        self.z = self.model.addVars(args.K, vtype=GRB.BINARY, name='Zk')
        self.theta = self.model.addVar(vtype=GRB.CONTINUOUS, name='theta')

        # Objective
        self.model.setObjective(sum((self.supplier_price[i][p] + self.supplier_area_distance[i][w]*self.house_volumn_factor[p]*self.trans_price)*self.x[i,w,p,t] for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T)) + self.theta,GRB.MINIMIZE);

        # self.model.setObjective(1 ,GRB.MINIMIZE);


        for w in range(args.W):
            for p in range(args.P):
                self.model.addConstr(self.m[w,p,0] == 0)

        # Recovery plan constraint
        self.model.addConstr(sum(self.z[k] for k in range(args.K)) == args.error)

        # Initial Machine Capacity && Machine used flow && Every time unit Machine Capacity
        for i in range(args.I):
            self.model.addConstr(self.a[i,0] == 0)
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
                self.model.addConstr(sum(self.house_volumn_factor[p]*self.x[i,w,p,t] for w in range(args.W) for p in range(args.P)) <= self.mid_supplier_flow[i])

        # First-stage Staging area inventory flow limitation
        for w in range(args.W): 
            for t in range(0,self.first_end_time_point):
                for p in range(args.P):
                    self.model.addConstr(self.m[w,p,t+1] == sum(self.x[i,w,p,t] for i in range(args.I)) + self.m[w,p,t])

        # Statgin area Capacity
        for w in range(args.W):
            for t in range(0,self.first_end_time_point):
                self.model.addConstr(sum(self.house_volumn_factor[p]*self.m[w,p,t] for p in range(args.P)) <= self.staging_area_capacity[w])
        
        self.model.setParam("OutputFlag", 0)

    def check_MIPSOL(self,model,where):
        if where == GRB.Callback.MIPSOL:

    		# Second-stage
            print("master solution:",model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
            sub = gp.Model("subprob");
            vk = sub.addVars(self.args.I, self.args.P, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Vk_ipt')
            nuk = sub.addVars(self.args.I, self.args.P, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Nuk_ipt')
            fk_ij = sub.addVars(self.args.I, self.args.J, self.args.P, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_ijpt')
            fk_wj = sub.addVars(self.args.W, self.args.J, self.args.P, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk_wjpt')
            fk0_j = sub.addVars(self.args.J, self.args.P, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Fk0_jpt')
            yk = sub.addVars(self.args.J, self.args.P, self.args.G, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='yk_jpgt')
            mk = sub.addVars(self.args.W, self.args.P, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='Mk_wpt')
            bk = sub.addVars(self.args.I, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='bk_it')
            dk = sub.addVars(self.args.J, self.args.G, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='dk_jgt')
            qk = sub.addVars(self.args.J, self.args.G, self.args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='qk_jgt')

            x_vals = model.cbGetSolution(model._xvars)
            s_vals = model.cbGetSolution(model._svars)
            m_vals = model.cbGetSolution(model._mvars)
            v_vals = model.cbGetSolution(model._vvars)
            z_vals = model.cbGetSolution(model._zvars)
            theta_vals = model.cbGetSolution(model.getVars()[-1])


            for i in range(self.args.I):
                for p in range(self.args.P):
                    for t in range(0,self.first_end_time_point):
                        sub.addConstr(vk[i,p,t] == 0)
                        sub.addConstr(nuk[i,p,t] == 0)

            for w in range(self.args.W):
                for j in range(self.args.J):
                    for p in range(self.args.P):
                        for t in range(0,self.first_end_time_point):
                            sub.addConstr(fk_wj[w,j,p,t] == 0)

            for j in range(self.args.J):
                for p in range(self.args.P):
                    for t in range(0,self.first_end_time_point):
                        sub.addConstr(fk0_j[j,p,t] == 0)

            for i in range(self.args.I):
                for j in range(self.args.J):
                    for p in range(self.args.P):
                        for t in range(0,self.first_end_time_point):
                            sub.addConstr(fk_ij[i,j,p,t] == 0)

            for j in range(self.args.J):
                for p in range(self.args.P):
                    for g in range(self.args.G):
                        sub.addConstr(yk[j,p,g,self.args.T-1] == 0)
                        for t in range(0,self.first_end_time_point):
                            sub.addConstr(yk[j,p,g,t] == 0)

            for w in range(self.args.W):
                for p in range(self.args.P):
                    for t in range(0,self.first_end_time_point):
                        sub.addConstr(mk[w,p,t] == 0)

            for i in range(self.args.I):
                for t in range(0,self.first_end_time_point):
                    sub.addConstr(bk[i,t] == 0)

            for j in range(self.args.J):
                for g in range(self.args.G):
                    for t in range(0,self.first_end_time_point):
                        sub.addConstr(dk[j,g,t] == 0)
                        sub.addConstr(qk[j,g,t] == 0)

            
            S_initial_inventory_cons = []
            # Second-stage Supplier initial Inventory
            for i in range(self.args.I):
                for p in range(self.args.P):
                    temp = sub.addConstr(vk[i,p,self.first_end_time_point] == v_vals[i,p,self.first_end_time_point])
                    S_initial_inventory_cons.append(temp)

            W_initial_inventory_cons = []
            # Second-stage staging area initail inventory
            for w in range(self.args.W):
                for p in range(self.args.P):
                    temp = sub.addConstr(mk[w,p,self.first_end_time_point] == m_vals[w,p,self.first_end_time_point])
                    W_initial_inventory_cons.append(temp)

            
            W_Capacity_cons = []
            # Statgin area Capacity (second-stage)
            for w in range(self.args.W):
                for t in range(self.first_end_time_point,self.args.T):
                    temp = sub.addConstr(sum(self.house_volumn_factor[p]*mk[w,p,t] for p in range(self.args.P)) <= self.staging_area_capacity[w])
                    W_Capacity_cons.append(temp)

            W_inventory_flow_cons = []
            #Second-stage stating area inventory flow
            for w in range(self.args.W):
                    for p in range(self.args.P):
                        for t in range(self.first_end_time_point,self.args.T-1):
                            temp = sub.addConstr(mk[w,p,t+1] + sum(fk_wj[w,j,p,t] for j in range(self.args.J)) - mk[w,p,t] == sum(x_vals[i,w,p,t] for i in range(self.args.I)))
                            W_inventory_flow_cons.append(temp)

            # match flow
            for j in range(self.args.J):
                for t in range(self.first_end_time_point,self.args.T):
                    for p in range(1,self.args.P):
                        sub.addConstr(1/self.house_0_match[p]*fk0_j[j,p,t] + sum(fk_wj[w,j,p,t] for w in range(self.args.W)) + sum(fk_ij[i,j,p,t] for i in range(self.args.I)) == sum(yk[j,p,g,t] for g in range(self.args.G))) 

            # demand flow
            for j in range(self.args.J):
                for t in range(self.args.T-1):
                    for g in range(self.args.G):
                        sub.addConstr(dk[j,g,t+1] == dk[j,g,t] + sum(yk[j,p,g,t] for p in range(1,self.args.P)))

            demand_cons = []
            Emergency_flow_cons = []
            emergency_product_cons = []
            staging_area_flow_cons = []
            novel_house_match_cons = []
            extra_production_cons = []
            supplier_inventory_cons = []

            pi_3 = np.zeros((self.args.K,self.args.W,self.args.T-self.first_end_time_point))
            pi_2b = np.zeros((self.args.K,self.args.W, self.args.P))
            pi_2c = np.zeros((self.args.K,self.args.W,self.args.P,self.args.T-self.first_end_time_point-1))
            pi_2d = np.zeros((self.args.K,self.args.W,self.args.T-self.first_end_time_point))
            pi_4c = np.zeros((self.args.K,self.args.I, self.args.P))
            pi_4d = np.zeros((self.args.K,self.args.I,self.args.P,self.args.T-1-self.first_end_time_point))
            pi_4i = np.zeros((self.args.K,self.args.I,self.args.T-self.first_end_time_point))
            pi_2h = np.zeros((self.args.K,self.args.J,self.args.G,self.args.T-self.first_end_time_point))

            const1 = np.zeros((self.args.K))
            const2 = np.zeros((self.args.K))

            M1 = np.zeros((self.args.K))
            M2 = np.zeros((self.args.K))

            for k in range(self.args.K):
                sub.setObjective(sum(self.area_region_distance[w][j]*self.house_volumn_factor[p]*self.trans_price*fk_wj[w,j,p,t] for w in range(self.args.W) for j in range(self.args.J) for p in range(self.args.P) for t in range(self.first_end_time_point,self.args.T))
                                + sum(self.unmet[g]*self.house_price[g+1]*(qk[j,g,self.args.T-1]) for j in range(self.args.J) for g in range(self.args.G))
                                + sum((self.mismatch[p-1][g])*self.house_price[p]*yk[j,p,g,t] for j in range(self.args.J) for p in range(1,self.args.P) for g in range(self.args.G) for t in range(self.first_end_time_point,self.args.T))
                                + sum(self.unused[p]*self.house_price[p]*mk[w,p,self.args.T-1] for w in range(self.args.W) for p in range(self.args.P))
                                + sum((self.deprivation_a0[0] + self.deprivation_a1[0]*(self.args.T-t))*self.house_price[g+1]*(qk[j,g,t]) for j in range(self.args.J) for t in range(self.deprivation_time_point,self.args.T) for g in range(self.args.G))
                                + sum((self.args.emergency_price_factor*self.house_price[p] + self.supplier_region_distance[i][j]*self.house_volumn_factor[p]*self.trans_price)*fk_ij[i,j,p,t] for i in range(self.args.I) for j in range(self.args.J) for t in range(self.first_end_time_point,self.args.T) for p in range(self.args.P))
                                ,GRB.MINIMIZE);

                # demand limitation
                
                if(k != 0):
                    while(len(demand_cons)>0):
                        sub.remove(demand_cons[0])
                        demand_cons.pop(0)

                for j in range(self.args.J):
                    for g in range(self.args.G):
                        for t in range(self.first_end_time_point,self.args.T):
                            temp = sub.addConstr(dk[j,g,t] + qk[j,g,t] == self.demand[k][g][j])
                            demand_cons.append(temp)

                # import pdb;pdb.set_trace()

                if(k != 0 and z_vals[k-1] == 0):    
                    while(len(Emergency_flow_cons)>0):
                        sub.remove(Emergency_flow_cons[0])
                        Emergency_flow_cons.pop(0)
                    while(len(staging_area_flow_cons)>0):
                        sub.remove(staging_area_flow_cons[0])
                        staging_area_flow_cons.pop(0)
                    while(len(novel_house_match_cons)>0):
                        sub.remove(novel_house_match_cons[0])
                        novel_house_match_cons.pop(0)

                if(k != 0 and z_vals[k-1] == 1): 
                    while(len(emergency_product_cons)>0):
                        sub.remove(emergency_product_cons[0])
                        emergency_product_cons.pop(0)
                    while(len(extra_production_cons)>0):
                        sub.remove(extra_production_cons[0])
                        extra_production_cons.pop(0)
                    while(len(supplier_inventory_cons)>0):
                        sub.remove(supplier_inventory_cons[0])
                        supplier_inventory_cons.pop(0)
                    while(len(novel_house_match_cons)>0):
                        sub.remove(novel_house_match_cons[0])
                        novel_house_match_cons.pop(0)


                if(z_vals[k] == 0):
                    for i in range(self.args.I):
                        for j in range(self.args.J):
                            for p in range(self.args.P):
                                for t in range(self.first_end_time_point,self.args.T):
                                    temp = sub.addConstr(fk_ij[i,j,p,t] == 0)
                                    Emergency_flow_cons.append(temp)


                    # second-stage staging area delivery flow limit
                    for w in range(self.args.W):
                        for t in range(self.first_end_time_point,self.args.T):
                            temp = sub.addConstr(sum(self.house_volumn_factor[p]*fk_wj[w,j,p,t] for p in range(self.args.P) for j in range(self.args.J)) <= self.mid_staging_area_flow[w]) 
                            staging_area_flow_cons.append(temp)

                    # novel house match
                    for j in range(self.args.J):
                        for t in range(self.args.T):
                            temp = sub.addConstr(sum(fk_wj[w,j,0,t] for w in range(self.args.W)) - sum(fk0_j[j,p,t+p*self.house_install_time[0]] for p in range(1,self.args.P) if t + p*self.house_install_time[0] <= self.args.T-1) == 0) 
                            novel_house_match_cons.append(temp)

                else:
                    # emergency production
                    for i in range(self.args.I):
                        for t in range(self.first_end_time_point,self.args.T-1):
                                temp = sub.addConstr(bk[i,t] + sum(nuk[i,p,t-self.low_supplier_prod_time[i][p]] for p in range(self.args.P)) == sum(nuk[i,p,t] for p in range(self.args.P)) + bk[i,t+1]) 
                                emergency_product_cons.append(temp)

                    # extra production line in emergency modality
                    for i in range(self.args.I):
                        for t in range(self.first_end_time_point,self.args.T):
                            temp = sub.addConstr(bk[i,t] <= self.extra_production_line[i])
                            extra_production_cons.append(temp)

                    # Second-stage Supplier Inventory flow
                    for i in range(self.args.I):
                        for p in range(self.args.P):
                            for t in range(self.first_end_time_point,self.args.T-1):
                                temp = sub.addConstr(vk[i,p,t+1] + sum(x_vals[i,w,p,t] for w in range(self.args.W)) + sum(fk_ij[i,j,p,t] for j in range(self.args.J)) - vk[i,p,t] - nuk[i,p,t-self.low_supplier_prod_time[i][p]] == s_vals[i,p,t-self.mid_supplier_prod_time[i][p]])
                                supplier_inventory_cons.append(temp)

                    # novel house match
                    for j in range(self.args.J):
                        for t in range(self.args.T):
                            temp = sub.addConstr(sum(fk_ij[i,j,0,t] for i in range(self.args.I)) + sum(fk_wj[w,j,0,t] for w in range(self.args.W)) == sum(fk0_j[j,p,t+p*self.house_install_time[0]] for p in range(1,self.args.P) if t + p*self.house_install_time[0] <= self.args.T-1)) 
                            novel_house_match_cons.append(temp)


                
                sub.setParam("OutputFlag", 0)
                sub.optimize()


                if sub.status == GRB.OPTIMAL:
                    print("sub",k,":",sub.ObjVal)
                else:
                    print("infesible or unbounded")

                sol_temp = 0

                # second-stage w limit
                for j in range(self.args.J):
                    for g in range(self.args.G):
                        for t in range(self.args.T-self.first_end_time_point):
                            pi_2h[k][j][g][t] = demand_cons[j*self.args.G*(self.args.T-self.first_end_time_point) + g*(self.args.T-self.first_end_time_point) + t].pi
                            sol_temp = sol_temp + pi_2h[k][j][g][t]*self.demand[k][g][j]

                for w in range(self.args.W):
                    for t in range(self.args.T-self.first_end_time_point):
                        pi_3[k][w][t] = W_Capacity_cons[w*(self.args.T-self.first_end_time_point)+t].pi
                        sol_temp = sol_temp + pi_3[k][w][t]*self.staging_area_capacity[w]


                for w in range(self.args.W):
                    for p in range(self.args.P):
                        pi_2b[k][w][p] = W_initial_inventory_cons[w*self.args.P+p].pi
                        sol_temp = sol_temp + pi_2b[k][w][p]*m_vals[w,p,self.first_end_time_point]

                
                for w in range(self.args.W):
                    for p in range(self.args.P):
                        for t in range(self.args.T-self.first_end_time_point-1):
                            pi_2c[k][w][p][t] = W_inventory_flow_cons[w*self.args.P*(self.args.T-self.first_end_time_point-1)+p*(self.args.T-self.first_end_time_point-1)+t].pi
                            sol_temp = sol_temp + pi_2c[k][w][p][t]*sum(x_vals[i,w,p,t+self.first_end_time_point] for i in range(self.args.I))


                for i in range(self.args.I):
                    for p in range(self.args.P):
                        pi_4c[k][i][p] = S_initial_inventory_cons[i*self.args.P+p].pi
                        sol_temp = sol_temp + pi_4c[k][i][p]*v_vals[i,p,self.first_end_time_point]

                
                if(z_vals[k] == 0):
                    
                    for w in range(self.args.W):
                        for t in range(self.args.T-self.first_end_time_point):
                            pi_2d[k][w][t] = staging_area_flow_cons[w*(self.args.T-self.first_end_time_point)+t].pi
                            sol_temp = sol_temp + self.mid_staging_area_flow[w]*pi_2d[k][w][t]


                if(z_vals[k] == 1):
                    
                    
                    for i in range(self.args.I):
                        for p in range(self.args.P):
                            for t in range(self.args.T-1-self.first_end_time_point):
                                pi_4d[k][i][p][t] = supplier_inventory_cons[i*self.args.P*(self.args.T-1-self.first_end_time_point)+p*(self.args.T-1-self.first_end_time_point)+t].pi
                                sol_temp = sol_temp + (s_vals[i,p,t-self.mid_supplier_prod_time[i][p]+self.first_end_time_point] - sum(x_vals[i,w,p,t+self.first_end_time_point] for w in range(self.args.W)))*pi_4d[k][i][p][t]
                    

                    for i in range(self.args.I):
                        for t in range(self.args.T-self.first_end_time_point):
                            pi_4i[k][i][t] = extra_production_cons[i*(self.args.T-self.first_end_time_point)+t].pi
                            sol_temp = sol_temp + self.extra_production_line[i]*pi_4i[k][i][t]

                print("k;",k)
                M1[k] = sum(self.staging_area_capacity[w]*pi_2b[k][w][p] for w in range(self.args.W) for p in range(self.args.P) if pi_2b[k][w][p]>0) 
                print(M1[k])
                M1[k] = M1[k] + sum(pi_2h[k][j][g][t]*self.demand[k][g][j] for j in range(self.args.J) for g in range(self.args.G) for t in range(self.args.T-self.first_end_time_point) if pi_2h[k][j][g][t]>0) 
                print(M1[k])
                M1[k] = M1[k] + sum(sum(self.mid_supplier_flow[i] for i in range(self.args.I))*pi_2c[k][w][p][t] for w in range(self.args.W) for p in range(self.args.P) for t in range(self.args.T-self.first_end_time_point-1) if pi_2c[k][w][p][t]>0) 
                print(M1[k])
                M1[k] = M1[k] + sum((self.mid_supplier_inventory[i][p] + self.num_production_line[i])*pi_4c[k][i][p] for i in range(self.args.I) for p in range(self.args.P) if pi_4c[k][i][p]>0)
                print(M1[k])
                M2[k] = M1[k] + sum(self.num_production_line[i]*pi_4d[k][i][p][t] for t in range(self.args.T-1-self.first_end_time_point) for p in range(self.args.P) for i in range(self.args.I) if pi_4d[k][i][p][t]>0)

                # print("bigM:",M1[k],M2[k])
                # import pdb;pdb.set_trace()
                # M1[k] = 8000000000
                # M2[k] = 8000000000
                                
                # import pdb;pdb.set_trace()
                # print("dual:",sol_temp)
            model.cbLazy(model.getVars()[-1] + sum(M1[k]*model._zvars[k] for k in range(self.args.K) if z_vals[k] == 0) + sum(M2[k]*(1-model._zvars[k]) for k in range(self.args.K) if z_vals[k] == 1) 
                    >= (1/self.args.K)*(sum(model._mvars[w,p,self.first_end_time_point]*pi_2b[k][w][p] for w in range(self.args.W) for p in range(self.args.P) for k in range(self.args.K))
                                       +sum(sum(model._xvars[i,w,p,t+self.first_end_time_point] for i in range(self.args.I))*pi_2c[k][w][p][t] for w in range(self.args.W) for p in range(self.args.P) for t in range(self.args.T-self.first_end_time_point-1) for k in range(self.args.K))
                                       +sum(pi_3[k][w][t]*self.staging_area_capacity[w] for k in range(self.args.K) for w in range(self.args.W) for t in range(self.args.T-self.first_end_time_point))
                                       +sum(pi_2h[k][j][g][t]*self.demand[k][g][j] for j in range(self.args.J) for g in range(self.args.G) for k in range(self.args.K) for t in range(self.args.T-self.first_end_time_point))
                                       +sum(model._vvars[i,p,self.first_end_time_point]*pi_4c[k][i][p] for i in range(self.args.I) for p in range(self.args.P) for k in range(self.args.K))
                                       +sum(self.mid_staging_area_flow[w]*pi_2d[k][w][t] for w in range(self.args.W) for t in range(self.args.T-self.first_end_time_point) for k in range(self.args.K) if z_vals[k] == 0)
                                       +sum((model._svars[i,p,t-self.mid_supplier_prod_time[i][p]+self.first_end_time_point] - sum(model._xvars[i,w,p,t+self.first_end_time_point] for w in range(self.args.W)))*pi_4d[k][i][p][t] for p in range(self.args.P) for i in range(self.args.I) for t in range(self.args.T-1-self.first_end_time_point) for k in range(self.args.K) if z_vals[k] == 1)
                                       +sum(self.extra_production_line[i]*pi_4i[k][i][t] for i in range(self.args.I) for t in range(self.args.T-self.first_end_time_point) for k in range(self.args.K) if z_vals[k] == 1)
                                      ))

            
            print("theta:",theta_vals + sum(M1*z_vals[k] for k in range(self.args.K) if z_vals[k] == 0) + sum(M2*(1-z_vals[k]) for k in range(self.args.K) if z_vals[k] == 1))
            print("lower bound",(1/self.args.K)*(sum(m_vals[w,p,self.first_end_time_point]*pi_2b[k][w][p] for w in range(self.args.W) for p in range(self.args.P) for k in range(self.args.K))
                                       +sum(sum(x_vals[i,w,p,t+self.first_end_time_point] for i in range(self.args.I))*pi_2c[k][w][p][t] for w in range(self.args.W) for p in range(self.args.P) for t in range(self.args.T-self.first_end_time_point-1) for k in range(self.args.K))
                                       +sum(pi_3[k][w][t]*self.staging_area_capacity[w] for k in range(self.args.K) for w in range(self.args.W) for t in range(self.args.T-self.first_end_time_point))
                                       +sum(pi_2h[k][j][g][t]*self.demand[k][g][j] for j in range(self.args.J) for g in range(self.args.G) for k in range(self.args.K) for t in range(self.args.T-self.first_end_time_point))
                                       +sum(v_vals[i,p,self.first_end_time_point]*pi_4c[k][i][p] for i in range(self.args.I) for p in range(self.args.P) for k in range(self.args.K))
                                       +sum(self.mid_staging_area_flow[w]*pi_2d[k][w][t] for w in range(self.args.W) for t in range(self.args.T-self.first_end_time_point) for k in range(self.args.K) if z_vals[k] == 0)
                                       +sum((s_vals[i,p,t-self.mid_supplier_prod_time[i][p]+self.first_end_time_point] - sum(x_vals[i,w,p,t+self.first_end_time_point] for w in range(self.args.W)))*pi_4d[k][i][p][t] for p in range(self.args.P) for i in range(self.args.I) for t in range(self.args.T-1-self.first_end_time_point) if t-self.mid_supplier_prod_time[i][p]>=0 for k in range(self.args.K) if z_vals[k] == 1)
                                       +sum(self.extra_production_line[i]*pi_4i[k][i][t] for i in range(self.args.I) for t in range(self.args.T-self.first_end_time_point) for k in range(self.args.K) if z_vals[k] == 1)
                                       ))

            print(z_vals)
            





    def run(self,args,xx=None,evaluation=False,name="TWCC.csv"):
        if(evaluation == True):
            for i in range(args.I):
                for w in range(args.W):
                    for p in range(args.P):
                        for t in range(args.T):
                            self.model.addConstr(self.x[i,w,p,t] == xx[i][w][p][t])
        
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
        self.model.optimize(lambda model, where: self.check_MIPSOL(model, where))
        # self.model.optimize(self.check_MIPSOL)



        if self.model.status == GRB.OPTIMAL:
            print(self.model.ObjVal)
            # for i in range(args.I):
            #     for w in range(args.W):
            #         for p in range(args.P):
            #             for t in range(args.T):
            #                 print(i,w,p,t,self.x[i,w,p,t].x)
     
        else:
            print("infesible or unbounded")

