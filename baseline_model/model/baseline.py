from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

class baseline_class():
    def __init__(self, args, input_data):

        self.args = args
        self.ID = input_data

        self.model = gp.Model("basline")

        # First-stage
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        
        # Second-stage
        self.v = self.model.addVars(args.W, args.P, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='vipt')
        self.f = self.model.addVars(args.W, args.J, args.P, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='fwjpgt')
        self.q = self.model.addVars(args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')
        self.s = self.model.addVars(args.I, args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='siwpt')
        self.r = self.model.addVars(args.W, args.P, args.K, vtype=GRB.BINARY, name='rwp')



        # Objective
        self.model.setObjective(sum((self.supplier_price[i][p] + self.supplier_area_distance[i][w]*self.house_volumn_factor[p]*self.trans_price)*self.x[i,w,p,t] for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T))
                              + (1/args.K)*(sum(self.area_region_distance[w][j]*self.house_volumn_factor[p]*self.trans_price*self.fk_wj[k,w,j,p,t] for k in range(args.K) for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(self.first_end_time_point,args.T))
                                + sum(self.unmet[g]*self.house_price[g+1]*(self.qk[k,j,g,args.T-1]) for k in range(args.K) for j in range(args.J) for g in range(args.G))
                                + sum((self.mismatch[p-1][g])*self.house_price[p]*self.yk[k,j,p,g,t] for k in range(args.K) for j in range(args.J) for p in range(1,args.P) for g in range(args.G) for t in range(self.first_end_time_point,args.T))
                                + sum(self.unused[p]*self.house_price[p]*self.mk[k,w,p,args.T-1] for k in range(args.K) for w in range(args.W) for p in range(args.P))
                                + sum((self.deprivation_a0[0] + self.deprivation_a1[0]*(args.T-t))*self.house_price[g+1]*(self.qk[k,j,g,t]) for k in range(args.K) for j in range(args.J) for t in range(self.deprivation_time_point,args.T) for g in range(args.G))
                                + sum((args.emergency_price_factor*self.house_price[p] + self.supplier_region_distance[i][j]*self.house_volumn_factor[p]*self.trans_price)*self.fk_ij[k,i,j,p,t] for k in range(args.K) for i in range(args.I) for j in range(args.J) for t in range(self.first_end_time_point,args.T) for p in range(args.P))
                                 ),GRB.MINIMIZE);

        # self.model.setObjective(1 ,GRB.MINIMIZE);


        # no second-stage decision will be made before T^H
        for k in range(args.K):
            for i in range(args.I):
                for p in range(args.P):
                    for t in range(0,self.first_end_time_point):
                        self.model.addConstr(self.vk[k,i,p,t] == 0)
                        self.model.addConstr(self.nuk[k,i,p,t] == 0)

        for k in range(args.K):
            for w in range(args.W):
                for j in range(args.J):
                    for p in range(args.P):
                        for t in range(0,self.first_end_time_point):
                            self.model.addConstr(self.fk_wj[k,w,j,p,t] == 0)

        for k in range(args.K):
            for j in range(args.J):
                for p in range(args.P):
                    for t in range(0,self.first_end_time_point):
                        self.model.addConstr(self.fk0_j[k,j,p,t] == 0)

        for i in range(args.I):
            for k in range(args.K):
                for j in range(args.J):
                    for p in range(args.P):
                        for t in range(0,self.first_end_time_point):
                            self.model.addConstr(self.fk_ij[k,i,j,p,t] == 0)

        for k in range(args.K):
            for j in range(args.J):
                for p in range(args.P):
                    for g in range(args.G):
                        self.model.addConstr(self.yk[k,j,p,g,args.T-1] == 0)
                        for t in range(0,self.first_end_time_point):
                            self.model.addConstr(self.yk[k,j,p,g,t] == 0)

        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    for t in range(0,self.first_end_time_point):
                        self.model.addConstr(self.mk[k,w,p,t] == 0)

        for k in range(args.K):
            for i in range(args.I):
                for t in range(0,self.first_end_time_point):
                    self.model.addConstr(self.bk[k,i,t] == 0)

        for k in range(args.K):
            for j in range(args.J):
                for g in range(args.G):
                    for t in range(0,self.first_end_time_point):
                        self.model.addConstr(self.dk[k,j,g,t] == 0)
                        self.model.addConstr(self.qk[k,j,g,t] == 0)

        for w in range(args.W):
            for p in range(args.P):
                self.model.addConstr(self.m[w,p,0] == 0)

        # Recovery plan constraint
        self.model.addConstr(sum(self.z[k] for k in range(args.K)) <= args.error)

            
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

        # Second-stage Supplier initial Inventory
        for k in range(args.K):
            for i in range(args.I):
                for p in range(args.P):
                    self.model.addConstr(self.vk[k,i,p,self.first_end_time_point] == self.v[i,p,self.first_end_time_point])

        # Second-stage Supplier Inventory flow
        for k in range(args.K):
            for i in range(args.I):
                for p in range(args.P):
                    for t in range(self.first_end_time_point,args.T-1):
                        self.model.addConstr(self.vk[k,i,p,t+1] + sum(self.x[i,w,p,t] for w in range(args.W)) + sum(self.fk_ij[k,i,j,p,t] for j in range(args.J)) == self.vk[k,i,p,t] + self.nuk[k,i,p,t-self.low_supplier_prod_time[i][p]] + self.s[i,p,t-self.mid_supplier_prod_time[i][p]])
                        
                    
        # Supply sending flow
        for t in range(args.T):
            for i in range(args.I):
                self.model.addConstr(sum(self.house_volumn_factor[p]*self.x[i,w,p,t] for w in range(args.W) for p in range(args.P)) <= self.mid_supplier_flow[i])

        # First-stage Staging area inventory flow limitation
        for w in range(args.W): 
            for t in range(0,self.first_end_time_point):
                for p in range(args.P):
                    self.model.addConstr(self.m[w,p,t+1] == sum(self.x[i,w,p,t] for i in range(args.I)) + self.m[w,p,t])

        # Second-stage staging area initail inventory
        for w in range(args.W):
            for k in range(args.K):
                for p in range(args.P):
                    self.model.addConstr(self.mk[k,w,p,self.first_end_time_point] == self.m[w,p,self.first_end_time_point])

        #Second-stage stating area inventory flow
        for w in range(args.W):
            for p in range(args.P):
                for t in range(self.first_end_time_point,args.T-1):
                    for k in range(args.K):
                        self.model.addConstr(self.mk[k,w,p,t+1] + sum(self.fk_wj[k,w,j,p,t] for j in range(args.J)) == sum(self.x[i,w,p,t] for i in range(args.I)) + self.mk[k,w,p,t])

        # Statgin area Capacity
        for w in range(args.W):
            for t in range(0,self.first_end_time_point):
                self.model.addConstr(sum(self.house_volumn_factor[p]*self.m[w,p,t] for p in range(args.P)) <= self.staging_area_capacity[w])

        # Statgin area Capacity (second-stage)
        for w in range(args.W):
            for t in range(self.first_end_time_point,args.T):
                for k in range(args.K):
                    self.model.addConstr(sum(self.house_volumn_factor[p]*self.mk[k,w,p,t] for p in range(args.P)) <= self.staging_area_capacity[w])

        # second-stage staging area delivery flow
        for w in range(args.W):
            for t in range(self.first_end_time_point,args.T):
                for k in range(args.K):
                    self.model.addConstr(sum(self.house_volumn_factor[p]*self.fk_wj[k,w,j,p,t] for p in range(args.P) for j in range(args.J)) <= self.mid_staging_area_flow[w] + self.z[k]*sum(self.house_volumn_factor[g]*self.demand[k][g][j] for j in range(args.J) for g in range(args.G))) 


        # emergency production
        for i in range(args.I):
            for t in range(self.first_end_time_point,args.T-1):
                for k in range(args.K):
                    self.model.addConstr(self.bk[k,i,t] + sum(self.nuk[k,i,p,t-self.low_supplier_prod_time[i][p]] for p in range(args.P)) == sum(self.nuk[k,i,p,t] for p in range(args.P)) + self.bk[k,i,t+1]) 

        # extra production line in emergency modality
        for i in range(args.I):
            for t in range(self.first_end_time_point,args.T):
                for k in range(args.K):
                    self.model.addConstr(self.bk[k,i,t] <= self.z[k]*self.extra_production_line[i])

        
        # novel house match
        for j in range(args.J):
            for t in range(args.T):
                for k in range(args.K):
                    self.model.addConstr(sum(self.fk_ij[k,i,j,0,t] for i in range(args.I)) + sum(self.fk_wj[k,w,j,0,t] for w in range(args.W)) == sum(self.fk0_j[k,j,p,t+p*self.house_install_time[0]] for p in range(1,args.P) if t + p*self.house_install_time[0] <= args.T-1)) 

        # match flow
        for j in range(args.J):
            for t in range(self.first_end_time_point,args.T):
                for k in range(args.K):
                    for p in range(1,args.P):
                        self.model.addConstr(1/self.house_0_match[p]*self.fk0_j[k,j,p,t] + sum(self.fk_wj[k,w,j,p,t] for w in range(args.W)) + sum(self.fk_ij[k,i,j,p,t] for i in range(args.I)) == sum(self.yk[k,j,p,g,t] for g in range(args.G))) 

        # demand flow
        for j in range(args.J):
            for t in range(args.T-1):
                for g in range(args.G):
                    for k in range(args.K):
                        self.model.addConstr(self.dk[k,j,g,t+1] == self.dk[k,j,g,t] + sum(self.yk[k,j,p,g,t] for p in range(1,args.P)))

        # demand limitation
        for j in range(args.J):
                for k in range(args.K):
                    for g in range(args.G):
                        for t in range(self.first_end_time_point,args.T):
                            self.model.addConstr(self.dk[k,j,g,t] + self.qk[k,j,g,t] == self.demand[k][g][j])

        # big M
        for j in range(args.J):
            for i in range(args.I):
                for p in range(args.P):
                    for k in range(args.K):
                        for t in range(args.T):
                            self.model.addConstr(self.fk_ij[k,i,j,p,t] <= self.z[k]*sum(self.house_0_match[g+1]*self.demand[k][g][j] for g in range(args.G)))


        self.model.setParam("OutputFlag", 0)

    def run(self,args,xx=None,evaluation=False,name="TSCC.csv"):

        self.model.update()
        self.model.optimize()
        print(self.model.ObjVal)
     

