from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import pdb

class StageProblem:
    """A data structure that keeps stage-wise problems"""
    
    def __init__(self, args, input_data, state, stage, last_stage=False, stage0=False):

        self.args = args
        self.idata = input_data
        self.state = state
        self.stage = stage
        
        self.model = gp.Model(f"Stage_{t}_model")
        # Stage variable
        self.u = self.model.addVars(args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='uw')
        self.y = self.model.addVars(args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='yw')
        self.v = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vwp')
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        self.z = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='zwp')

        
        # Scen_Path variable
        self.vk = self.model.addVars(args.K, args.T+1, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vktwp')
        self.bk = self.model.addVars(args.K, args.T+1, args.I, lb=0.0, vtype=GRB.CONTINUOUS, name='bkti')
        self.ak = self.model.addVars(args.K, args.T+1, args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aktiwp')
        self.fk = self.model.addVars(args.K, args.T+1, args.W, args.J, args.P, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='fktwjpg')
        self.sk = self.model.addVars(args.K, args.T+1, args.J, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='sktjg')
        self.aak = self.model.addVars(args.K, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aaktwp')
        self.bbk = self.model.addVars(args.K, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='bbktwp')
        if(last_stage == False):
            self.theta = self.model.addVars(args.N, lb=0.0, vtype=GRB.CONTINUOUS, name='theta')




        # Objective
        if(last_stage == False):
            self.model.setObjective(quicksum(self.idata.E_w[w]*self.y[n] for w in range(args.W)) 
                                  + quicksum(self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) for w in range(args.W) for p in range(args.P))
                                  + (1/args.K)*quicksum(quicksum(self.idata.O_p[p]*self.aak[k,w,p] - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[k,w,p] for w in range(args.W) for p in range(args.P)) 
                                                      + quicksum( quicksum(self.idata.O_p[p]*self.ak[k,t,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                                                + quicksum(self.idata.CU_g[g]*self.sk[n,k,t,j,g] for j in range(args.J) for g in range(args.G)) for t in range(args.T+1)) for k in range(args.K))
                                  + quicksum(self.idata.MC_tran_matrix[state][n]*self.theta[n] for n in range(args.N)) 
                                 , GRB.MINIMIZE);
        else:
            self.model.setObjective(quicksum(self.idata.E_w[w]*self.y[n] for w in range(args.W)) 
                      + quicksum(self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) for w in range(args.W) for p in range(args.P))
                      + (1/args.K)*quicksum(quicksum(self.idata.O_p[p]*self.aak[k,w,p] - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[k,w,p] for w in range(args.W) for p in range(args.P)) 
                                          + quicksum( quicksum(self.idata.O_p[p]*self.ak[k,t,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                                    + quicksum(self.idata.CU_g[g]*self.sk[n,k,t,j,g] for j in range(args.J) for g in range(args.G)) for t in range(args.T+1)) for k in range(args.K))
                     , GRB.MINIMIZE);


        # Staging Area Capacity
        # Dual
        # Receive self.u[parent_node,w] 
        self.b_staging_cap = [0  for W in range(args.W)]
        for w in range(args.W):
            if stage0 == True:
                self.model.addConstr(self.u[w] == self.y[w])
            else:
                self.b_staging_cap[w] = self.model.addConstr(self.u[w] - self.y[w] == 0)

            # Staging Area Capacity >= Invenotry Level
            self.model.addConstr(quicksum(self.v[w,p] for p in range(args.P)) <= self.u[w])


        # Invenory Level
        # Dual
        # Receive self.v[parent_node,w,p]
        self.c_inv_level = [[0  for p in range(args.P)] for W in range(args.W)]
        for w in range(args.W):
            for p in range(args.P):
                if stage0 == True:
                    self.model.addConstr(self.v[w,p] == self.x[w,p] - self.z[w,p])
                else:
                    self.c_inv_level[w][p] = self.model.addConstr(self.v[w,p] - self.x[n,w,p] + self.z[n,w,p] ==  0)


        # Initial Invenory Level in Short-term
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    self.model.addConstr(self.vk[k,0,w,p] == self.v[w,p])



        # Initial Production Capacity Occupied
        for k in range(args.K):
            for i in range(args.I):
                self.model.addConstr(self.bk[k,0,i] == quicksum(self.ak[k,0,i,w,p] for p in range(args.P) for w in range(args.W)))



        # Production Leadtime (assume 1 month lead time)
        for k in range(args.K):
            for t in range(1,args.T+1):
                for i in range(args.I):
                    self.model.addConstr(self.bk[k,t-1,i] + quicksum(self.ak[k,t,i,w,p] for p in range(args.P) for w in range(args.W)) ==  self.bk[k,t,i] + quicksum(self.ak[k,t-self.idata.P_p[p],i,w,p] for p in range(args.P) for w in range(args.W) if t-self.idata.P_p[p] > 0))

        # Production Capacity E_i
        # Dual
        self.h_pro_cap = [[[0  for i in range(args.I)] for t in range(args.T+1)] for k in range(args.K)]
        for k in range(args.K):
            for t in range(args.T+1):
                for i in range(args.I):
                     self.h_pro_cap[k][t][w] = self.model.addConstr(self.bk[k,t,i] <= self.idata.B_i[i])


        # Staging Area Constraints
        for k in range(args.K):
            for t in range(args.T+1):
                for w in range(args.W):
                    self.model.addConstr(quicksum(self.vk[k,t,w,p] for p in range(args.P)) <= self.u[w])


        # Delviery Flow
        for n in range(args.n):
            for k in range(args.K):
                for t in range(1,args.T+1):
                    for w in range(args.W):
                        for p in range(args.P):
                            if(t -self.idata.P_p[p] > 0):
                                self.model.addConstr(self.vk[k,t-1,w,p] + quicksum(self.ak[k,t-self.idata.P_p[p],i,w,p] for i in range(args.I)) == self.vk[k,t,w,p] + quicksum(self.fk[k,t,w,j,p,g] for j in range(args.J) for g in range(args.G)))
                            else:
                                self.model.addConstr(self.vk[k,t-1,w,p]  == self.vk[k,t,w,p] + quicksum(self.fk[k,t,w,j,p,g] for j in range(args.J) for g in range(args.G)))


        
        # Satify Demand Flow
        # Dual
        # Receive self.tree[n].children_blackpath[k].demand[g][t-1]*self.idata.J_pro[j]
        self.k_demand = [[[[0  for g in range(args.G)] for j in range(args.J)] for t in range(args.T+1)] for k in range(args.K)]
        for k in range(args.K):
            for t in range(1,args.T+1):
                for j in range(args.J):
                    for g in range(args.G):
                        self.k_demand[k][t][j][g] = self.model.addConstr(quicksum(self.fk[k,t,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[k,t,j,g] == 0)


        # Assumption Replensih by MHS
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    self.model.addConstr(self.vk[k,args.T,w,p] + self.aak[k,w,p] + self.bbk[k,w,p] == self.v[w,p])


    def run(self,args,n,u,v,demand):

        # Input first-stage solution

        
        if stage0 != True:
            # Staging Capacity
            for w in range(args.W):
                self.b_staging_cap[w].setAttr(GRB.Attr.RHS, u[w].x)

            # Invenory Level
            for w in range(args.W):
                for p in range(args.P):
                    self.c_inv_level[w][p].setAttr(GRB.Attr.RHS, v[w,p].x)

        # Satify Demand Flow
        for k in range(args.K):
            for t in range(1,args.T+1):
                for j in range(args.J):
                    for g in range(args.G):
                        self.k_demand[k][t][j][g].setAttr(GRB.Attr.RHS, demand[k][t][j][g])


        self.model.reset()
        self.model.setParam("OutputFlag", 0)
        self.model.optimize()


        return self.u,self.v


        pi_b = np.zeros((args.W))
        pi_c = np.zeros((args.W, args.P))
        pi_h = np.zeros((args.K, args.T+1, args.I))
        pi_k = np.zeros((args.K, args.T+1,args.J,args.G))

        if stage0 != True:
            # Staging Capacity
            for w in range(args.W):
                pi_b[w] = self.b_staging_cap[w].pi

            # Invenory Level
            for w in range(args.W):
                for p in range(args.P):
                    pi_c[w][p] = self.c_inv_level[w][p].pi

        # Production Capacity E_i
        for k in range(args.K):
            for t in range(args.T+1):
                for i in range(args.I):
                     pi_h[k][t][i] = self.h_pro_cap[k][t][w].pi


        # Satify Demand Flow
        for k in range(args.K):
            for t in range(1,args.T+1):
                for j in range(args.J):
                    for g in range(args.G):
                        pi_k[k][t][j][g] = self.k_demand[k][t][j][g].pi

        return pi_b,pi_c,pi_h,pi_k,self.model.ObjVal
        



class solve_SDDP:
    def __init__(self, args, input_data, m):

        self.args = args
        self.stage_root = StageProblem(args,input_data,0,0,stage0=True)
        self.stage = [[StageProblem(args,input_data,n,t+1,stage0=True) for n in range(args.N)] for in range(args.T-1)] 
        self.stage_leaf = [StageProblem(args,input_data,n,args.T,last_stage=True) for n in range(args.N)];