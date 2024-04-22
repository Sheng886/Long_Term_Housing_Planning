from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import pdb

class baseline_class():
    def __init__(self, args, input_data):



        self.args = args
        self.idata = input_data

        self.model = gp.Model("basline")

        # First-stage
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        self.diff_x = self.model.addVars(args.P, args.P, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_x = self.model.addVars(args.P, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        self.xk = self.model.addVars(args.W, args.P, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        self.xkk = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        
        # Second-stage
        self.v = self.model.addVars(args.W, args.P, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='vipt')
        self.f = self.model.addVars(args.W, args.J, args.P, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='fwjpgt')
        self.q = self.model.addVars(args.J, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')
        self.s = self.model.addVars(args.I, args.W, args.P, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='siwpt')
        self.r = self.model.addVars(args.W, args.P, args.K, lb=0.0, name='rwp')

        self.f_p1 = self.model.addVars(args.J, args.P1, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='fjpgt')

        self.diff_group = self.model.addVars(args.G, args.G, args.T, args.K, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_group = self.model.addVars(args.G, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        self.diff_region = self.model.addVars(args.J, args.J, args.T, args.K, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_region = self.model.addVars(args.J, args.J, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        self.diff_value = self.model.addVars(args.G, args.G, args.T, args.K, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_value = self.model.addVars(args.G, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        # pdb.set_trace()

        # Objective
        self.model.setObjective((1/(args.K))*quicksum((quicksum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t,k]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
                                +quicksum(self.idata.O_p1[p]*self.f_p1[j,p,g,t,k] for j in range(args.J) for p in range(args.P1) for g in range(args.G) for t in range(args.T))
                                +quicksum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t,k] for g in range(args.G) for t in range(args.T) for j in range(args.J))
                                +quicksum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t,k] for w in range(args.W) for p in range(args.P) for t in range(args.T))
                                +quicksum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p,k] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                +args.dprate*quicksum(self.idata.O_p[p]*self.x[w,p] for w in range(args.W) for p in range(args.P))
                                -quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))) 
                                -quicksum(self.idata.A_H_flood_p1[a][p]*self.idata.Hd_weight[a][g]*self.f_p1[j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P1) for g in range(args.G) for t in range(args.T) for a in range(args.A)) 
                                for k in range(args.K)), GRB.MINIMIZE);

        
        # Policy
        if(args.model == "perfect"):
            for w in range(args.W):
                for k in range(args.K):
                    self.model.addConstr(quicksum(self.idata.u_p[p]*self.xk[w,p,k] for p in range(args.P)) <= self.idata.Cap_w[w])
            
        else:
            for w in range(args.W):
                self.model.addConstr(quicksum(self.idata.u_p[p]*self.x[w,p] for p in range(args.P)) <= self.idata.Cap_w[w])
           

        # Initail Inventory
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    if(args.model == "perfect"):
                        self.model.addConstr(self.v[w,p,0,k] == self.xk[w,p,k])
                    else:
                        self.model.addConstr(self.v[w,p,0,k] == self.x[w,p])
                    

        # Initail flow
        for w in range(args.W):
            for j in range(args.J):
                for p in range(args.P):
                    for g in range(args.G):
                        for k in range(args.K):
                            self.model.addConstr(self.f[w,j,p,g,0,k] == 0)

        # Flow
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    for t in range(args.T-1):
                        self.model.addConstr(self.v[w,p,t+1,k] == self.v[w,p,t,k] - quicksum(self.f[w,j,p,g,t+1,k] for g in range(args.G) for j in range(args.J)))

        # Refill the Inventory
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    if(args.model == "perfect"):
                        if(k+1 == args.K):
                            self.model.addConstr(self.v[w,p,args.T-1,k] + quicksum(self.s[i,w,p,k] for i in range(args.I))  == self.xk[w,p,0])
                        else:
                            self.model.addConstr(self.v[w,p,args.T-1,k] + quicksum(self.s[i,w,p,k] for i in range(args.I))  == self.xk[w,p,k+1])
                    else:
                        self.model.addConstr(self.v[w,p,args.T-1,k] + quicksum(self.s[i,w,p,k] for i in range(args.I))  == self.x[w,p])

        # Recycle Inventory
        for k in range(args.K):
            for p in range(args.P):
                if(args.model == "perfect"):
                    self.model.addConstr(quicksum(self.r[w,p,k] for w in range(args.W)) == self.idata.R_p[p]*(quicksum(self.xk[w,p,k]-self.v[w,p,args.T-1,k] for w in range(args.W))))
                else:
                    self.model.addConstr(quicksum(self.r[w,p,k] for w in range(args.W)) == self.idata.R_p[p]*(quicksum(self.x[w,p]-self.v[w,p,args.T-1,k] for w in range(args.W))))

        
        if(args.model == "avg"):
            self.avg_demand = np.zeros((args.J,args.G,args.T))
            for j in range(args.J):
                for g in range(args.G):
                    for t in range(args.T):
                        for k in range(args.K):
                            self.avg_demand[j][g][t] = self.avg_demand[j][g][t] + self.idata.demand[k][j][g][t]

            self.avg_demand = self.avg_demand/args.K


        # Demand
        for k in range(args.K):
            for g in range(args.G):
                for t in range(1,args.T):
                    for j in range(args.J):
                        self.model.addConstr(quicksum(self.f[w,j,p,g,t,k] for w in range(args.W) for p in range(args.P)) + quicksum(self.f_p1[j,p,g,t,k] for p in range(args.P1)) + self.q[j,g,t,k] == self.avg_demand[j][g][t])

        for k in range(args.K):
            for p in range(args.P1):
                self.model.addConstr(quicksum(self.f_p1[j,p,g,t,k] for j in range(args.J) for g in range(args.G) for t in range(args.T)) <= self.idata.S_p1[p])

    def run(self,args):

        self.model.update()
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print(self.model.ObjVal)
            return self.x
        else:
            print("Infeasible or unbounded!")


     

