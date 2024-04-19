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

        self.avg_demand = np.zeros((args.J,args.G,args.T))

        for j in range(args.J):
            for g in range(args.G):
                for t in range(args.T):
                    for k in range(args.K):
                        self.avg_demand[j][g][t] = self.avg_demand[j][g][t] + self.idata.demand[k][j][g][t]

        self.avg_demand = self.avg_demand/args.K


        # First-stage
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        # Second-stage
        self.v = self.model.addVars(args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='vipt')
        self.f = self.model.addVars(args.W, args.J, args.P, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='fwjpgt')
        self.q = self.model.addVars(args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')
        self.s = self.model.addVars(args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='siwpt')
        self.r = self.model.addVars(args.W, args.P, lb=0.0, name='rwp')

        self.f_p1 = self.model.addVars(args.J, args.P1, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='fjpgt')

        # pdb.set_trace()

        # Objective
        self.model.setObjective((quicksum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
                                +quicksum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t] for g in range(args.G) for t in range(args.T) for j in range(args.J))
                                +quicksum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t] for w in range(args.W) for p in range(args.P) for t in range(args.T))
                                +quicksum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                +args.dprate*quicksum(self.idata.O_p[p]*self.x[w,p] for w in range(args.W) for p in range(args.P))
                                -quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))) 
                                -quicksum(self.idata.A_H_flood_p1[a][p]*self.idata.Hd_weight[a][g]*self.f_p1[j,p,g,t]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P1) for g in range(args.G) for t in range(args.T) for a in range(args.A)) 
                                , GRB.MINIMIZE);



        # Policy
        for w in range(args.W):
            self.model.addConstr(quicksum(self.idata.u_p[p]*self.x[w,p] for p in range(args.P)) <= self.idata.Cap_w[w])
           

        # Initail Inventory
        for w in range(args.W):
            for p in range(args.P):
                self.model.addConstr(self.v[w,p,0] == self.x[w,p])

        # Initail flow
        for w in range(args.W):
            for j in range(args.J):
                for p in range(args.P):
                    for g in range(args.G):
                        self.model.addConstr(self.f[w,j,p,g,0] == 0)

        # Flow
        for w in range(args.W):
            for p in range(args.P):
                for t in range(args.T-1):
                    self.model.addConstr(self.v[w,p,t+1] == self.v[w,p,t] - quicksum(self.f[w,j,p,g,t+1] for g in range(args.G) for j in range(args.J)))

        # Refill the Inventory
        for w in range(args.W):
            for p in range(args.P):
                self.model.addConstr(self.v[w,p,args.T-1] + quicksum(self.s[i,w,p] for i in range(args.I))  == self.x[w,p])

        # Recycle Inventory
        for p in range(args.P):
            self.model.addConstr(quicksum(self.r[w,p] for w in range(args.W)) == self.idata.R_p[p]*(quicksum(self.x[w,p]-self.v[w,p,args.T-1] for w in range(args.W))))


        # Demand
        for k in range(args.K):
            for g in range(args.G):
                for t in range(1,args.T):
                    for j in range(args.J):
                        self.model.addConstr(quicksum(self.f[w,j,p,g,t] for w in range(args.W) for p in range(args.P)) + quicksum(self.f_p1[j,p,g,t] for p in range(args.P1))  + self.q[j,g,t] == self.idata.demand[k][j][g][t])


        for k in range(args.K):
            for p in range(args.P1):
                for t in range(1,args.T):
                    self.model.addConstr(quicksum(self.f_p1[j,p,g,t] for j in range(args.J) for p in range(args.P1)) <= self.idata.S_p1[p])


    def run(self,args):

        self.model.update()
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print(self.model.ObjVal)
            return self.x
        else:
            print("Infeasible or unbounded!")


     

