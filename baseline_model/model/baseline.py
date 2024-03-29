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
        
        # Second-stage
        self.v = self.model.addVars(args.W, args.P, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='vipt')
        self.f = self.model.addVars(args.W, args.J, args.P, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='fwjpgt')
        self.q = self.model.addVars(args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')
        self.s = self.model.addVars(args.I, args.W, args.P, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='siwpt')
        self.r = self.model.addVars(args.W, args.P, args.K, lb=0.0, name='rwp')

        # pdb.set_trace()

        # Objective
        self.model.setObjective((1/(args.K))*quicksum((quicksum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t,k]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
                                +quicksum(self.idata.CU_g[g]*self.q[g,t,k] for g in range(args.G) for t in range(args.T))
                                +quicksum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t,k] for w in range(args.W) for p in range(args.P) for t in range(args.T))
                                +quicksum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p,k] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                -quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))) 
                                for k in range(args.K)), GRB.MINIMIZE);



        # Policy
        for w in range(args.W):
            self.model.addConstr(quicksum(self.idata.u_p[p]*self.x[w,p] for p in range(args.P)) <= self.idata.Cap_w[w])

        # Initail Inventory
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
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
                    self.model.addConstr(self.v[w,p,args.T-1,k] + quicksum(self.s[i,w,p,k] for i in range(args.I))  == self.x[w,p])

        # Recycle Inventory
        for k in range(args.K):
            for p in range(args.P):
                self.model.addConstr(quicksum(self.r[w,p,k] for w in range(args.W)) == self.idata.R_p[p]*(quicksum(self.x[w,p]-self.v[w,p,args.T-1,k] for w in range(args.W))))

        # Demand
        for k in range(args.K):
            for g in range(args.G):
                for t in range(1,args.T):
                    for j in range(args.J):
                        self.model.addConstr(quicksum(self.f[w,j,p,g,t,k] for w in range(args.W) for p in range(args.P)) + self.q[g,t,k] == self.idata.demand[k][j][g][t])


    def run(self,args):

        self.model.update()
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print(self.model.ObjVal)
        else:
            print("Infeasible or unbounded!")

        inventory_level = np.zeros((args.W,args.P))
        for w in range(args.W):
            for p in range(args.P):
                inventory_level[w][p] = self.x[w,p].x

        Used_time = np.zeros((args.T,args.K))

        for t in range(args.T):
            for k in range(args.K):
                Used_time[t][k] = sum(self.v[w,p,t,k].x for w in range(args.W) for p in range(args.P))

        
        operation_cost_total = sum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t,k].x*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for k in range(args.K))/args.K
        holding_cost_total = sum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t,k].x for w in range(args.W) for p in range(args.P) for t in range(args.T) for k in range(args.K))/args.K
        unmet_cost_total = sum(self.idata.CU_g[g]*self.q[g,t,k].x for g in range(args.G) for t in range(args.T) for k in range(args.K))/args.K
        replenish_cost_total = sum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p,k].x for i in range(args.I) for w in range(args.W) for p in range(args.P) for k in range(args.K))/args.K
        group_value_cost = sum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k].x*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A) for k in range(args.K))/args.K

        value_group = np.zeros((args.G))
        for g in range(args.G):
            value_group = sum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(args.T) for a in range(args.A) for k in range(args.K))

        df_name = ["OPT","Operation_Cost","Holding_Cost","Unmet_Cost","Replenish_Cost","Victim_value"]
        data = [[self.model.ObjVal,operation_cost_total,holding_cost_total,unmet_cost_total,replenish_cost_total,group_value_cost]]
        df = pd.DataFrame(data, columns=[df_name])
        df.to_csv("Cost_structure.csv")

        df = pd.DataFrame(inventory_level)
        df.to_csv("Inventory_Policy.csv")

        df = pd.DataFrame(Used_time)
        df.to_csv("Used_time.csv")
     

