from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import pdb

class baseline_class():
    def __init__(self, args, input_data, tree):

        self.args = args
        self.idata = input_data
        self.tree = tree

        self.model = gp.Model("MHSP_extend")

        # Stage variable
        self.u = self.model.addVars(args.n, args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='unw')
        self.y = self.model.addVars(args.n, args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='ynw')
        self.v = self.model.addVars(args.n, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vnwp')
        self.x = self.model.addVars(args.n, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xnwp')
        self.z = self.model.addVars(args.n, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='znwp')

        
        # Scen_Path variable
        self.vk = self.model.addVars(args.n, args.K, args.T, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vnktwp')
        self.bk = self.model.addVars(args.n, args.K, args.T, args.I, lb=0.0, vtype=GRB.CONTINUOUS, name='bnkti')
        self.ak = self.model.addVars(args.n, args.K, args.T, args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='anktiwp')
        self.fk = self.model.addVars(args.n, args.K, args.T, args.W, args.J, args.P, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='fnktwjpg')
        self.sk = self.model.addVars(args.n, args.K, args.T, args.J, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='snktjg')
        self.aak = self.model.addVars(args.n, args.K, args.T, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aanktwp')
        self.bbk = self.model.addVars(args.n, args.K, args.T, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='bbnktwp')



        # pdb.set_trace()

        # Objective
        # self.model.setObjective((1/(args.K))*quicksum((quicksum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t,k]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
        #                         +quicksum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t,k] for g in range(args.G) for t in range(args.T) for j in range(args.J))
        #                         +quicksum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t,k] for w in range(args.W) for p in range(args.P) for t in range(args.T))
        #                         +quicksum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p,k] for i in range(args.I) for w in range(args.W) for p in range(args.P))
        #                         -quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))) 
        #                         for k in range(args.K)), GRB.MINIMIZE);



        # Staging Area Capacity
        for n in range(args.n):
            for w in range(args.W):
                if n == 0:
                    self.model.addConstr(self.u[n,w] == self.y[n,w])
                else:
                    parent_node = tree.node_all[n].parent
                    self.model.addConstr(self.u[n,w] == self.u[parent_node,w] + self.y[n,w])

                # Staging Area Capacity >= Invenotry Level
                self.model.addConstr(quicksum(self.v[n,w,p] for p in range(args.P)) <= self.u[n,w])


        # Invenory Level
        for n in range(args.n):
            for w in range(args.W):
                for p in range(args.P):
                    if n == 0:
                        self.model.addConstr(self.v[n,w,p] == self.x[n,w,p] - self.z[n,w,p])
                    else:
                        parent_node = tree.node_all[n].parent
                        self.model.addConstr(self.v[n,w,p] == self.v[parent_node,w,p] + self.x[n,w,p] - self.z[n,w,p])


        # Initial Invenory Level in Short-term
        for n in range(args.n):
            for k in range(args.K):
                for w in range(args.W):
                    for p in range(args.P):
                        self.model.addConstr(self.vk[n,k,0,w,p] == self.v[n,w,p])


        for n in range(args.n):
            for k in range(args.K):
                for i in range(args.I):
                    self.model.addConstr(self.bk[n,k,0,i] =)



        # Production Leadtime (assume 1 month lead time)
        for n in range(args.n):
            for k in range(args.K):
                for t in range(1,args.T-1):
                    for i in range(args.I):
                        self.model.addConstr(self.bk[n,k,t-1,i] + quicksum(self.ak[n,k,t,i,w,p] for p in range(args.P) for w in range(args.W)) ==  self.bk[n,k,t,i] + quicksum(self.ak[n,k,t+1,i,w,p] for p in range(args.P) for w in range(args.W)))






    def run(self,args):

        self.model.update()
        self.model.optimize()


