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
        self.r = self.model.addVars(args.W, args.P, args.K, vtype=GRB.BINARY, name='rwp')

        # pdb.set_trace()

        # Objective
        self.model.setObjective((1/(args.K))*quicksum((quicksum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t,k]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
                                +quicksum(self.idata.CU_g[g]*self.q[g,t,k] for g in range(args.G) for t in range(args.T))
                                +quicksum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t,k] for w in range(args.W) for p in range(args.P) for t in range(args.T))
                                +quicksum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p,k] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                -quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))) 
                                for k in range(args.K)), GRB.MINIMIZE);



        # Policy
        # for w in range(args.W):
        #     self.model.addConstr(quicksum(self.idata.u_p[p]*x[w,p] for p in range(args.P)) <= )


    def run(self,args,xx=None,evaluation=False,name="TSCC.csv"):

        self.model.update()
        self.model.optimize()
        print(self.model.ObjVal)
     

