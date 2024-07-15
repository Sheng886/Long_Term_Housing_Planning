from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import pdb
import os 

class baseline_class():
    def __init__(self, args, input_data, xx = None):



        self.args = args
        self.idata = input_data

        self.model = gp.Model("basline")

        for k in range(args.K):
            for j in range(args.J):
                for g in range(args.G):
                    for t in range(args.T):
                        self.idata.demand[k][j][g][t] = 50

        # for k in range(args.K):
        #     for j in range(args.J):
        #         for g in range(args.G):
        #                 self.idata.demand[k][j][g][1] = 250
        #                 self.idata.demand[k][j][g][2] = 0
        #                 self.idata.demand[k][j][g][3] = 0
        #                 self.idata.demand[k][j][g][4] = 0
        #                 self.idata.demand[k][j][g][5] = 0



        # First-stage
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')

        
        # Second-stage
        self.v = self.model.addVars(args.W, args.P, args.T+1, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='vipt')
        self.f = self.model.addVars(args.W, args.J, args.P, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='fwjpgt')
        self.q = self.model.addVars(args.J, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')
        self.s = self.model.addVars(args.I, args.W, args.P, args.T , args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='siwpt')
        self.r = self.model.addVars(args.W, args.P, args.K, lb=0.0, name='rwp')

        # pdb.set_trace()

        # Objective
        self.model.setObjective(quicksum(self.idata.O_p[p]*self.x[w,p] for p in range(args.P) for w in range(args.W)) + (1/(args.K))*quicksum((quicksum((self.idata.wj_dis[w][j])*self.f[w,j,p,g,t,k]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
                                +quicksum(100*self.idata.CU_g[g]*self.q[j,g,t,k] for g in range(args.G) for t in range(args.T) for j in range(args.J))
                                +quicksum((self.idata.O_p[p])*self.s[i,w,p,t,k] for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T))
                                +quicksum((self.idata.O_p[p])*self.r[w,p,k] for w in range(args.W) for p in range(args.P))) 
                                for k in range(args.K)), GRB.MINIMIZE);

        
    

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

        # Lead time flow
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    for t in range(args.T):
                        if(t == 0):
                            self.model.addConstr(self.v[w,p,t,k] == quicksum(self.f[w,j,p,g,t,k] for g in range(args.G) for j in range(args.J)) + self.v[w,p,t+1,k])
                        else:
                            self.model.addConstr(self.v[w,p,t,k] + quicksum(self.s[i,w,p,t-1,k] for i in range(args.I)) == quicksum(self.f[w,j,p,g,t,k] for g in range(args.G) for j in range(args.J)) + self.v[w,p,t+1,k])

        for k in range(args.K):
            for i in range(args.I):
                for t in range(args.T):
                    self.model.addConstr(quicksum(self.s[i,w,p,t,k] for p in range(args.P) for w in range(args.W)) <= 50)
        

        # Demand
        for k in range(args.K):
            for g in range(args.G):
                for t in range(1,args.T):
                    for j in range(args.J):
                        self.model.addConstr(quicksum(self.f[w,j,p,g,t,k] for w in range(args.W) for p in range(args.P)) + self.q[j,g,t,k] == self.idata.demand[k][j][g][t])

        
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    self.model.addConstr(self.v[w,p,args.T,k] + self.r[w,p,k] == self.v[w,p,0,k])


    def run(self,args):


        self.model.update()
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print(self.model.ObjVal)
        else:
            print("Infeasible or unbounded!")


        print("total cost", self.model.ObjVal)
        print("inventory1,warehouse1", self.x[0,0].x)
        print("inventory1,warehouse2", self.x[0,1].x)
        print("inventory2,warehouse1", self.x[1,0].x)
        print("inventory2,warehouse2", self.x[1,1].x)
        print("shortage cost", sum(self.q[j,g,t,k].x for g in range(args.G) for t in range(args.T) for j in range(args.J) for k in range(args.K)))
        print("shortage cost", sum(self.q[j,g,t,k].x for g in range(args.G) for t in range(args.T) for j in range(args.J) for k in range(args.K)))

        # print("trans", sum((self.idata.wj_dis[w][j])*self.f[w,j,p,g,t,k].x*args.t_cost for k in range(args.K) for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T)))
        # print("shortage", args.s_factor*self.idata.CU_g[g]*self.q[j,g,t,k].x for k in range(args.K) for g in range(args.G) for t in range(args.T) for j in range(args.J))
        # print(sum((self.idata.wj_dis[w][j])*self.f[w,j,p,g,t,k].x*args.t_cost for k in range(args.K) for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T)))
        # print(sum((self.idata.wj_dis[w][j])*self.f[w,j,p,g,t,k].x*args.t_cost for k in range(args.K) for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T)))


    # self.model.setObjective((1/(args.K))*quicksum((quicksum((self.idata.wj_dis[w][j])*self.f[w,j,p,g,t,k]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
    #                                 +quicksum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t,k] for g in range(args.G) for t in range(args.T) for j in range(args.J))
    #                                 +quicksum((self.idata.O_p[p])*self.s[i,w,p,t,k] for i in range(args.I) for w in range(args.W) for p in range(args.P) for t in range(args.T))
    #                                 +quicksum((self.idata.O_p[p])*self.r[w,p,k] for w in range(args.W) for p in range(args.P))) 
    #                                 for k in range(args.K))

