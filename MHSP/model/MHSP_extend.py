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
        self.tree = input_data.tree.node_all

        self.model = gp.Model("MHSP_extend")

        # Stage variable
        self.u = self.model.addVars(args.TN, args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='unw')
        self.y = self.model.addVars(args.TN, args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='ynw')
        self.v = self.model.addVars(args.TN, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vnwp')
        self.x = self.model.addVars(args.TN, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xnwp')
        self.z = self.model.addVars(args.TN, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='znwp')

        
        # Scen_Path variable
        self.vk = self.model.addVars(args.TN, args.K, args.M+1, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vnktwp')
        self.bk = self.model.addVars(args.TN, args.K, args.M+1, args.I, lb=0.0, vtype=GRB.CONTINUOUS, name='bnkti')
        self.ak = self.model.addVars(args.TN, args.K, args.M+1, args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='anktiwp')
        self.fk = self.model.addVars(args.TN, args.K, args.M+1, args.W, args.J, args.P, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='fnktwjpg')
        self.sk = self.model.addVars(args.TN, args.K, args.M+1, args.J, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='snktjg')
        self.aak = self.model.addVars(args.TN, args.K, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aanktwp')
        self.bbk = self.model.addVars(args.TN, args.K, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='bbnktwp')




        # Objective
        self.model.setObjective(quicksum(self.tree[n].prob_to_node*(quicksum(self.idata.E_w[w]*self.y[n,w] for w in range(args.W)) 
                                                                  + quicksum(self.idata.O_p[p]*(self.x[n,w,p] - self.idata.R_p[p]*self.z[n,w,p]) for w in range(args.W) for p in range(args.P))
                                                                  + (1/(args.K))*quicksum((quicksum(self.idata.O_p[p]*self.aak[n,k,w,p] - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[n,k,w,p] for w in range(args.W) for p in range(args.P)) 
                                                                              + quicksum(quicksum(self.idata.O_p[p]*self.ak[n,k,t,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                                                              + quicksum(self.idata.CU_g[g]*self.sk[n,k,t,j,g] for j in range(args.J) for g in range(args.G)) for t in range(args.M+1))) for k in range(args.K))) 
                                for n in range(args.TN)), GRB.MINIMIZE);



        # Staging Area Capacity
        for n in range(args.TN):
            for w in range(args.W):
                if n == 0:
                    self.model.addConstr(self.u[n,w] == self.y[n,w])
                else:
                    parent_node = self.tree[n].parent
                    self.model.addConstr(self.u[n,w] == self.u[parent_node,w] + self.y[n,w])

                # Staging Area Capacity >= Invenotry Level
                self.model.addConstr(quicksum(self.v[n,w,p] for p in range(args.P)) <= self.u[n,w])


        # Invenory Level
        for n in range(args.TN):
            for w in range(args.W):
                for p in range(args.P):
                    if n == 0:
                        self.model.addConstr(self.v[n,w,p] == self.x[n,w,p] - self.z[n,w,p])
                    else:
                        parent_node =  self.tree[n].parent
                        self.model.addConstr(self.v[n,w,p] == self.v[parent_node,w,p] + self.x[n,w,p] - self.z[n,w,p])


        # Initial Invenory Level in Short-term
        for n in range(args.TN):
            for k in range(args.K):
                for w in range(args.W):
                    for p in range(args.P):
                        self.model.addConstr(self.vk[n,k,0,w,p] == self.v[n,w,p])



        # Initial Production Capacity Occupied
        for n in range(args.TN):
            for k in range(args.K):
                for i in range(args.I):
                    self.model.addConstr(self.bk[n,k,0,i] == quicksum(self.ak[n,k,0,i,w,p] for p in range(args.P) for w in range(args.W)))



        # Production Leadtime (assume 1 month lead time)
        for n in range(args.TN):
            for k in range(args.K):
                for t in range(1,args.M+1):
                    for i in range(args.I):
                        self.model.addConstr(self.bk[n,k,t-1,i] + quicksum(self.ak[n,k,t,i,w,p] for p in range(args.P) for w in range(args.W)) ==  self.bk[n,k,t,i] + quicksum(self.ak[n,k,t-self.idata.P_p[p],i,w,p] for p in range(args.P) for w in range(args.W) if t-self.idata.P_p[p] > 0))

        # Production Capacity E_i
        for n in range(args.TN):
            for k in range(args.K):
                for t in range(args.M+1):
                    for i in range(args.I):
                         self.model.addConstr(self.bk[n,k,t,i] <= self.idata.B_i[i])


        # Staging Area Constraints
        for n in range(args.TN):
            for k in range(args.K):
                for t in range(args.M+1):
                    for w in range(args.W):
                        self.model.addConstr(quicksum(self.vk[n,k,t,w,p] for p in range(args.P)) <= self.u[n,w])


        # Delviery Flow
        for n in range(args.TN):
            for k in range(args.K):
                for t in range(1,args.M+1):
                    for w in range(args.W):
                        for p in range(args.P):
                            if(t -self.idata.P_p[p] > 0):
                                self.model.addConstr(self.vk[n,k,t-1,w,p] + quicksum(self.ak[n,k,t-self.idata.P_p[p],i,w,p] for i in range(args.I)) == self.vk[n,k,t,w,p] + quicksum(self.fk[n,k,t,w,j,p,g] for j in range(args.J) for g in range(args.G)))
                            else:
                                self.model.addConstr(self.vk[n,k,t-1,w,p]  == self.vk[n,k,t,w,p] + quicksum(self.fk[n,k,t,w,j,p,g] for j in range(args.J) for g in range(args.G)))


        
        # Satify Demand Flow
        for n in range(args.TN):
            for k in range(args.K):
                for t in range(1,args.M+1):
                    for j in range(args.J):
                        for g in range(args.G):
                            self.model.addConstr(quicksum(self.fk[n,k,t,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[n,k,t,j,g] == self.tree[n].demand[k][g][t-1]*self.idata.J_pro[j])


        # Assumption Replensih by MHS
        for n in range(args.TN):
            for k in range(args.K):
                for w in range(args.W):
                    for p in range(args.P):
                        self.model.addConstr(self.vk[n,k,args.M,w,p] + self.aak[n,k,w,p] + self.bbk[n,k,w,p] == self.v[n,w,p])

                            

    def run(self,args):

        self.model.update()
        self.model.optimize()


