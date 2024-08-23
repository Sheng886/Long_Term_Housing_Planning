from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import pdb


cut_vio_thred = 1e-5

class subporblem():
    def __init__(self, args, input_data):

        self.args = args
        self.idata = input_data
        self.tree = input_data.tree.node_all

        self.sub = gp.Model("subproblom");

        # Scen_Path variable
        self.vk = self.sub.addVars(args.M+1, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vnktwp')
        self.bk = self.sub.addVars(args.M+1, args.I, lb=0.0, vtype=GRB.CONTINUOUS, name='bnkti')
        self.ak = self.sub.addVars(args.M+1, args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='anktiwp')
        self.fk = self.sub.addVars(args.M+1, args.W, args.J, args.P, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='fnktwjpg')
        self.sk = self.sub.addVars(args.M+1, args.J, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='snktjg')
        self.aak = self.sub.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aanktwp')
        self.bbk = self.sub.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='bbnktwp')


        # Objective
        self.sub.setObjective(quicksum(self.idata.O_p[p]*self.aak[w,p] - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[w,p] for w in range(args.W) for p in range(args.P)) 
                              + quicksum(quicksum(self.idata.O_p[p]*self.ak[t,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                              + quicksum(self.idata.CU_g[g]*self.sk[t,j,g] for j in range(args.J) for g in range(args.G)) for t in range(args.M+1)), GRB.MINIMIZE);


        # Initial Invenory 
        # receive v_nwp
        # Dual
        self.f_Initial_Invenory_cons = [[0 for p in range(args.P)] for W in range(args.W)]
        for w in range(args.W):
            for p in range(args.P):
                self.f_Initial_Invenory_cons[w][p] = self.sub.addConstr(self.vk[0,w,p] == 0)


        # Initial Production Capacity Occupied
        for i in range(args.I):
            self.sub.addConstr(self.bk[0,i] == quicksum(self.ak[0,i,w,p] for p in range(args.P) for w in range(args.W)))


        # Production Leadtime (assume 1 month lead time)
        for t in range(1,args.M+1):
            for i in range(args.I):
                self.sub.addConstr(self.bk[t-1,i] + quicksum(self.ak[t,i,w,p] for p in range(args.P) for w in range(args.W)) ==  self.bk[t,i] + quicksum(self.ak[t-self.idata.P_p[p],i,w,p] for p in range(args.P) for w in range(args.W) if t-self.idata.P_p[p] > 0))

        # Production Capacity E_i
        # Dual
        self.i_Production_Capacity_cons = [[0 for i in range(args.I)] for t in range(args.M+1)]
        for t in range(args.M+1):
            for i in range(args.I):
                 self.i_Production_Capacity_cons[t][i] = self.sub.addConstr(self.bk[t,i] <= self.idata.B_i[i])

        # Staging Area Constraints
        # receive self.u[n,w]
        # Dual
        self.k_Staging_Capacity_cons = [[0 for w in range(args.W)] for t in range(args.M+1)]
        for t in range(args.M+1):
            for w in range(args.W):
                self.k_Staging_Capacity_cons[t][w] = self.sub.addConstr(quicksum(self.vk[t,w,p] for p in range(args.P)) <= 0)

        # Delviery Flow
        for t in range(1,args.M+1):
            for w in range(args.W):
                for p in range(args.P):
                    if(t -self.idata.P_p[p] > 0):
                        self.sub.addConstr(self.vk[t-1,w,p] + quicksum(self.ak[t-self.idata.P_p[p],i,w,p] for i in range(args.I)) == self.vk[t,w,p] + quicksum(self.fk[t,w,j,p,g] for j in range(args.J) for g in range(args.G)))
                    else:
                        self.sub.addConstr(self.vk[t-1,w,p]  == self.vk[t,w,p] + quicksum(self.fk[t,w,j,p,g] for j in range(args.J) for g in range(args.G)))


        
        # Satify Demand Flow
        # receive self.tree[n].children_blackpath[k].demand[g][t-1]*self.idata.J_pro[j]
        # Dual
        self.l_Demand_Flow_cons = [[[0 for g in range(args.G)] for j in range(args.J)] for t in range(args.M+1)]
        for t in range(1,args.M+1):
            for j in range(args.J):
                for g in range(args.G):
                    self.l_Demand_Flow_cons[t][j][g] = self.sub.addConstr(quicksum(self.fk[t,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[t,j,g] == 0)


        # Assumption Replensih by MHS
        # receive self.v[n,w,p]
        # Dual
        self.m_Replensih_cons = [[0 for p in range(args.P)] for w in range(args.W)]
        for w in range(args.W):
            for p in range(args.P):
                self.m_Replensih_cons[w][p] = self.sub.addConstr(self.vk[args.M,w,p] + self.aak[w,p] + self.bbk[w,p] == 0)

    def run(self,args,n,k,v_vals,u_vals):

        # Input first-stage solution

        # Initial Invenory 
        for w in range(args.W):
            for p in range(args.P):
                self.f_Initial_Invenory_cons[w][p].setAttr(GRB.Attr.RHS, v_vals[n,w,p].x)

        # Staging Area Constraints
        for t in range(args.M+1):
            for w in range(args.W):
                self.k_Staging_Capacity_cons[t][w].setAttr(GRB.Attr.RHS, u_vals[n,w].x)

        # Satify Demand Flow
        for t in range(1,args.M+1):
            for j in range(args.J):
                for g in range(args.G):
                    self.l_Demand_Flow_cons[t][j][g].setAttr(GRB.Attr.RHS, self.tree[n].demand[k][g][t-1]*self.idata.J_pro[j])

        # Assumption Replensih by MHS
        for w in range(args.W):
            for p in range(args.P):
                self.m_Replensih_cons[w][p].setAttr(GRB.Attr.RHS, v_vals[n,w,p].x)

        self.sub.reset()
        self.sub.setParam("OutputFlag", 0)
        self.sub.optimize()


        pi_f = np.zeros((args.W,args.P))
        pi_i = np.zeros((args.M+1, args.I))
        pi_k = np.zeros((args.M+1, args.W))
        pi_l = np.zeros((args.M+1,args.J,args.G))
        pi_m = np.zeros((args.W,args.P))

        temp = 0

        for w in range(args.W):
            for p in range(args.P):
                pi_f[w][p] = self.f_Initial_Invenory_cons[w][p].pi
                temp = temp + pi_f[w][p]*v_vals[n,w,p].x

        for t in range(args.M+1):
            for i in range(args.I):
                 pi_i[t][i] = self.i_Production_Capacity_cons[t][i].pi
                 temp = temp + pi_i[t][i]*self.idata.B_i[i]

        for t in range(args.M+1):
            for w in range(args.W):
                pi_k[t][w] = self.k_Staging_Capacity_cons[t][w].pi
                temp = temp + pi_k[t][w]*u_vals[n,w].x

        for t in range(1,args.M+1):
            for j in range(args.J):
                for g in range(args.G):
                    pi_l[t][j][g] = self.l_Demand_Flow_cons[t][j][g].pi
                    temp = temp + pi_l[t][j][g]*self.tree[n].demand[k][g][t-1]*self.idata.J_pro[j]

        for w in range(args.W):
            for p in range(args.P):
                self.m_Replensih_cons[w][p]
                pi_m[w][p] = self.m_Replensih_cons[w][p].pi
                temp = temp + pi_m[w][p]*v_vals[n,w,p].x

        if( abs(temp-self.sub.ObjVal) >= 1e-5):
            print("problematic dual solution!")
            pdb.set_trace()

        return pi_f,pi_i,pi_k,pi_l,pi_m,self.sub.ObjVal




class Benders():
    def __init__(self, args, input_data, subporblem):

        self.LB = 0
        self.UB = 1000000
        self.eps = 1e-3
        self.max_iterations = 100

        self.args = args
        self.idata = input_data
        self.tree = input_data.tree.node_all
        self.sub = subporblem

        self.master = gp.Model("MHSP_master")

        # Stage variable
        self.u = self.master.addVars(args.TN, args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='unw')
        self.y = self.master.addVars(args.TN, args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='ynw')
        self.v = self.master.addVars(args.TN, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vnwp')
        self.x = self.master.addVars(args.TN, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xnwp')
        self.z = self.master.addVars(args.TN, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='znwp')
        self.theta = self.master.addVars(args.TN, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='thetanwp')


        # Objective
        self.master.setObjective(quicksum(self.tree[n].prob_to_node*(quicksum(self.idata.E_w[w]*self.y[n,w] for w in range(args.W)) 
                                                                  + quicksum(self.idata.O_p[p]*(self.x[n,w,p] - self.idata.R_p[p]*self.z[n,w,p]) for w in range(args.W) for p in range(args.P)) 
                                                                  + (1/args.K)*quicksum(self.theta[n,k] for k in range(args.K))) for n in range(args.TN)), GRB.MINIMIZE);



        # Staging Area Capacity
        for n in range(args.TN):
            for w in range(args.W):
                if n == 0:
                    self.master.addConstr(self.u[n,w] == self.y[n,w])
                else:
                    parent_node = self.tree[n].parent
                    self.master.addConstr(self.u[n,w] == self.u[parent_node,w] + self.y[n,w])

                # Staging Area Capacity >= Invenotry Level
                self.master.addConstr(quicksum(self.v[n,w,p] for p in range(args.P)) <= self.u[n,w])


        # Invenory Level
        for n in range(args.TN):
            for w in range(args.W):
                for p in range(args.P):
                    if n == 0:
                        self.master.addConstr(self.v[n,w,p] == self.x[n,w,p] - self.z[n,w,p])
                    else:
                        parent_node =  self.tree[n].parent
                        self.master.addConstr(self.v[n,w,p] == self.v[parent_node,w,p] + self.x[n,w,p] - self.z[n,w,p])


                            

    def run(self,args):

        self.master.setParam("OutputFlag", 0)
        
        itr = 0

        while(abs((self.UB - self.LB)/self.UB) >= self.eps):

            self.master.optimize()
            self.LB = self.master.ObjVal

            pi_f = np.zeros((args.W,args.P))
            pi_i = np.zeros((args.M+1, args.I))
            pi_k = np.zeros((args.M+1, args.W))
            pi_l = np.zeros((args.M+1,args.J,args.G))
            pi_m = np.zeros((args.W,args.P))
            obj = np.zeros((args.TN,args.K))

            for n in range(args.TN):
                for k in range(args.K):
                    pi_f,pi_i,pi_k,pi_l,pi_m,obj[n][k] = self.sub.run(args,n,k,self.v,self.u)
                    

                    if(self.theta[n,k].x < obj[n][k] - cut_vio_thred and abs(self.theta[n,k].x - obj[n][k])/max(abs(self.theta[n,k].x),1e-10) > cut_vio_thred):

                        self.master.addConstr(self.theta[n,k] >= quicksum(self.v[n,w,p]*pi_f[w][p] for w in range(args.W) for p in range(args.P)) 
                                                                + quicksum(self.idata.B_i[i]*pi_i[t][i] for t in range(args.M+1) for i in range(args.I))
                                                                + quicksum(self.u[n,w]*pi_k[t][w] for t in range(args.M+1) for w in range(args.W))
                                                                + quicksum(self.tree[n].demand[k][g][t-1]*self.idata.J_pro[j]*pi_l[t][j][g] for t in range(1,args.M+1) for j in range(args.J) for g in range(args.G))
                                                                + quicksum(self.v[n,w,p]*pi_m[w][p] for w in range(args.W) for p in range(args.P)))

            UB_temp = sum(self.tree[n].prob_to_node*(sum(self.idata.E_w[w]*self.y[n,w].x for w in range(args.W)) 
                                                   + sum(self.idata.O_p[p]*(self.x[n,w,p].x - self.idata.R_p[p]*self.z[n,w,p].x) for w in range(args.W) for p in range(args.P))
                                                   + (1/args.K)*sum(obj[n][k] for k in range(args.K))) for n in range(args.TN)) 

            self.UB = min(self.UB,UB_temp)

            itr += 1 

        print("Opt",self.UB)



