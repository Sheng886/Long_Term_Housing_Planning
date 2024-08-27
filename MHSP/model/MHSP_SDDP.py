from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import pdb

cut_vio_thred = 1e-5

class StageProblem:
    """A data structure that keeps stage-wise problems"""
    
    def __init__(self, args, input_data, state, stage, last_stage=False, stage0=False):

        self.args = args
        self.idata = input_data
        self.state = state
        self.stage = stage
        self.stage0 = stage0
        
        self.model = gp.Model(f"Stage_{stage}_State_{state}_model")
        # Stage variable
        self.u = self.model.addVars(args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='uw')
        self.y = self.model.addVars(args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='yw')
        self.v = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vwp')
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        self.z = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='zwp')

        
        # Scen_Path variable
        self.vk = self.model.addVars(args.K, args.M+1, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vktwp')
        self.bk = self.model.addVars(args.K, args.M+1, args.I, lb=0.0, vtype=GRB.CONTINUOUS, name='bkti')
        self.ak = self.model.addVars(args.K, args.M+1, args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aktiwp')
        self.fk = self.model.addVars(args.K, args.M+1, args.W, args.J, args.P, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='fktwjpg')
        self.sk = self.model.addVars(args.K, args.M+1, args.J, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='sktjg')
        self.aak = self.model.addVars(args.K, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aaktwp')
        self.bbk = self.model.addVars(args.K, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='bbktwp')
        if(last_stage == False):
            self.theta = self.model.addVars(args.N, lb=0.0, vtype=GRB.CONTINUOUS, name='theta')




        # Objective
        if(last_stage == False):
            self.model.setObjective(quicksum(self.idata.E_w[w]*self.y[w] for w in range(args.W)) 
                                  + quicksum(self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) for w in range(args.W) for p in range(args.P))
                                  + (1/args.K)*quicksum(quicksum(self.idata.O_p[p]*self.aak[k,w,p] - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[k,w,p] for w in range(args.W) for p in range(args.P)) 
                                                      + quicksum( quicksum(self.idata.O_p[p]*self.ak[k,m,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                                                + quicksum(self.idata.CU_g[g]*self.sk[k,m,j,g] for j in range(args.J) for g in range(args.G)) for m in range(args.M+1)) for k in range(args.K))
                                  + quicksum(self.idata.MC_tran_matrix[state][n]*self.theta[n] for n in range(args.N)) 
                                 , GRB.MINIMIZE);
        else:
            self.model.setObjective(quicksum(self.idata.E_w[w]*self.y[w] for w in range(args.W)) 
                      + quicksum(self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) for w in range(args.W) for p in range(args.P))
                      + (1/args.K)*quicksum(quicksum(self.idata.O_p[p]*self.aak[k,w,p] - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[k,w,p] for w in range(args.W) for p in range(args.P)) 
                                          + quicksum( quicksum(self.idata.O_p[p]*self.ak[k,m,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                                    + quicksum(self.idata.CU_g[g]*self.sk[k,m,j,g] for j in range(args.J) for g in range(args.G)) for m in range(args.M+1)) for k in range(args.K))
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
                    self.c_inv_level[w][p] = self.model.addConstr(self.v[w,p] - self.x[w,p] + self.z[w,p] ==  0)


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
            for m in range(1,args.M+1):
                for i in range(args.I):
                    self.model.addConstr(self.bk[k,m-1,i] + quicksum(self.ak[k,m,i,w,p] for p in range(args.P) for w in range(args.W)) ==  self.bk[k,m,i] + quicksum(self.ak[k,m-self.idata.P_p[p],i,w,p] for p in range(args.P) for w in range(args.W) if m-self.idata.P_p[p] > 0))

        # Production Capacity E_i
        # Dual
        self.h_pro_cap = [[[0  for i in range(args.I)] for m in range(args.M+1)] for k in range(args.K)]
        for k in range(args.K):
            for m in range(args.M+1):
                for i in range(args.I):
                     self.h_pro_cap[k][m][i] = self.model.addConstr(self.bk[k,m,i] <= self.idata.B_i[i])


        # Staging Area Constraints
        for k in range(args.K):
            for m in range(args.M+1):
                for w in range(args.W):
                    self.model.addConstr(quicksum(self.vk[k,m,w,p] for p in range(args.P)) <= self.u[w])


        # Delviery Flow
        for k in range(args.K):
            for m in range(1,args.M+1):
                for w in range(args.W):
                    for p in range(args.P):
                        if(m-self.idata.P_p[p] > 0):
                            self.model.addConstr(self.vk[k,m-1,w,p] + quicksum(self.ak[k,m-self.idata.P_p[p],i,w,p] for i in range(args.I)) == self.vk[k,m,w,p] + quicksum(self.fk[k,m,w,j,p,g] for j in range(args.J) for g in range(args.G)))
                        else:
                            self.model.addConstr(self.vk[k,m-1,w,p]  == self.vk[k,m,w,p] + quicksum(self.fk[k,m,w,j,p,g] for j in range(args.J) for g in range(args.G)))


        
        # Satify Demand Flow
        # Dual
        # Receive self.tree[n].children_blackpath[k].demand[g][t-1]*self.idata.J_pro[j]

        self.k_demand = [[[[0  for g in range(args.G)] for j in range(args.J)] for m in range(args.M+1)] for k in range(args.K)]
        for k in range(args.K):
            for m in range(1,args.M+1):
                for j in range(args.J):
                    for g in range(args.G):
                        if(stage0 == False):
                            self.k_demand[k][m][j][g] = self.model.addConstr(quicksum(self.fk[k,m,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[k,m,j,g] == self.idata.demand[stage][state][k][g][m]*self.idata.J_pro[j])
                        else:
                            self.k_demand[k][m][j][g] = self.model.addConstr(quicksum(self.fk[k,m,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[k,m,j,g] == self.idata.demand_root[k][g][m]*self.idata.J_pro[j])


        # Assumption Replensih by MHS
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    self.model.addConstr(self.vk[k,args.M,w,p] + self.aak[k,w,p] + self.bbk[k,w,p] == self.v[w,p])


    def forward_run(self,u=None,v=None):

        # Input first-stage solution

        
        if self.stage0 != True:
            # Staging Capacity
            for w in range(self.args.W):
                self.b_staging_cap[w].setAttr(GRB.Attr.RHS, u[w].x)

            # Invenory Level
            for w in range(self.args.W):
                for p in range(self.args.P):
                    self.c_inv_level[w][p].setAttr(GRB.Attr.RHS, v[w,p].x)

        self.model.update()
        self.model.setParam("OutputFlag", 0)
        self.model.optimize()


        return self.u,self.v

    def backward_run(self):


        pi_b = np.zeros((self.args.W))
        pi_c = np.zeros((self.args.W, self.args.P))
        pi_e = np.zeros((self.args.K, self.args.M+1, self.args.I))
        pi_h = np.zeros((self.args.K, self.args.M+1,self.args.J,self.args.G))

        if self.stage0 != True:
            # Staging Capacity
            for w in range(self.args.W):
                pi_b[w] = self.b_staging_cap[w].pi

            # Invenory Level
            for w in range(self.args.W):
                for p in range(self.args.P):
                    pi_c[w][p] = self.c_inv_level[w][p].pi

        # Production Capacity E_i
        for k in range(self.args.K):
            for m in range(self.args.M+1):
                for i in range(self.args.I):
                     pi_e[k][m][i] = self.h_pro_cap[k][m][i].pi


        # Satify Demand Flow
        for k in range(self.args.K):
            for m in range(1,self.args.M+1):
                for j in range(self.args.J):
                    for g in range(self.args.G):
                        pi_h[k][m][j][g] = self.k_demand[k][m][j][g].pi

        return pi_b,pi_c,pi_e,pi_h,self.model.ObjVal

    def add_cut(self,obj,stage_next,state_sample_path,state_next,pi_b,pi_c,pi_e,pi_h):

        if(self.state == state_sample_path or self.stage0==True):
            if(self.theta[state_next].x < obj - cut_vio_thred and abs(self.theta[state_next].x - obj)/max(abs(self.theta[state_next].x),1e-10) > cut_vio_thred):

                self.model.addConstr(self.theta[state_next] >= quicksum(pi_b[w]*self.u[w] for w in range(self.args.W)) 
                                                            + quicksum(pi_c[w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                            + quicksum(pi_e[k][m][i]*self.idata.B_i[i] for k in range(self.args.K) for m in range(self.args.M+1) for i in range(self.args.I))
                                                            + quicksum(pi_h[k][m][j][g]*self.idata.demand[stage_next][state_next][k][g][m]*self.idata.J_pro[j] for k in range(self.args.K) for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G)))
        else:
            self.model.addConstr(self.theta[state_next] >= quicksum(pi_b[w]*self.u[w] for w in range(self.args.W)) 
                                                        + quicksum(pi_c[w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                        + quicksum(pi_e[k][m][i]*self.idata.B_i[i] for k in range(self.args.K) for m in range(self.args.M+1) for i in range(self.args.I))
                                                        + quicksum(pi_h[k][m][j][g]*self.idata.demand[stage_next][state_next][k][g][m]*self.idata.J_pro[j] for k in range(self.args.K) for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G)))
                      



class solve_SDDP:
    def __init__(self, args, input_data):

        self.args = args
        self.idata = input_data
        self.stage_root = StageProblem(args,input_data,args.initial_state,0,stage0=True)
        self.stage = [[StageProblem(args,input_data,n,t,stage0=False) for n in range(args.N)] for t in range(args.T-1)] 
        self.stage_leaf = [StageProblem(args,input_data,n,args.T-1,last_stage=True) for n in range(args.N)];

    def sample_path(self, args):

        path = []
        
        self.initial_state = args.initial_state
        state = self.initial_state

        for stage in range(args.T):
            next_state = np.random.choice(args.N, 1, self.idata.MC_tran_matrix[state])
            state = next_state[0]
            path.append(state)

        print(path)
        return path

    def run(self):

        old_LB = 0

        for iter in range(self.args.MAX_ITER):

            # sample path
            sample_path = self.sample_path(self.args)

            # ---------------------------------------------------- Forward ----------------------------------------------------
            u,v = self.stage_root.forward_run()
            for stage in range(self.args.T-1):
                u,v = self.stage[stage][sample_path[stage]].forward_run(u,v)
            u,v = self.stage_leaf[sample_path[self.args.T-1]].forward_run(u,v)


            # ----------------------------------- Backward -----------------------------------
            pi_b,pi_c,pi_e,pi_h,LB = self.stage_leaf[sample_path[self.args.T-1]].backward_run()
            for stage in reversed(range(self.args.T-1)):

                # ---------------------------- Cut Sharing ----------------------------
                for state in range(self.args.N):
                    self.stage[stage][state].add_cut(LB,stage+1,sample_path[stage],sample_path[stage+1],pi_b,pi_c,pi_e,pi_h)

                pi_b,pi_c,pi_e,pi_h,LB =  self.stage[stage][sample_path[stage]].backward_run()

            self.stage_root.add_cut(LB,0,0,sample_path[0],pi_b,pi_c,pi_e,pi_h)
            pi_b,pi_c,pi_e,pi_h,LB =  self.stage_root.backward_run()

            print(LB)

            if ((LB - old_LB) / LB <= self.args.LB_TOL and iter>= (self.args.MAX_ITER/2)):
                print(LB)
                break;

            old_LB = LB

            




