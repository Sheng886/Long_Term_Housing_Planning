from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import time
import pdb
import sys
# import matplotlib.pyplot as plt

sub_problem_time = 0
operation_Benders_problem_time = 0
strategic_problem_time = 0

cut_vio_thred = 1e-4

class op_node:

    def __init__(self, args, input_data, state, stage, k, last_stage=False):

        self.args = args
        self.idata = input_data

        self.last_stage = last_stage

        self.stage = stage
        self.state = state
        self.k = k

        self.cut_rhs = []
        self.cut = []

        self.cut_rhs_temp = []
        self.cut_temp = []


        self.sub = gp.Model("sub")

        # Scen_Path variable
        self.vk = self.sub.addVars(args.M+1, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vktwp')
        
        self.v = self.sub.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vwp')

        self.bk = self.sub.addVars(args.M+1, args.I, lb=0.0, vtype=GRB.CONTINUOUS, name='bkti')
        self.ak = self.sub.addVars(args.M+1, args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aktiwp')
        self.fk = self.sub.addVars(args.M+1, args.W, args.J, args.P, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='fktwjpg')
        self.sk = self.sub.addVars(args.M+1, args.J, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='sktjg')
        if(last_stage == False):
            self.phi = self.sub.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='phi')


        if(last_stage == False):
            self.sub.setObjective(quicksum(quicksum(self.idata.O_p[p]*self.ak[m,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                           + quicksum(self.idata.CU_g[g]*self.sk[m,j,g] for j in range(args.J) for g in range(args.G)) for m in range(args.M+1))
                                  + self.phi, GRB.MINIMIZE);
        else:
            self.sub.setObjective(quicksum(quicksum(self.idata.O_p[p]*self.ak[m,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                           + quicksum(self.idata.CU_g[g]*self.sk[m,j,g] for j in range(args.J) for g in range(args.G)) for m in range(args.M+1))
                                  , GRB.MINIMIZE);


        # return
        for w in range(args.W):
            for p in range(args.P):
                self.sub.addConstr(self.vk[args.M,w,p] == self.v[w,p])



        # Initial Inventory Level in Short-term
        # Dual
        # rhs
        self.ini_invent_cons = [[0  for p in range(args.P)] for w in range(args.W)]
        for w in range(args.W):
            for p in range(args.P):
                self.ini_invent_cons[w][p] = self.sub.addConstr(self.vk[0,w,p] == 0)



        # Initial Production Capacity Occupied
        for i in range(args.I):
            self.sub.addConstr(self.bk[0,i] == quicksum(self.ak[0,i,w,p] for p in range(args.P) for w in range(args.W)))



        # Production Leadtime (assume 1 month lead time)
        for m in range(1,args.M+1):
            for i in range(args.I):
                self.sub.addConstr(self.bk[m-1,i] + quicksum(self.ak[m,i,w,p] for p in range(args.P) for w in range(args.W) if m+self.idata.P_p[p] <= args.M) ==  self.bk[m,i] + quicksum(self.ak[m-self.idata.P_p[p],i,w,p] for p in range(args.P) for w in range(args.W) if m-self.idata.P_p[p] >= 0))

        # Production Capacity E_i
        # Dual
        self.prod_cap_cons = [[0 for i in range(args.I)] for m in range(args.M+1)]
        for m in range(args.M+1):
            for i in range(args.I):
                 self.prod_cap_cons[m][i] = self.sub.addConstr(self.bk[m,i] <= self.idata.B_i[i])


        # Staging Area Constraints
        # Dual
        # rhs
        self.stagarea_cons = [[0 for w in range(args.W)] for m in range(args.M+1)]
        for m in range(args.M+1):
            for w in range(args.W):
                self.stagarea_cons[m][w] = self.sub.addConstr(quicksum(self.vk[m,w,p] for p in range(args.P)) <= 0)


        # Delviery Flow
        for m in range(1,args.M+1):
            for w in range(args.W):
                for p in range(args.P):
                    if(m-self.idata.P_p[p] >= 0):
                        self.sub.addConstr(self.vk[m-1,w,p] + quicksum(self.ak[m-self.idata.P_p[p],i,w,p] for i in range(args.I)) == self.vk[m,w,p] + quicksum(self.fk[m,w,j,p,g] for j in range(args.J) for g in range(args.G)))
                    else:
                        self.sub.addConstr(self.vk[m-1,w,p]  == self.vk[m,w,p] + quicksum(self.fk[m,w,j,p,g] for j in range(args.J) for g in range(args.G)))


        # Satify Demand Flow
        # Dual
        # rhs
        self.demand_cons = [[[0 for g in range(args.G)] for j in range(args.J)] for m in range(args.M+1)]
        for m in range(1,args.M+1):
            for j in range(args.J):
                for g in range(args.G):
                    self.demand_cons[m][j][g] = self.sub.addConstr(quicksum(self.fk[m,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[m,j,g] == self.idata.demand[self.state][self.k][m][j][g])

    def forward_run(self,u,v):

        self.u_pre = u
        self.v_pre = v

       
        # Initial Inventory Level in Short-term
        for w in range(self.args.W):
            for p in range(self.args.P):
                self.ini_invent_cons[w][p].setAttr(GRB.Attr.RHS, v[w,p].x)
                # print(v[w,p].x)


        # Staging Area Constraints
        for m in range(self.args.M+1):
            for w in range(self.args.W):
                self.stagarea_cons[m][w].setAttr(GRB.Attr.RHS, u[w].x)
                # print(u[w].x)
        

        for c in range(len(self.cut_rhs_temp)):
            self.cut_rhs.append(self.cut_rhs_temp[c])
            self.cut.append(self.cut_temp[c])

        self.cut_rhs_temp = []
        self.cut_temp = []

        self.sub.update()
        self.sub.setParam("OutputFlag", 0)
        self.sub.optimize()

        if self.sub.status == gp.GRB.OPTIMAL:   # 2 = optimal
            objval = self.sub.ObjVal
        else:
            print("Warning: no optimal solution, status =", self.sub.status)
            pdb.set_trace()

        replenmship_cost = 0
        Shortage_cost = 0
        acquire_cost = 0
       

        if(self.args.evaluate_switch == True):
 

            replenmship_cost = sum(self.args.price_strategic*self.idata.O_p[p]*self.aak[w,p].x - self.args.price_strategic*self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[w,p].x for w in range(self.args.W) for p in range(self.args.P))
            Shortage_cost =  sum(sum(self.idata.CU_g[g]*self.sk[m,j,g].x for j in range(self.args.J) for g in range(self.args.G)) for m in range(self.args.M+1))
            acquire_cost =  sum( sum(self.idata.O_p[p]*self.ak[m,i,w,p].x for i in range(self.args.I) for w in range(self.args.W) for p in range(self.args.P)) for m in range(self.args.M+1))


        return u,self.v,self.sub.ObjVal,replenmship_cost,Shortage_cost,acquire_cost

    def backward_run(self):


        # ---------------------------------------- Get Dual -----------------------------------------

        pi_8b = np.zeros((self.args.W, self.args.P))
        pi_8e = np.zeros((self.args.M+1, self.args.I))
        pi_8g = np.zeros((self.args.M+1, self.args.W))
        pi_8h = np.zeros((self.args.M+1, self.args.J, self.args.G))

        temp = 0

        # Initial Inventory Level in Short-term
        for w in range(self.args.W):
            for p in range(self.args.P):
                pi_8b[w][p] = self.ini_invent_cons[w][p].pi
                temp = temp + self.ini_invent_cons[w][p].pi*self.v_pre[w,p].x


        for m in range(self.args.M+1):
            for i in range(self.args.I):
                 pi_8e[m][i] = self.prod_cap_cons[m][i].pi
                 temp = temp + self.idata.B_i[i]*self.prod_cap_cons[m][i].pi


        # Staging Area Constraints
        for m in range(self.args.M+1):
            for w in range(self.args.W):
                pi_8g[m][w] = self.stagarea_cons[m][w].pi
                temp = temp + self.stagarea_cons[m][w].pi*self.u_pre[w].x
        
        # Demand Flow
        for m in range(1,self.args.M+1):
            for j in range(self.args.J):
                for g in range(self.args.G):
                    pi_8h[m][j][g] = self.demand_cons[m][j][g].pi
                    temp = temp + self.demand_cons[m][j][g].pi*self.idata.demand[self.state][self.k][m][j][g]

        Benders_cut_pi = 0

        # Cut
        if(self.cut):
            # print(self.stage,self.state)
            for c in range(len(self.cut_rhs)):
                temp = temp + self.cut[c].pi*self.cut_rhs[c]
                Benders_cut_pi = Benders_cut_pi + self.cut[c].pi*self.cut_rhs[c]


        if(abs(temp-self.sub.ObjVal) >= 1e-3):
            print("Subproblem problematic dual solution!")
            print("temp:",temp)
            print("obj:",self.sub.ObjVal)
            pdb.set_trace()


        return pi_8b, pi_8e, pi_8g, pi_8h, Benders_cut_pi, self.sub.ObjVal

    def add_cut(self,obj,scenario,pi_b,pi_c,Benders_cut_pi):

        temp_cut_itr = 0

        obj = sum(self.idata.MC_tran_matrix[self.stage][self.state][n]*obj[n] for n in range(self.args.N))

        if(scenario == self.k):

            if(self.phi.x < obj - cut_vio_thred and abs(self.phi.x - obj)/max(abs(self.phi.x),1e-10) > cut_vio_thred):

                temp_constraint = self.sub.addConstr(self.phi >= quicksum(self.idata.MC_tran_matrix[self.stage][self.state][n]*(quicksum(pi_b[n][w]*self.u_pre[w] for w in range(self.args.W)) 
                                                                    + quicksum(pi_c[n][w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                    + Benders_cut_pi[n]) for n in range(self.args.N)))

                temp_rhs = Benders_cut_pi
        
                self.cut_temp.append(temp_constraint)
                self.cut_rhs_temp.append(temp_rhs)

                temp_cut_itr = 1
                # print("stage:",self.stage,"state:",self.state,"SDDP Cut")
            else:
                temp_cut_itr = 0

        else:
            # cut sharing
            temp_constraint = self.sub.addConstr(self.phi >= quicksum(self.idata.MC_tran_matrix[self.stage][self.state][n]*(quicksum(pi_b[n][w]*self.u[w] for w in range(self.args.W)) 
                                                                    + quicksum(pi_c[n][w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                    + Benders_cut_pi[n]) for n in range(self.args.N)))

            temp_rhs = Benders_cut_pi
        
            self.cut_temp.append(temp_constraint)
            self.cut_rhs_temp.append(temp_rhs)


        return temp_cut_itr

class st_node:
    """A data structure that keeps stage-wise problems"""
    
    def __init__(self, args, input_data, state, stage, stage0=False):

        self.args = args
        self.idata = input_data

        self.state = state
        self.stage = stage
        self.stage0 = stage0

        self.cut_rhs = []
        self.cut = []

        self.cut_rhs_temp = []
        self.cut_temp = []
        
        
        
        self.model = gp.Model(f"Stage_{stage}_State_{state}_model")
        # Stage variable
        self.u = self.model.addVars(args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='uw')
        self.y = self.model.addVars(args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='yw')
        self.v = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vwp')
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        self.z = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='zwp')
        self.phi = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='phi')

        # Objective
        self.model.setObjective(quicksum(self.idata.E_w[w]*self.y[w] for w in range(args.W)) 
                              + quicksum(args.price_strategic*self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) + self.idata.H_p[p]*self.v[w,p] for w in range(args.W) for p in range(args.P))
                              + self.phi 
                            , GRB.MINIMIZE);



        # Staging Area Capacity
        # Dual
        # Receive self.u[parent_node,w] 
        self.b_staging_cap = [0  for W in range(args.W)]
        for w in range(args.W):
            if stage0 == True:
                self.model.addConstr(self.u[w] == self.y[w] + self.idata.Cap_w[w])
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
                    self.model.addConstr(self.v[w,p] == self.x[w,p] - self.z[w,p] + self.idata.II_w[w][p])
                else:
                    self.c_inv_level[w][p] = self.model.addConstr(self.v[w,p] - self.x[w,p] + self.z[w,p] ==  0)

    def forward_run(self,u=None,v=None):

        # Input first-stage solution

        self.u_pre = u
        self.v_pre = v

        
        if self.stage0 != True:
            # Staging Capacity
            for w in range(self.args.W):
                self.b_staging_cap[w].setAttr(GRB.Attr.RHS, u[w].x)

            # Invenory Level
            for w in range(self.args.W):
                for p in range(self.args.P):
                    self.c_inv_level[w][p].setAttr(GRB.Attr.RHS, v[w,p].x)


        for c in range(len(self.cut_rhs_temp)):
            self.cut_rhs.append(self.cut_rhs_temp[c])
            self.cut.append(self.cut_temp[c])

        self.cut_rhs_temp = []
        self.cut_temp = []

        self.model.update()
        self.model.setParam("OutputFlag", 0)
        self.model.optimize()

        if self.model.status == gp.GRB.OPTIMAL:   # 2 = optimal
            objval = self.model.ObjVal
        else:
            print("Warning: no optimal solution, status =", self.model.status)
            pdb.set_trace()

        staging_area_expand_cost_temp = 0
        inventory_expand_cost_temp = 0
        holding_cost_temp = 0


        if(self.args.evaluate_switch == True):

            staging_area_expand_cost_temp = sum(self.idata.E_w[w]*self.y[w].x for w in range(self.args.W))                                   
            inventory_expand_cost_temp =  sum(self.args.price_strategic*self.idata.O_p[p]*(self.x[w,p].x - self.idata.R_p[p]*self.z[w,p].x) for w in range(self.args.W) for p in range(self.args.P))
            holding_cost_temp = sum(self.idata.H_p[p]*self.v[w,p].x for w in range(self.args.W) for p in range(self.args.P))

        return self.u,self.v, self.model.ObjVal, staging_area_expand_cost_temp, inventory_expand_cost_temp,holding_cost_temp

    def backward_run(self):


        # ---------------------------------------- Get Dual -----------------------------------------

        pi_b = np.zeros((self.args.W))
        pi_c = np.zeros((self.args.W, self.args.P))


        temp = 0

        if self.stage0 != True:
            # Staging Capacity
            for w in range(self.args.W):
                pi_b[w] = self.b_staging_cap[w].pi
                temp = temp + self.b_staging_cap[w].pi*self.u_pre[w].x


            # Invenory Level
            for w in range(self.args.W):
                for p in range(self.args.P):
                    pi_c[w][p] = self.c_inv_level[w][p].pi
                    temp = temp + self.c_inv_level[w][p].pi*self.v_pre[w,p].x


        Benders_cut_pi = 0

        # Cut
        if(self.cut):
            # print(self.stage,self.state)
            # pdb.set_trace()
            for c in range(len(self.cut_rhs)):
                temp = temp + self.cut[c].pi*self.cut_rhs[c]
                Benders_cut_pi = Benders_cut_pi + self.cut[c].pi*self.cut_rhs[c]

                

        if(abs(temp-self.model.ObjVal) >= 1e-3):
            print("iteration:",iter,"stage:",self.stage,"problematic dual solution!")
            print("temp:",temp)
            print("obj:",self.model.ObjVal)
            pdb.set_trace()


        return pi_b,pi_c,Benders_cut_pi,self.model.ObjVal

    def add_cut(self,obj,state_sample_path,pi_8b,pi_8e,pi_8g,pi_8h,Benders_cut_pi):

        temp_cut_itr = 0

        obj = (1/self.args.K)*sum(obj[k] for k in range(self.args.K))

        if(state_sample_path == self.state):

            if(self.phi.x < obj - cut_vio_thred and abs(self.phi.x - obj)/max(abs(self.phi.x),1e-10) > cut_vio_thred):
                temp_constraint = self.model.addConstr(self.phi >= (1/self.args.K)*quicksum(quicksum(self.v[w,p]*pi_8b[k][w][p] for w in range(self.args.W) for p in range(self.args.P))
                                                                                  +quicksum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I))
                                                                                  +quicksum(pi_8g[k][m][w]*self.u[w] for m in range(self.args.M+1) for w in range(self.args.W))
                                                                                  +quicksum(pi_8h[k][m][j][g]*self.idata.demand[self.state][k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                                                                                  +Benders_cut_pi[k] for k in range(self.args.K)))

                temp_rhs = (1/self.args.K)*sum(pi_8h[k][m][j][g]*self.idata.demand[self.state][k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G) for k in range(self.args.K))
                temp_rhs = temp_rhs + (1/self.args.K)*sum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I) for k in range(self.args.K))
                temp_rhs = temp_rhs + (1/self.args.K)*sum(Benders_cut_pi[k] for k in range(self.args.K))


                self.cut_temp.append(temp_constraint)
                self.cut_rhs_temp.append(temp_rhs)


                temp_cut_itr = 1
            else:
                temp_cut_itr = 0

        else:
            # cut sharing

            temp_constraint = self.model.addConstr(self.phi >= (1/self.args.K)*quicksum(quicksum(self.v[w,p]*pi_8b[k][w][p] for w in range(self.args.W) for p in range(self.args.P))
                                                                                  +quicksum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I))
                                                                                  +quicksum(pi_8g[k][m][w]*self.u[w] for m in range(self.args.M+1) for w in range(self.args.W))
                                                                                  +quicksum(pi_8h[k][m][j][g]*self.idata.demand[self.state][k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                                                                                  +Benders_cut_pi[k] for k in range(self.args.K)))

            temp_rhs = (1/self.args.K)*sum(pi_8h[k][m][j][g]*self.idata.demand[self.state][k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G) for k in range(self.args.K))
            temp_rhs = temp_rhs + (1/self.args.K)*sum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I) for k in range(self.args.K))
            temp_rhs = temp_rhs + (1/self.args.K)*(1/self.args.K)*sum(Benders_cut_pi[k] for k in range(self.args.K))

            self.cut_temp.append(temp_constraint)
            self.cut_rhs_temp.append(temp_rhs)


        return temp_cut_itr


class solve_SDDP:
    def __init__(self, args, input_data):

        self.args = args
        self.idata = input_data


        self.stage = []

        for t in range(args.T):   
            stage_temp = []
            for n in range(args.N):   

                
                op_list = []
                for k in range(args.K):
                    op_list.append(op_node(args, input_data, state=n, stage=t, k=k, last_stage=False))

                st = st_node(args,input_data,state=n,stage=t,stage0=(t == 0))
                stage_temp.append([st, op_list])

            self.stage.append(stage_temp)

        print("Memory Used.",sys.getsizeof(self.stage) )

    def sample_path(self, args):

        path = []
        
        self.initial_state = args.initial_state
        state = self.initial_state
        scenario = np.random.choice(args.K, 1)[0]

        path.append([state,scenario])
        for stage in range(args.T):
            if(self.args.evaluate_switch == True or self.args.sample_path == "MC"):
                next_state = np.random.choice(args.N, 1, p=self.idata.MC_tran_matrix[stage][state].tolist())
                scenario = np.random.choice(args.K, 1)[0]
            else:
                next_state = np.random.choice(args.N, 1)
                scenario = np.random.choice(args.K, 1)[0]
                # print("sample_path")
            state = next_state[0]
            path.append([state,scenario])

        return path

    def termination_check(self, iter, relative_gap, LB, start, cutviol_iter):
        flag = 0
        Elapsed = time.time() - start
        if(iter > self.args.MAX_ITER):
            flag = 1
            print("max iteration is reached", "stop at iteration", iter)
        elif (Elapsed > self.args.time_limit):
            flag = 2
            print("time limit is reached", "stop at iteration", iter)
        elif (cutviol_iter > self.args.CUTVIOL_MAXITER):
            flag = 3
            print("cut violation is reached", "stop at iteration", iter)
        else:
            if iter > self.args.STALL:
                relative_gap = ((LB[iter-1]-LB[iter-1-self.args.STALL])/max(1e-10,abs(LB[iter-1-self.args.STALL])))
                if relative_gap < self.args.LB_TOL:
                    flag = 4
                    print(relative_gap)
                    print("the LB is not making significant progress", "stop at iteration", iter)
        return flag, Elapsed

    def run(self):

        start = time.time()
        
        relative_gap = 1e10
        cutviol_iter = 0

        iter_list = []
        LB_list = []

        for iter in range(self.args.MAX_ITER):

            # sample path
            sample_path = self.sample_path(self.args)
            iter_list.append(iter)
            # print(sample_path)

            u = 0
            v = 0
            obj_ex = 0
            LB_temp = 0


            for t in range(self.args.T):

                print("stage:",t)
                # st node forward

                next_u = 0
                next_v = 0

                # st node forward
                
                if t == 0:
                    next_u,next_v,ObjVal,temp1,temp2,temp3 = self.stage[t][sample_path[t][0]][0].forward_run()
                    LB_temp = ObjVal
                else:
                    for n in range(self.args.N):
                        if n == sample_path[t][0]:
                            next_u,next_v,ObjVal,temp1,temp2,temp3 = self.stage[t][n][0].forward_run(u,v)
                        else:
                            temp_u,temp_v,ObjVal,temp1,temp2,temp3 = self.stage[t][n][0].forward_run(u,v)

                u = next_u
                v = next_v

                
                # op node forward

                for k in range(self.args.K):

                    if k == sample_path[t][1]:
                        next_u,next_v,ObjVal,temp1,temp2,temp3 = self.stage[t][sample_path[t][0]][1][k].forward_run(u,v)
                    else:
                        temp_u,temp_v,ObjVal,temp1,temp2,temp3 = self.stage[t][sample_path[t][0]][1][k].forward_run(u,v)


                u = next_u
                v = next_v



            pi_8b_total = np.zeros((self.args.K,self.args.W, self.args.P))
            pi_8e_total = np.zeros((self.args.K,self.args.M+1, self.args.I))
            pi_8g_total = np.zeros((self.args.K,self.args.M+1, self.args.W))
            pi_8h_total = np.zeros((self.args.K,self.args.M+1, self.args.J, self.args.G))
            Benders_cut_total_op = np.zeros((self.args.K))
            obj_total_op = np.zeros((self.args.K))

            pi_b_total = np.zeros((self.args.N, self.args.W))
            pi_c_total = np.zeros((self.args.N, self.args.W, self.args.P))
            Benders_cut_total_st = np.zeros((self.args.N))
            obj_total_st = np.zeros((self.args.N))


            # ----------------------------------- Backward -----------------------------------
            for t in reversed(range(self.args.T)):

                print("stage:",t)

                # op node backward
                for k in range(self.args.K):
                    if t != self.args.T-1:
                        print("op add cut")
                        cutviol_iter += self.stage[t][sample_path[t][0]][1][k].add_cut(obj_total_st,k,pi_b_total,pi_c_total,Benders_cut_total_st)
                    print("op back")
                    pi_8b_total[k],pi_8e_total[k],pi_8g_total[k],pi_8h_total[k],Benders_cut_total_op[k],obj_total_op[k] = self.stage[t][sample_path[t][0]][1][k].backward_run()

                # st node backward
                if t == 0:
                    cutviol_iter += self.stage[t][sample_path[t][0]][0].add_cut(obj_total_op,n,pi_8b_total,pi_8e_total,pi_8g_total,pi_8h_total,Benders_cut_total_op)
                else:                    
                    for n in range(self.args.N):
                        print("st add cut")
                        cutviol_iter +=  self.stage[t][n][0].add_cut(obj_total_op,n,pi_8b_total,pi_8e_total,pi_8g_total,pi_8h_total,Benders_cut_total_op)
                        print("st back")
                        pi_b_total[n],pi_c_total[n],Benders_cut_total_st[n],obj_total_st[n] = self.stage[t][n][0].backward_run()


            LB_list.append(LB_temp)
            print("iteration:", iter, "LB:", LB_temp)

            if(iter%20 == 0):
                print("iteration:", iter, "LB:", LB_temp)

            # pdb.set_trace()

            
            # ----------------------------------- Stop Criteria -----------------------------------
            flag, Elapsed = self.termination_check(iter, relative_gap, LB_list, start, cutviol_iter)
            if flag != 0:
                train_time = Elapsed
                print("training total time:", Elapsed)

                self.args.evaluate_switch = True

                # plt.plot(LB_list, marker='o', linestyle='-', color='b', label='Data')

                # # Customize the plot
                # plt.xlabel("iter")
                # plt.ylabel("LB")
                # plt.legend()
                # plt.grid(True)
                # plt.show()

                # pdb.set_trace()

                # ---------------------------------------------------- Polciy Simulation ----------------------------------------------------
                time_test = time.time()
                simulate_iter = 1000
                solution_u = np.zeros((self.args.T+1,self.args.N))
                solution_v = np.zeros((self.args.T+1,self.args.N))
                solution_obj = np.zeros((self.args.T+1,self.args.N))
                
                solution_total = np.zeros((simulate_iter))


                staging_area_expand_cost = np.zeros((self.args.T+1,self.args.N))
                inventory_expand_cost = np.zeros((self.args.T+1,self.args.N))
                replenmship_cost = np.zeros((self.args.T+1,self.args.N))
                acquire_cost = np.zeros((self.args.T+1,self.args.N))
                Shortage_cost = np.zeros((self.args.T+1,self.args.N))
                holding_cost = np.zeros((self.args.T+1,self.args.N))

                path_count = np.zeros((self.args.T+1,self.args.N))

                if(self.args.Policy == "baseline"):

                    for k in range(self.args.K):
                        for m in range(1,self.args.M+1):
                            for j in range(self.args.J):
                                for g in range(self.args.G):
                                    self.stage_root.k_demand[k][m][j][g].setAttr(GRB.Attr.RHS, self.demnad[self.args.initial_state][k][m][j][g])


                    for stage in range(self.args.T-1):
                        for state in range(self.args.N):
                            for k in range(self.args.K):
                                for m in range(1,self.args.M+1):
                                    for j in range(self.args.J):
                                        for g in range(self.args.G):
                                            self.stage[stage][state].k_demand[k][m][j][g].setAttr(GRB.Attr.RHS, self.demnad[state][k][m][j][g])

                    for state in range(self.args.N):
                        for k in range(self.args.K):
                            for m in range(1,self.args.M+1):
                                for j in range(self.args.J):
                                    for g in range(self.args.G):
                                        self.stage_leaf[state].k_demand[k][m][j][g].setAttr(GRB.Attr.RHS, self.demnad[state][k][m][j][g])

                for counts in range(simulate_iter):

                    # sample path
                    sample_path = self.sample_path(self.args)
                    # print(sample_path)

                    u = 0
                    v = 0
                    obj_ex = 0
                    staging_area_expand_cost_temp = 0
                    inventory_expand_cost_temp = 0
                    replenmship_cost_temp = 0
                    Shortage_cost_temp = 0
                    acquire_cost_temp = 0
                    holdings_cost_temp = 0
                    # ---------------------------------------------------- Forward ----------------------------------------------------
                    if(self.args.Strategic_node_sovling == 0):
                        u,v,obj_ex,staging_area_expand_cost_temp,inventory_expand_cost_temp,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp,holdings_cost_temp = self.stage_root.forward_run()

                    elif(self.args.Strategic_node_sovling == 1):
                        u,v,obj_ex,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,staging_area_expand_cost_temp,inventory_expand_cost_temp,holdings_cost_temp = self.stage_root.forward_run()
                        for k in range(self.args.K):
                            pi_8b,pi_8e,pi_8g,pi_8h,pi_8i,sub_opt,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp = self.stage_root.sub.run(u,v,self.idata.demand[self.args.initial_state][k])

                    solution_u[0][self.args.initial_state] += sum(u[w].x for w in range(self.args.W))
                    solution_v[0][self.args.initial_state] += sum(v[w,p].x for w in range(self.args.W) for p in range(self.args.P))
                    solution_obj[0][self.args.initial_state] += obj_ex
                    staging_area_expand_cost[0][self.args.initial_state] += staging_area_expand_cost_temp
                    inventory_expand_cost[0][self.args.initial_state] += inventory_expand_cost_temp
                    replenmship_cost[0][self.args.initial_state] += replenmship_cost_temp
                    Shortage_cost[0][self.args.initial_state] += Shortage_cost_temp
                    acquire_cost[0][self.args.initial_state] += acquire_cost_temp
                    holding_cost[0][self.args.initial_state] += holdings_cost_temp
                    path_count[0][self.args.initial_state] += 1
                    solution_total[counts] = solution_total[counts] + obj_ex

                    
                    for stage in range(self.args.T-1):
                        if(self.args.Strategic_node_sovling == 0):
                            u,v,obj_ex,staging_area_expand_cost_temp,inventory_expand_cost_temp,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp,holdings_cost_temp = self.stage[stage][sample_path[stage]].forward_run(u,v)
                        elif(self.args.Strategic_node_sovling == 1):
                            u,v,obj_ex,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,staging_area_expand_cost_temp,inventory_expand_cost_temp,holdings_cost_temp = self.stage[stage][sample_path[stage]].forward_run(u,v)
                            for k in range(self.args.K):
                                pi_8b,pi_8e,pi_8g,pi_8h,pi_8i,sub_opt,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp = self.stage[stage][sample_path[stage]].sub.run(u,v,self.idata.demand[sample_path[stage]][k])
                        
                        solution_u[stage+1][sample_path[stage]] += sum(u[w].x for w in range(self.args.W))
                        solution_v[stage+1][sample_path[stage]] += sum(v[w,p].x for w in range(self.args.W) for p in range(self.args.P))
                        solution_obj[stage+1][sample_path[stage]] += obj_ex
                        staging_area_expand_cost[stage+1][sample_path[stage]] += staging_area_expand_cost_temp
                        inventory_expand_cost[stage+1][sample_path[stage]] += inventory_expand_cost_temp
                        replenmship_cost[stage+1][sample_path[stage]] += replenmship_cost_temp
                        Shortage_cost[stage+1][sample_path[stage]] += Shortage_cost_temp
                        acquire_cost[stage+1][sample_path[stage]] += acquire_cost_temp
                        holding_cost[stage+1][sample_path[stage]] += holdings_cost_temp
                        path_count[stage+1][sample_path[stage]] += 1
                        solution_total[counts] = solution_total[counts] + obj_ex
                    
                    if(self.args.Strategic_node_sovling == 0):
                        u,v,obj_ex,staging_area_expand_cost_temp,inventory_expand_cost_temp,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp,holdings_cost_temp = self.stage_leaf[sample_path[self.args.T-1]].forward_run(u,v)
                    elif(self.args.Strategic_node_sovling == 1):
                        u,v,obj_ex,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,staging_area_expand_cost_temp,inventory_expand_cost_temp,holdings_cost_temp = self.stage_leaf[sample_path[self.args.T-1]].forward_run(u,v)
                        for k in range(self.args.K):
                                pi_8b,pi_8e,pi_8g,pi_8h,pi_8i,sub_opt,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp = self.stage_leaf[sample_path[self.args.T-1]].sub.run(u,v,self.idata.demand[sample_path[stage]][k])
                        
                    solution_u[self.args.T][sample_path[self.args.T-1]] += sum(u[w].x for w in range(self.args.W))
                    solution_v[self.args.T][sample_path[self.args.T-1]] += sum(v[w,p].x for w in range(self.args.W) for p in range(self.args.P))
                    solution_obj[self.args.T][sample_path[self.args.T-1]] += obj_ex
                    staging_area_expand_cost[self.args.T][sample_path[self.args.T-1]] += staging_area_expand_cost_temp
                    inventory_expand_cost[self.args.T][sample_path[self.args.T-1]] += inventory_expand_cost_temp
                    replenmship_cost[self.args.T][sample_path[self.args.T-1]] += replenmship_cost_temp
                    Shortage_cost[self.args.T][sample_path[self.args.T-1]] += Shortage_cost_temp
                    acquire_cost[self.args.T][sample_path[self.args.T-1]] += acquire_cost_temp
                    holding_cost[self.args.T][sample_path[self.args.T-1]] += holdings_cost_temp
                    path_count[self.args.T][sample_path[self.args.T-1]] += 1
                    solution_total[counts] = solution_total[counts] + obj_ex

                    # print(solution_total[counts])

                
                solution = []
                time_test_end = time.time()


                # pdb.set_trace()
                for t in range(self.args.T+1):
                    for n in range(self.args.N):
                        if(path_count[t][n] == 0):
                            solution.append([t,n,solution_u[t][n],solution_v[t][n],solution_obj[t][n],staging_area_expand_cost[t][n],inventory_expand_cost[t][n],replenmship_cost[t][n],Shortage_cost[t][n],acquire_cost[t][n],holding_cost[t][n]])
                        else:
                            solution.append([t,n,solution_u[t][n]/path_count[t][n],solution_v[t][n]/path_count[t][n],solution_obj[t][n]/path_count[t][n],staging_area_expand_cost[t][n]/path_count[t][n],inventory_expand_cost[t][n]/path_count[t][n],replenmship_cost[t][n]/path_count[t][n],Shortage_cost[t][n]/path_count[t][n],acquire_cost[t][n]/path_count[t][n],holding_cost[t][n]/path_count[t][n]])
                
                
                df = pd.DataFrame(solution, columns=[ 'stage','state','Staging Area Capacity','Inventory Level','obj','staging_area_expand_cost','inventory_expand_cost','replenmship_cost','Shortage_cost','acquire_cost','holding_cost'])
                filename = str(self.args.Model) + str(self.args.Strategic_node_sovling) + "result_Stage_" + str(self.args.T) + "_States_" + str(self.args.N) + "_Study_" + str(self.args.J) + "_month_" + str(self.args.M)  + "_K_" + str(self.args.K)  + "_Pp_" + str(self.args.P_p_factor) + "_Cu_" + str(self.args.C_u_factor) + "_Ew_" +  str(self.args.E_w_factor) + "_Cpw_" +  str(self.args.Cp_w_factor)  + "_policy_" +  str(self.args.Policy)

                df.to_csv(f'{filename}.csv', index=False) 
                print("testing_time:", time_test_end-time_test)
                print("LB:", LB_temp)
                print("UB mean_so:", np.mean(solution_total))
                print("UB std_sol:", np.std(solution_total))
                print("gap:", (np.mean(solution_total) - LB_temp)/np.mean(solution_total))
                print("sub_problem_time:",sub_problem_time)
                print("operation_Benders_problem_time:",operation_Benders_problem_time)
                print("strategic_problem_time:",strategic_problem_time)
                # plt.boxplot(solution_total)
                # plt.savefig(f'{filename}.png')
                # plt.close()
                np.save(f'{filename}.npy', solution_total) 


                break
