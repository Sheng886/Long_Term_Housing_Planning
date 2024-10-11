from model import func
from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import time
import pdb



cut_vio_thred = 1e-5

class subproblem:

    def __init__(self, args, input_data):

        self.args = args
        self.idata = input_data
        self.sub = gp.Model("sub")

        # Scen_Path variable
        self.vk = self.sub.addVars(args.M+1, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vktwp')
        self.bk = self.sub.addVars(args.M+1, args.I, lb=0.0, vtype=GRB.CONTINUOUS, name='bkti')
        self.ak = self.sub.addVars(args.M+1, args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aktiwp')
        self.fk = self.sub.addVars(args.M+1, args.W, args.J, args.P, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='fktwjpg')
        self.sk = self.sub.addVars(args.M+1, args.J, args.G, lb=0.0, vtype=GRB.CONTINUOUS, name='sktjg')
        self.aak = self.sub.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='aaktwp')
        self.bbk = self.sub.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='bbktwp')

        
        self.sub.setObjective(quicksum(self.idata.O_p[p]*self.aak[w,p] - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[w,p] for w in range(args.W) for p in range(args.P)) 
                              + quicksum(quicksum(self.idata.O_p[p]*self.ak[m,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                       + quicksum(self.idata.CU_g[g]*self.sk[m,j,g] for j in range(args.J) for g in range(args.G)) for m in range(args.M+1)), GRB.MINIMIZE);


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
                    self.demand_cons[m][j][g] = self.sub.addConstr(quicksum(self.fk[m,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[m,j,g] == 0)

        # Assumption Replensih by MHS
        # Dual
        # rhs
        self.MHSP_assu_cons = [[0 for p in range(args.P)] for w in range(args.W)]
        for w in range(args.W):
            for p in range(args.P):
                self.MHSP_assu_cons[w][p] = self.sub.addConstr(self.vk[args.M,w,p] + self.aak[w,p] - self.bbk[w,p] == 0)


    def run(self,u,v,demand):
       
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

        
        # Demand Flow
        # self.idata.demand[stage][state][k][g][m]*self.idata.J_pro[j]
        for m in range(1,self.args.M+1):
            for j in range(self.args.J):
                for g in range(self.args.G):
                    self.demand_cons[m][j][g].setAttr(GRB.Attr.RHS, demand[m][j][g])
        
    

        # Assumption Replensih by MHS
        for w in range(self.args.W):
            for p in range(self.args.P):
                self.MHSP_assu_cons[w][p].setAttr(GRB.Attr.RHS, v[w,p].x)
                # print(v[w,p].x)



        self.sub.update()
        self.sub.setParam("OutputFlag", 0)
        self.sub.optimize()

        pi_8b = np.zeros((self.args.W, self.args.P))
        pi_8e = np.zeros((self.args.M+1, self.args.I))
        pi_8g = np.zeros((self.args.M+1, self.args.W))
        pi_8h = np.zeros((self.args.M+1, self.args.J, self.args.G))
        pi_8i = np.zeros((self.args.W, self.args.P))

        temp = 0

        # Initial Inventory Level in Short-term
        for w in range(self.args.W):
            for p in range(self.args.P):
                pi_8b[w][p] = self.ini_invent_cons[w][p].pi
                temp = temp + self.ini_invent_cons[w][p].pi*v[w,p].x


        for m in range(self.args.M+1):
            for i in range(self.args.I):
                 pi_8e[m][i] = self.prod_cap_cons[m][i].pi
                 temp = temp + self.idata.B_i[i]*self.prod_cap_cons[m][i].pi


        # Staging Area Constraints
        for m in range(self.args.M+1):
            for w in range(self.args.W):
                pi_8g[m][w] = self.stagarea_cons[m][w].pi
                temp = temp + self.stagarea_cons[m][w].pi*u[w].x
        
        # Demand Flow
        for m in range(1,self.args.M+1):
            for j in range(self.args.J):
                for g in range(self.args.G):
                    pi_8h[m][j][g] = self.demand_cons[m][j][g].pi
                    temp = temp + self.demand_cons[m][j][g].pi*demand[m][j][g]

        # Assumption Replensih by MHS
        for w in range(self.args.W):
            for p in range(self.args.P):
                pi_8i[w][p] = self.MHSP_assu_cons[w][p].pi
                temp = temp + self.MHSP_assu_cons[w][p].pi*v[w,p].x

        if(abs(temp-self.sub.ObjVal) >= 1e-5):
            print("Subproblem problematic dual solution!")
            print("temp:",temp)
            print("obj:",self.sub.ObjVal)
            pdb.set_trace()

        
        replenmship_cost = 0
        Shortage_cost = 0
        acquire_cost = 0

        if(self.args.evaluate_switch == True):
            replenmship_cost = sum(self.idata.O_p[p]*self.aak[w,p].x - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[w,p].x for w in range(self.args.W) for p in range(self.args.P))
            Shortage_cost =  sum(sum(self.idata.CU_g[g]*self.sk[m,j,g].x for j in range(self.args.J) for g in range(self.args.G)) for m in range(self.args.M+1))
            acquire_cost =  sum( sum(self.idata.O_p[p]*self.ak[m,i,w,p].x for i in range(self.args.I) for w in range(self.args.W) for p in range(self.args.P)) for m in range(self.args.M+1))


        return pi_8b,pi_8e,pi_8g,pi_8h,pi_8i,self.sub.ObjVal,replenmship_cost,Shortage_cost,acquire_cost

class StageProblem_Decomposition:
    """A data structure that keeps stage-wise problems"""
    
    def __init__(self, args, input_data, state, stage, last_stage=False, stage0=False):

        self.args = args
        self.idata = input_data
        
        self.sub = subproblem(args,input_data)

        self.state = state
        self.stage = stage
        self.stage0 = stage0
        self.last_stage = last_stage

        self.cut_rhs = []
        self.cut = []

        self.cut_rhs_temp = []
        self.cut_temp = []
        
        
        self.u_value = np.zeros((args.W))
        self.y_value = np.zeros((args.W))
        self.v_value = np.zeros((args.W, args.P))
        self.x_value = np.zeros((args.W, args.P))
        self.z_value = np.zeros((args.W, args.P))
        
        
        self.model = gp.Model(f"Stage_{stage}_State_{state}_model")
        # Stage variable
        self.u = self.model.addVars(args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='uw')
        self.y = self.model.addVars(args.W, lb=0.0, vtype=GRB.CONTINUOUS, name='yw')
        self.v = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='vwp')
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        self.z = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='zwp')
        self.phi = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='phi')
        if(last_stage == False):
            self.theta = self.model.addVars(args.N, lb=0.0, vtype=GRB.CONTINUOUS, name='theta')

        # Objective
        if(last_stage == False):
            self.model.setObjective(quicksum(self.idata.E_w[w]*self.y[w] for w in range(args.W)) 
                                  + quicksum(self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) + self.idata.H_p[p]*self.v[w,p] for w in range(args.W) for p in range(args.P))
                                  + self.phi
                                  + quicksum(self.idata.MC_tran_matrix[state][n]*self.theta[n] for n in range(args.N)) 
                                , GRB.MINIMIZE);
        else:
            self.model.setObjective(quicksum(self.idata.E_w[w]*self.y[w] for w in range(args.W)) 
                                  + quicksum(self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) + self.idata.H_p[p]*self.v[w,p] for w in range(args.W) for p in range(args.P))
                                  + self.phi
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


        if(args.Policy == "WS"):
            for w in range(args.W):
                self.model.addConstr(self.u[w] == 0)
        elif(args.Policy == "avg"):
            if stage0 != True:
                for w in range(args.W):
                    self.model.addConstr(self.y[w] == 0)
                    for p in range(args.P):
                        self.model.addConstr(self.x[w,p] == 0 )
                        self.model.addConstr(self.z[w,p] == 0)



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



        # ---------------------------------------- Benders Decompoistion on Strategic Node  -----------------------------------------

        LB = 0
        UB = 1e10
        eps = 1e-3


        while(True):
            
            self.model.update()
            self.model.setParam("OutputFlag", 0)
            self.model.optimize()
            
            for w in range(self.args.W):
              self.u_value[w] = self.u[w].x
              self.y_value[w] = self.y[w].x
              for p in range(self.args.P):
                  self.v_value[w][p] = self.v[w,p].x
                  self.x_value[w][p] = self.x[w,p].x
                  self.z_value[w][p] = self.z[w,p].x
                  

            LB = self.model.ObjVal

            pi_8b = np.zeros((self.args.K, self.args.W, self.args.P))
            pi_8e = np.zeros((self.args.K, self.args.M+1, self.args.I))
            pi_8g = np.zeros((self.args.K, self.args.M+1, self.args.W))
            pi_8h = np.zeros((self.args.K, self.args.M+1, self.args.J, self.args.G))
            pi_8i = np.zeros((self.args.K, self.args.W, self.args.P))
            sub_opt = np.zeros((self.args.K))

            sub_opt_total = 0

            for k in range(self.args.K):
                if(self.stage0 == False):
                    pi_8b[k],pi_8e[k],pi_8g[k],pi_8h[k],pi_8i[k],sub_opt[k],temp1,temp2,temp3 = self.sub.run(self.u,self.v,self.idata.demand[self.stage][self.state][k])
                else:
                    pi_8b[k],pi_8e[k],pi_8g[k],pi_8h[k],pi_8i[k],sub_opt[k],temp1,temp2,temp3 = self.sub.run(self.u,self.v,self.idata.demand_root[k])
                sub_opt_total = sub_opt_total + sub_opt

            sub_opt_total = (1/self.args.K)*sum(sub_opt[k] for k in range(self.args.K))

            # ---------------------------------------- UB,LB Check  -----------------------------------------

            temp_UB = 0

            if(self.last_stage == False):
                temp_UB = sub_opt_total + sum(self.idata.E_w[w]*self.y[w].x for w in range(self.args.W)) 
                temp_UB = temp_UB + sum(self.idata.O_p[p]*(self.x[w,p].x - self.idata.R_p[p]*self.z[w,p].x) for w in range(self.args.W) for p in range(self.args.P))
                temp_UB = temp_UB + sum(self.idata.MC_tran_matrix[self.state][n]*self.theta[n].x for n in range(self.args.N))
            else:
                temp_UB = sub_opt_total + sum(self.idata.E_w[w]*self.y[w].x for w in range(self.args.W)) 
                temp_UB = temp_UB + sum(self.idata.O_p[p]*(self.x[w,p].x - self.idata.R_p[p]*self.z[w,p].x) for w in range(self.args.W) for p in range(self.args.P))

            UB = min(UB,temp_UB)
            # print("LB/UB:",LB,UB)

            if(abs((UB - LB)/UB) <= eps):
                if(self.args.Cost_print == True):
                    print("Benders Strategic Node Cost:",LB)
                    print("Benders Second-stage Cost:", sub_opt_total)
                    for k in range(self.args.K):
                        print(f"Benders sub {k} Cost",sub_opt[k])
                break

            # ---------------------------------------- Benders Cut  -----------------------------------------

            if(self.phi.x < sub_opt_total - cut_vio_thred and abs(self.phi.x - sub_opt_total)/max(abs(self.phi.x),1e-10) > cut_vio_thred):

                if(self.stage0 == False):
                    temp_constraint = self.model.addConstr(self.phi >= (1/self.args.K)*quicksum(quicksum(self.v[w,p]*pi_8b[k][w][p] for w in range(self.args.W) for p in range(self.args.P))
                                                                                      +quicksum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I))
                                                                                      +quicksum(pi_8g[k][m][w]*self.u[w] for m in range(self.args.M+1) for w in range(self.args.W))
                                                                                      +quicksum(pi_8h[k][m][j][g]*self.idata.demand[self.stage][self.state][k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                                                                                      +quicksum(pi_8i[k][w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                                      for k in range(self.args.K)))

                    temp_rhs = (1/self.args.K)*sum(pi_8h[k][m][j][g]*self.idata.demand[self.stage][self.state][k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G) for k in range(self.args.K))
                    temp_rhs = temp_rhs + (1/self.args.K)*sum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I) for k in range(self.args.K))

                    self.cut.append(temp_constraint)
                    self.cut_rhs.append(temp_rhs)
                
                else:
                    temp_constraint = self.model.addConstr(self.phi >= (1/self.args.K)*quicksum(quicksum(self.v[w,p]*pi_8b[k][w][p] for w in range(self.args.W) for p in range(self.args.P))
                                                                                      +quicksum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I))
                                                                                      +quicksum(pi_8g[k][m][w]*self.u[w] for m in range(self.args.M+1) for w in range(self.args.W))
                                                                                      +quicksum(pi_8h[k][m][j][g]*self.idata.demand_root[k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                                                                                      +quicksum(pi_8i[k][w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                                      for k in range(self.args.K)))

                    temp_rhs = (1/self.args.K)*sum(pi_8h[k][m][j][g]*self.idata.demand_root[k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G) for k in range(self.args.K))
                    temp_rhs = temp_rhs + (1/self.args.K)*sum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I) for k in range(self.args.K))
               
                    self.cut.append(temp_constraint)
                    self.cut_rhs.append(temp_rhs)


                
                # print("stage:",self.stage,"state:",self.state,"add Benders cut")




        staging_area_expand_cost_temp = 0
        inventory_expand_cost_temp = 0
        holding_cost_temp = 0


        if(self.args.evaluate_switch == True):

            staging_area_expand_cost_temp = sum(self.idata.E_w[w]*self.y[w].x for w in range(self.args.W))                                   
            inventory_expand_cost_temp =  sum(self.idata.O_p[p]*(self.x[w,p].x - self.idata.R_p[p]*self.z[w,p].x) for w in range(self.args.W) for p in range(self.args.P))
            holding_cost_temp = sum(self.idata.H_p[p]*self.v[w,p].x for w in range(self.args.W) for p in range(self.args.P))

           

        return self.u,self.v,LB, pi_8b, pi_8e, pi_8g, pi_8h, pi_8i, staging_area_expand_cost_temp, inventory_expand_cost_temp,holding_cost_temp

    def backward_run(self,iter):


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
            for c in range(len(self.cut_rhs)):
                temp = temp + self.cut[c].pi*self.cut_rhs[c]
                Benders_cut_pi = Benders_cut_pi + self.cut[c].pi*self.cut_rhs[c]
                


        if(abs(temp-self.model.ObjVal) >= 1e-3):
            print("iteration:",iter,"stage:",self.stage,"problematic dual solution!")
            print("temp:",temp)
            print("obj:",self.model.ObjVal)
            pdb.set_trace()


        return pi_b,pi_c,Benders_cut_pi,self.model.ObjVal

    def add_cut(self,obj,stage_next,state_sample_path,state_next,pi_b,pi_c,Benders_cut_pi):

        temp_cut_itr = 0

        if(self.state == state_sample_path or self.stage0==True):

            if(self.theta[state_next].x < obj - cut_vio_thred and abs(self.theta[state_next].x - obj)/max(abs(self.theta[state_next].x),1e-10) > cut_vio_thred):

                temp_constraint = self.model.addConstr(self.theta[state_next] >= quicksum(pi_b[w]*self.u[w] for w in range(self.args.W)) 
                                                                                + quicksum(pi_c[w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                                + Benders_cut_pi)

                temp_rhs = Benders_cut_pi
        
                self.cut_temp.append(temp_constraint)
                self.cut_rhs_temp.append(temp_rhs)

                temp_cut_itr = 1
                # print("stage:",self.stage,"state:",self.state,"SDDP Cut")
            else:
                temp_cut_itr = 0
        else:
            temp_constraint = self.model.addConstr(self.theta[state_next] >= quicksum(pi_b[w]*self.u[w] for w in range(self.args.W)) 
                                                                + quicksum(pi_c[w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                + Benders_cut_pi)
            temp_rhs = Benders_cut_pi
        
            self.cut_temp.append(temp_constraint)
            self.cut_rhs_temp.append(temp_rhs)

            # print("stage:",self.stage,"state:",self.state,"SDDP Cut")

        return temp_cut_itr

    def add_Benders_cut_shraing(self, pi_8b, pi_8e, pi_8g, pi_8h, pi_8i):

        if(self.stage0 == False):
            temp_constraint = self.model.addConstr(self.phi >= (1/self.args.K)*quicksum(quicksum(self.v[w,p]*pi_8b[k][w][p] for w in range(self.args.W) for p in range(self.args.P))
                                                                              +quicksum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I))
                                                                              +quicksum(pi_8g[k][m][w]*self.u[w] for m in range(self.args.M+1) for w in range(self.args.W))
                                                                              +quicksum(pi_8h[k][m][j][g]*self.idata.demand[self.stage][self.state][k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                                                                              +quicksum(pi_8i[k][w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                              for k in range(self.args.K)))

            temp_rhs = (1/self.args.K)*sum(pi_8h[k][m][j][g]*self.idata.demand[self.stage][self.state][k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G) for k in range(self.args.K))
            temp_rhs = temp_rhs + (1/self.args.K)*sum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I) for k in range(self.args.K))

            self.cut.append(temp_constraint)
            self.cut_rhs.append(temp_rhs)
        
        else:
            temp_constraint = self.model.addConstr(self.phi >= (1/self.args.K)*quicksum(quicksum(self.v[w,p]*pi_8b[k][w][p] for w in range(self.args.W) for p in range(self.args.P))
                                                                              +quicksum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I))
                                                                              +quicksum(pi_8g[k][m][w]*self.u[w] for m in range(self.args.M+1) for w in range(self.args.W))
                                                                              +quicksum(pi_8h[k][m][j][g]*self.idata.demand_root[k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                                                                              +quicksum(pi_8i[k][w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                              for k in range(self.args.K)))

            temp_rhs = (1/self.args.K)*sum(pi_8h[k][m][j][g]*self.idata.demand_root[k][m][j][g] for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G) for k in range(self.args.K))
            temp_rhs = temp_rhs + (1/self.args.K)*sum(self.idata.B_i[i]*pi_8e[k][m][i] for m in range(self.args.M+1) for i in range(self.args.I) for k in range(self.args.K))
       
            self.cut.append(temp_constraint)
            self.cut_rhs.append(temp_rhs)

class StageProblem_extended:
    """A data structure that keeps stage-wise problems"""
    
    def __init__(self, args, input_data, state, stage, last_stage=False, stage0=False):

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
        
        self.u_value = np.zeros((args.W))
        self.y_value = np.zeros((args.W))
        self.v_value = np.zeros((args.W, args.P))
        self.x_value = np.zeros((args.W, args.P))
        self.z_value = np.zeros((args.W, args.P))

 
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
                                  + quicksum(self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) + self.idata.H_p[p]*self.v[w,p] for w in range(args.W) for p in range(args.P))
                                  + (1/args.K)*quicksum(quicksum(self.idata.O_p[p]*self.aak[k,w,p] - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[k,w,p] for w in range(args.W) for p in range(args.P)) 
                                                      + quicksum( quicksum(self.idata.O_p[p]*self.ak[k,m,i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                                                + quicksum(self.idata.CU_g[g]*self.sk[k,m,j,g] for j in range(args.J) for g in range(args.G)) for m in range(args.M+1)) for k in range(args.K))
                                  + quicksum(self.idata.MC_tran_matrix[state][n]*self.theta[n] for n in range(args.N)) 
                                , GRB.MINIMIZE);
        else:
            self.model.setObjective(quicksum(self.idata.E_w[w]*self.y[w] for w in range(args.W)) 
                                  + quicksum(self.idata.O_p[p]*(self.x[w,p] - self.idata.R_p[p]*self.z[w,p]) + self.idata.H_p[p]*self.v[w,p] for w in range(args.W) for p in range(args.P))
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


        if(args.Policy == "WS"):
            for w in range(args.W):
                self.model.addConstr(self.u[w] == 0)
        elif(args.Policy == "avg"):
            if stage0 != True:
                for w in range(args.W):
                    for p in range(args.P):
                        self.model.addConstr(self.y[w] == 0)
                        self.model.addConstr(self.x[w,p] == 0 )
                        self.model.addConstr(self.z[w,p] == 0)


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
                    self.model.addConstr(self.bk[k,m-1,i] + quicksum(self.ak[k,m,i,w,p] for p in range(args.P) for w in range(args.W) if m+self.idata.P_p[p] <= args.M) ==  self.bk[k,m,i] + quicksum(self.ak[k,m-self.idata.P_p[p],i,w,p] for p in range(args.P) for w in range(args.W) if m-self.idata.P_p[p] >= 0))

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
                        if(m-self.idata.P_p[p] >= 0):
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
                            self.k_demand[k][m][j][g] = self.model.addConstr(quicksum(self.fk[k,m,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[k,m,j,g] == self.idata.demand[stage][state][k][m][j][g])
                        else:
                            self.k_demand[k][m][j][g] = self.model.addConstr(quicksum(self.fk[k,m,w,j,p,g] for w in range(args.W) for p in range(args.P)) + self.sk[k,m,j,g] == self.idata.demand_root[k][m][j][g])


        # Assumption Replensih by MHS
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    self.model.addConstr(self.vk[k,args.M,w,p] + self.aak[k,w,p] - self.bbk[k,w,p] == self.v[w,p])


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
        
        for w in range(self.args.W):
            self.u_value[w] = self.u[w].x
            self.y_value[w] = self.y[w].x
            for p in range(self.args.P):
                self.v_value[w][p] = self.v[w,p].x
                self.x_value[w][p] = self.x[w,p].x
                self.z_value[w][p] = self.z[w,p].x

        if(self.args.Cost_print == True):
            print("Extend Strategic Node Cost:", self.model.ObjVal)
            print("Extend Second-stage Cost:", (1/self.args.K)*sum(quicksum(self.idata.O_p[p]*self.aak[k,w,p].x - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[k,w,p].x for w in range(self.args.W) for p in range(self.args.P)) 
                                                          + quicksum( quicksum(self.idata.O_p[p]*self.ak[k,m,i,w,p].x for i in range(self.args.I) for w in range(self.args.W) for p in range(self.args.P))
                                                                    + quicksum(self.idata.CU_g[g]*self.sk[k,m,j,g].x for j in range(self.args.J) for g in range(self.args.G)) for m in range(self.args.M+1)) for k in range(self.args.K)))
            for k in range(self.args.K):
                print(f"Extend sub {k} Cost:", (sum(self.idata.O_p[p]*self.aak[k,w,p].x - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[k,w,p].x for w in range(self.args.W) for p in range(self.args.P)) 
                                                        + sum( quicksum(self.idata.O_p[p]*self.ak[k,m,i,w,p].x for i in range(self.args.I) for w in range(self.args.W) for p in range(self.args.P))
                                                        + sum(self.idata.CU_g[g]*self.sk[k,m,j,g].x for j in range(self.args.J) for g in range(self.args.G)) for m in range(self.args.M+1))))



        staging_area_expand_cost = 0
        inventory_expand_cost = 0
        replenmship_cost = 0
        Shortage_cost = 0
        acquire_cost = 0
        holding_cost_temp = 0

        if(self.args.evaluate_switch == True):

            
            staging_area_expand_cost = sum(self.idata.E_w[w]*self.y[w].x for w in range(self.args.W))                                   
            inventory_expand_cost =  sum(self.idata.O_p[p]*(self.x[w,p].x - self.idata.R_p[p]*self.z[w,p].x) for w in range(self.args.W) for p in range(self.args.P))
            replenmship_cost = (1/self.args.K)*sum(sum(self.idata.O_p[p]*self.aak[k,w,p].x - self.idata.R_p[p]*self.idata.O_p[p]*self.bbk[k,w,p].x for w in range(self.args.W) for p in range(self.args.P)) for k in range(self.args.K))
            Shortage_cost =  (1/self.args.K)*sum(sum(sum(self.idata.CU_g[g]*self.sk[k,m,j,g].x for j in range(self.args.J) for g in range(self.args.G)) for m in range(self.args.M+1)) for k in range(self.args.K))
            acquire_cost =  (1/self.args.K)*sum(sum( sum(self.idata.O_p[p]*self.ak[k,m,i,w,p].x for i in range(self.args.I) for w in range(self.args.W) for p in range(self.args.P)) for m in range(self.args.M+1)) for k in range(self.args.K))
            holding_cost_temp = sum(self.idata.H_p[p]*self.v[w,p].x for w in range(self.args.W) for p in range(self.args.P))


        return self.u,self.v,self.model.ObjVal,staging_area_expand_cost,inventory_expand_cost,replenmship_cost,Shortage_cost,acquire_cost,holding_cost_temp

    def backward_run(self):


        pi_b = np.zeros((self.args.W))
        pi_c = np.zeros((self.args.W, self.args.P))
        pi_e = np.zeros((self.args.K, self.args.M+1, self.args.I))
        pi_h = np.zeros((self.args.K, self.args.M+1,self.args.J,self.args.G))

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

        # Production Capacity E_i
        for k in range(self.args.K):
            for m in range(self.args.M+1):
                for i in range(self.args.I):
                     pi_e[k][m][i] = self.h_pro_cap[k][m][i].pi
                     temp = temp + self.h_pro_cap[k][m][i].pi*self.idata.B_i[i]


        # Satify Demand Flow
        for k in range(self.args.K):
            for m in range(1,self.args.M+1):
                for j in range(self.args.J):
                    for g in range(self.args.G):
                        pi_h[k][m][j][g] = self.k_demand[k][m][j][g].pi
                        if(self.stage0 == False):
                            temp = temp + self.k_demand[k][m][j][g].pi*self.idata.demand[self.stage][self.state][k][m][j][g]
                        else:
                            temp = temp + self.k_demand[k][m][j][g].pi*self.idata.demand_root[k][m][j][g]

        
        cut_pi = 0

        # Cut
        if(self.cut):
            for c in range(len(self.cut_rhs)):
                temp = temp + self.cut[c].pi*self.cut_rhs[c]
                cut_pi = cut_pi + self.cut[c].pi*self.cut_rhs[c] 



        if(abs(temp-self.model.ObjVal) >= 1e-3):
            print("stage:",self.stage,"problematic dual solution!")
            print("temp:",temp)
            print("obj:",self.model.ObjVal)
            pdb.set_trace()

        return pi_b,pi_c,pi_e,pi_h,cut_pi,self.model.ObjVal

    def add_cut(self,obj,stage_next,state_sample_path,state_next,pi_b,pi_c,pi_e,pi_h,cut_pi):

        temp_cut_itr = 0

        if(self.state == state_sample_path or self.stage0==True):

            if(self.theta[state_next].x < obj - cut_vio_thred and abs(self.theta[state_next].x - obj)/max(abs(self.theta[state_next].x),1e-10) > cut_vio_thred):

                temp_constraint = self.model.addConstr(self.theta[state_next] >= quicksum(pi_b[w]*self.u[w] for w in range(self.args.W)) 
                                                                                + quicksum(pi_c[w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                                + quicksum(pi_e[k][m][i]*self.idata.B_i[i] for k in range(self.args.K) for m in range(self.args.M+1) for i in range(self.args.I))
                                                                                + quicksum(pi_h[k][m][j][g]*self.idata.demand[stage_next][state_next][k][m][j][g] for k in range(self.args.K) for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                                                                                + cut_pi)
                
                temp_rhs = sum(pi_e[k][m][i]*self.idata.B_i[i] for k in range(self.args.K) for m in range(self.args.M+1) for i in range(self.args.I)) 
                temp_rhs = temp_rhs + sum(pi_h[k][m][j][g]*self.idata.demand[stage_next][state_next][k][m][j][g] for k in range(self.args.K) for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                temp_rhs = temp_rhs + cut_pi

                self.cut_temp.append(temp_constraint)
                self.cut_rhs_temp.append(temp_rhs)
                temp_cut_itr = 1
            else:
                temp_cut_itr = 0
        else:
            temp_constraint = self.model.addConstr(self.theta[state_next] >= quicksum(pi_b[w]*self.u[w] for w in range(self.args.W)) 
                                                                + quicksum(pi_c[w][p]*self.v[w,p] for w in range(self.args.W) for p in range(self.args.P))
                                                                + quicksum(pi_e[k][m][i]*self.idata.B_i[i] for k in range(self.args.K) for m in range(self.args.M+1) for i in range(self.args.I))
                                                                + quicksum(pi_h[k][m][j][g]*self.idata.demand[stage_next][state_next][k][m][j][g] for k in range(self.args.K) for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
                                                                + cut_pi)
                
            temp_rhs = sum(pi_e[k][m][i]*self.idata.B_i[i] for k in range(self.args.K) for m in range(self.args.M+1) for i in range(self.args.I))
            temp_rhs = temp_rhs + sum(pi_h[k][m][j][g]*self.idata.demand[stage_next][state_next][k][m][j][g] for k in range(self.args.K) for m in range(1,self.args.M+1) for j in range(self.args.J) for g in range(self.args.G))
            temp_rhs = temp_rhs + cut_pi
            
            self.cut_temp.append(temp_constraint)
            self.cut_rhs_temp.append(temp_rhs)

        return temp_cut_itr


class solve_SDDP:
    def __init__(self, args, input_data):

        self.args = args
        self.idata = input_data

        if(args.Strategic_node_sovling == 0):
            self.stage_root = StageProblem_extended(args,input_data,args.initial_state,0,stage0=True)
            self.stage = [[StageProblem_extended(args,input_data,n,t,stage0=False) for n in range(args.N)] for t in range(args.T-1)] 
            self.stage_leaf = [StageProblem_extended(args,input_data,n,args.T-1,last_stage=True) for n in range(args.N)];
        elif(args.Strategic_node_sovling == 1):
            self.stage_root = StageProblem_Decomposition(args,input_data,args.initial_state,0,stage0=True)
            self.stage = [[StageProblem_Decomposition(args,input_data,n,t,stage0=False) for n in range(args.N)] for t in range(args.T-1)] 
            self.stage_leaf = [StageProblem_Decomposition(args,input_data,n,args.T-1,last_stage=True) for n in range(args.N)];

    def sample_path(self, args):

        path = []
        
        self.initial_state = args.initial_state
        state = self.initial_state

        for stage in range(args.T):
            next_state = np.random.choice(args.N, 1, self.idata.MC_tran_matrix[state].tolist)
            state = next_state[0]
            path.append(state)

        # print(path)
        return path


    def termination_check(self, iter, relative_gap, LB, start, cutviol_iter):
        flag = 0
        Elapsed = time.time() - start
        if(iter > self.args.MAX_ITER):
            flag = 1
            print("max iteration is reached")
        elif (Elapsed > self.args.time_limit):
            flag = 2
            print("time limit is reached")
        elif (cutviol_iter > self.args.CUTVIOL_MAXITER):
            flag = 3
            print("cut violation is reached")
        else:
            if iter > self.args.STALL:
                relative_gap = (LB[iter-1]-LB[iter-1-self.args.STALL])/max(1e-10,abs(LB[iter-1-self.args.STALL]))
                if relative_gap < self.args.LB_TOL:
                    flag = 4
                    print("the LB is not making significant progress")
        return flag, Elapsed

    def Benders_cut_shraing(self,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,root=False,leaf=False,state=None,stage=None):

        if(root == True):
            for stage_add_Benders_cut in range(self.args.T-1):
                for state_add_Benders_cut in range(self.args.N):
                    self.stage[stage_add_Benders_cut][state_add_Benders_cut].add_Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i)

            for state_add_Benders_cut in range(self.args.N):
                self.stage_leaf[state_add_Benders_cut].add_Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i)

        elif(leaf == True):
            # self.stage_root.add_Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i)

            # for stage_add_Benders_cut in range(self.args.T-1):
            #     for state_add_Benders_cut in range(self.args.N):
            #         self.stage[stage_add_Benders_cut][state_add_Benders_cut].add_Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i)

            for state_add_Benders_cut in range(self.args.N):
                if(state != state_add_Benders_cut):
                    self.stage_leaf[state_add_Benders_cut].add_Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i)
        
        else:
            # self.stage_root.add_Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i)

            for stage_add_Benders_cut in range(self.args.T-1):
                for state_add_Benders_cut in range(self.args.N):
                    if(state_add_Benders_cut != state and stage_add_Benders_cut>= stage):
                        # print("stage/state:",stage_add_Benders_cut,state_add_Benders_cut)
                        self.stage[stage_add_Benders_cut][state_add_Benders_cut].add_Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i)

            for state_add_Benders_cut in range(self.args.N):
                self.stage_leaf[state_add_Benders_cut].add_Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i)


    def run(self):

        start = time.time()
        LB_list = []
        relative_gap = 1e10
        cutviol_iter = 0

        for iter in range(self.args.MAX_ITER):

            # sample path
            sample_path = self.sample_path(self.args)
            # print(sample_path)

            u = 0
            v = 0
            obj_ex = 0

            # ---------------------------------------------------- Forward ----------------------------------------------------
            if(self.args.Strategic_node_sovling == 0):
                u,v,obj_ex,temp1,temp2,temp3,temp4,temp5,temp6 = self.stage_root.forward_run()
            elif(self.args.Strategic_node_sovling == 1):
                u,v,obj_ex,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,temp1,temp2,temp3 = self.stage_root.forward_run()

                # Benders Cut Sharing
                self.Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,root=True)

            
            for stage in range(self.args.T-1):
                if(self.args.Strategic_node_sovling == 0):
                    u,v,obj_ex,temp1,temp2,temp3,temp4,temp5,temp6 = self.stage[stage][sample_path[stage]].forward_run(u,v)
                elif(self.args.Strategic_node_sovling == 1):
                    u,v,obj_ex,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,temp1,temp2,temp3 = self.stage[stage][sample_path[stage]].forward_run(u,v)
                    
                    # Benders Cut Sharing
                    self.Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,state=sample_path[stage],stage=stage)
            
            if(self.args.Strategic_node_sovling == 0):
                u,v,obj_ex,temp1,temp2,temp3,temp4,temp5,temp6 = self.stage_leaf[sample_path[self.args.T-1]].forward_run(u,v)
            elif(self.args.Strategic_node_sovling == 1):
                u,v,obj_ex,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,temp1,temp2,temp3 = self.stage_leaf[sample_path[self.args.T-1]].forward_run(u,v)
                
                # Benders Cut Sharing
                self.Benders_cut_shraing(pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,leaf=True,state=sample_path[self.args.T-1])

            


            # ----------------------------------- Backward -----------------------------------
            if(self.args.Strategic_node_sovling == 0):
                pi_b,pi_c,pi_e,pi_h,cut_pi,LB = self.stage_leaf[sample_path[self.args.T-1]].backward_run()
            elif(self.args.Strategic_node_sovling == 1):
                pi_b,pi_c,Benders_cut_pi,LB = self.stage_leaf[sample_path[self.args.T-1]].backward_run(iter)
            
            for stage in reversed(range(self.args.T-1)):

                # ---------------------------- Cut Sharing ----------------------------
                for state in range(self.args.N):
                    if(self.args.Strategic_node_sovling == 0):
                        cut_iter_temp = self.stage[stage][state].add_cut(LB,stage+1,sample_path[stage],sample_path[stage+1],pi_b,pi_c,pi_e,pi_h,cut_pi)
                    elif(self.args.Strategic_node_sovling == 1):
                        cut_iter_temp = self.stage[stage][state].add_cut(LB,stage+1,sample_path[stage],sample_path[stage+1],pi_b,pi_c,Benders_cut_pi)
                    
                    cutviol_iter = cutviol_iter + cut_iter_temp

                if(self.args.Strategic_node_sovling == 0):
                    pi_b,pi_c,pi_e,pi_h,cut_pi,LB =  self.stage[stage][sample_path[stage]].backward_run()
                elif(self.args.Strategic_node_sovling == 1):
                    pi_b,pi_c,Benders_cut_pi,LB =  self.stage[stage][sample_path[stage]].backward_run(iter)

            if(self.args.Strategic_node_sovling == 0):
                cut_iter_temp = self.stage_root.add_cut(LB,0,0,sample_path[0],pi_b,pi_c,pi_e,pi_h,cut_pi)
            elif(self.args.Strategic_node_sovling == 1):
                cut_iter_temp = self.stage_root.add_cut(LB,0,0,sample_path[0],pi_b,pi_c,Benders_cut_pi)

            cutviol_iter = cutviol_iter + cut_iter_temp
            if(self.args.Strategic_node_sovling == 0):
                pi_b,pi_c,pi_e,pi_h,cut_pi,LB =  self.stage_root.backward_run()
            elif(self.args.Strategic_node_sovling == 1):
                pi_b,pi_c,LB,Benders_cut_pi =  self.stage_root.backward_run(iter)

            LB_list.append(LB)
            if(iter%50 == 0):
                print("iteration:", iter, "LB:", LB)

            
            # ----------------------------------- Stop Criteria -----------------------------------
            flag, Elapsed = self.termination_check(iter, relative_gap, LB_list, start, cutviol_iter)
            if flag != 0:
                train_time = Elapsed
                print("total time:", Elapsed)

                self.args.evaluate_switch = True

                # ---------------------------------------------------- Polciy Simulation ----------------------------------------------------
                
                simulate_iter = 1000
                solution_u = np.zeros((self.args.T+1,self.args.N))
                solution_v = np.zeros((self.args.T+1,self.args.N))
                solution_obj = np.zeros((self.args.T+1,self.args.N))
                staging_area_expand_cost = np.zeros((self.args.T+1,self.args.N))
                inventory_expand_cost = np.zeros((self.args.T+1,self.args.N))
                replenmship_cost = np.zeros((self.args.T+1,self.args.N))
                acquire_cost = np.zeros((self.args.T+1,self.args.N))
                Shortage_cost = np.zeros((self.args.T+1,self.args.N))
                holding_cost = np.zeros((self.args.T+1,self.args.N))

                path_count = np.zeros((self.args.T+1,self.args.N))


                for iter in range(simulate_iter):

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
                            pi_8b,pi_8e,pi_8g,pi_8h,pi_8i,sub_opt,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp = self.stage_root.sub.run(u,v,self.idata.demand_root[k])

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

                    
                    for stage in range(self.args.T-1):
                        if(self.args.Strategic_node_sovling == 0):
                            u,v,obj_ex,staging_area_expand_cost_temp,inventory_expand_cost_temp,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp,holdings_cost_temp = self.stage[stage][sample_path[stage]].forward_run(u,v)
                        elif(self.args.Strategic_node_sovling == 1):
                            u,v,obj_ex,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,staging_area_expand_cost_temp,inventory_expand_cost_temp,holdings_cost_temp = self.stage[stage][sample_path[stage]].forward_run(u,v)
                            for k in range(self.args.K):
                                pi_8b,pi_8e,pi_8g,pi_8h,pi_8i,sub_opt,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp = self.stage[stage][sample_path[stage]].sub.run(u,v,self.idata.demand[stage][sample_path[stage]][k])
                        
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
                    
                    if(self.args.Strategic_node_sovling == 0):
                        u,v,obj_ex,staging_area_expand_cost_temp,inventory_expand_cost_temp,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp,holdings_cost_temp = self.stage_leaf[sample_path[self.args.T-1]].forward_run(u,v)
                    elif(self.args.Strategic_node_sovling == 1):
                        u,v,obj_ex,pi_8b, pi_8e, pi_8g, pi_8h, pi_8i,staging_area_expand_cost_temp,inventory_expand_cost_temp,holdings_cost_temp = self.stage_leaf[sample_path[self.args.T-1]].forward_run(u,v)
                        for k in range(self.args.K):
                                pi_8b,pi_8e,pi_8g,pi_8h,pi_8i,sub_opt,replenmship_cost_temp,Shortage_cost_temp,acquire_cost_temp = self.stage_leaf[sample_path[self.args.T-1]].sub.run(u,v,self.idata.demand[stage][sample_path[stage]][k])
                        
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

                
                solution = []

                # pdb.set_trace()
                for t in range(self.args.T+1):
                    for n in range(self.args.N):
                        if(path_count[t][n] == 0):
                            solution.append([t,n,solution_u[t][n],solution_v[t][n],solution_obj[t][n],staging_area_expand_cost[t][n],inventory_expand_cost[t][n],replenmship_cost[t][n],Shortage_cost[t][n],acquire_cost[t][n],holding_cost[t][n]])
                        else:
                            solution.append([t,n,solution_u[t][n]/path_count[t][n],solution_v[t][n]/path_count[t][n],solution_obj[t][n]/path_count[t][n],staging_area_expand_cost[t][n]/path_count[t][n],inventory_expand_cost[t][n]/path_count[t][n],replenmship_cost[t][n]/path_count[t][n],Shortage_cost[t][n]/path_count[t][n],acquire_cost[t][n]/path_count[t][n],holding_cost[t][n]/path_count[t][n]])
                
                
                df = pd.DataFrame(solution, columns=[ 'stage','state','Staging Area Capacity','Inventory Level','obj','staging_area_expand_cost','inventory_expand_cost','replenmship_cost','Shortage_cost','acquire_cost','holding_cost'])
                filename = "result_Stage_" + str(self.args.T) + "_States_" + str(self.args.N) + "_Study_" + str(self.args.J) + "_month_" + str(self.args.M)  + "_K_" + str(self.args.K)  + "_Pp_" + str(self.args.P_p_factor) + "_Cu_" + str(self.args.C_u_factor) + "_Op_" +  str(self.args.O_p_factor) + "_Hp_" +  str(self.args.H_p_factor)  + "_policy_" +  str(self.args.Policy)

                df.to_csv(f'{filename}.csv', index=False) 
                break


            




