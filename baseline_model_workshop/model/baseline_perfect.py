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
        self.diff_x = self.model.addVars(args.P, args.P, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_x = self.model.addVars(args.P, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')
        
        # Second-stage
        self.v = self.model.addVars(args.W, args.P, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='vipt')
        self.f = self.model.addVars(args.W, args.J, args.P, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='fwjpgt')
        self.q = self.model.addVars(args.J, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')
        self.s = self.model.addVars(args.I, args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='siwpt')
        self.r = self.model.addVars(args.W, args.P, lb=0.0, name='rwp')
        self.diff = self.model.addVars(args.J, args.G, args.G, args.T, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs = self.model.addVars(args.J, args.G, args.G, args.T, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        # pdb.set_trace()

        # Objective
        self.model.setObjective((quicksum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
                                +quicksum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t] for g in range(args.G) for t in range(args.T) for j in range(args.J))
                                +quicksum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t] for w in range(args.W) for p in range(args.P) for t in range(args.T))
                                +quicksum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                -quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))) 
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

        for g1 in range(args.G):
            for g2 in range(args.G):
                for t in range(1,args.T):
                    self.model.addGenConstrAbs(self.abs[j,g1,g2,t], self.diff[j,g1,g2,t])
                    self.model.addConstr(self.abs[j,g1,g2,t] <= args.fair)

    


    def run(self,args):

        
        inventory_level = np.zeros((args.W,args.P,args.K))
        Used_time = np.zeros((args.T,args.K))
        value_group = np.zeros((args.G,args.K))

        operation_cost_total = 0
        holding_cost_total = 0
        unmet_cost_total = 0
        replenish_cost_total = 0
        group_value_cost = 0

        for k in range(args.K):

            print("-----------")
            print("Scen:", k)
            print("-----------")

            demand_cons = []
            fair_cons = []


            # Demand
            for g in range(args.G):
                for t in range(1,args.T):
                    for j in range(args.J):
                        temp = self.model.addConstr(quicksum(self.f[w,j,p,g,t] for w in range(args.W) for p in range(args.P)) + self.q[j,g,t] == self.idata.demand[k][j][g][t])
                        demand_cons.append(temp)

            if(args.fair_sw == 1):
            # Fair Constraint (Shoratge)
                for g1 in range(args.G):
                    for g2 in range(args.G):
                        for t in range(1,args.T):
                            temp = self.model.addConstr(self.q[j,g1,t]*self.idata.demand[k][j][g2][t] - self.q[j,g2,t]*self.idata.demand[k][j][g1][t] == self.diff[j,g1,g2,t]*self.idata.demand[k][j][g1][t]*self.idata.demand[k][j][g2][t])
                            fair_cons.append(temp)


            self.model.update()
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                print(self.model.ObjVal)
            else:
                print("Infeasible or unbounded!")

            for w in range(args.W):
                for p in range(args.P):
                    inventory_level[w][p][k] = self.x[w,p].x

            

            for t in range(args.T):
                for k in range(args.K):
                    Used_time[t][k] = sum(self.v[w,p,t].x for w in range(args.W) for p in range(args.P))



        
            operation_cost_total = operation_cost_total + sum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t].x*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) )
            holding_cost_total = operation_cost_total + sum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t].x for w in range(args.W) for p in range(args.P) for t in range(args.T))
            unmet_cost_total = unmet_cost_total + sum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t].x for g in range(args.G) for t in range(args.T) for j in range(args.J))
            replenish_cost_total = replenish_cost_total + sum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p].x for i in range(args.I) for w in range(args.W) for p in range(args.P))
            group_value_cost = group_value_cost + sum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t].x*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))

        
            for g in range(args.G):
                value_group[g][k] = sum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t].x*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(args.T) for a in range(args.A))
            
            for constraint in demand_cons:
                self.model.remove(constraint)

            for constraint in fair_cons:
                self.model.remove(constraint)

        df_name = ["OPT","Operation_Cost","Holding_Cost","Unmet_Cost","Replenish_Cost","Victim_value"]
        data = [[self.model.ObjVal,operation_cost_total,holding_cost_total,unmet_cost_total,replenish_cost_total,group_value_cost]]
        df = pd.DataFrame(data, columns=[df_name])
        df.to_csv("Cost_structure.csv")

        # df = pd.DataFrame(inventory_level)
        # df.to_csv("Inventory_Policy.csv")

        df = pd.DataFrame(Used_time)
        df.to_csv("Used_time.csv")
     

