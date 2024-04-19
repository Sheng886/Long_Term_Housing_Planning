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

        # First-stage
        self.x = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        self.diff_x = self.model.addVars(args.P, args.P, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_x = self.model.addVars(args.P, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        self.xk = self.model.addVars(args.W, args.P, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        self.xkk = self.model.addVars(args.W, args.P, lb=0.0, vtype=GRB.CONTINUOUS, name='xwp')
        
        # Second-stage
        self.v = self.model.addVars(args.W, args.P, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='vipt')
        self.f = self.model.addVars(args.W, args.J, args.P, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='fwjpgt')
        self.q = self.model.addVars(args.J, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')
        self.s = self.model.addVars(args.I, args.W, args.P, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='siwpt')
        self.r = self.model.addVars(args.W, args.P, args.K, lb=0.0, name='rwp')

        self.f_p1 = self.model.addVars(args.J, args.P1, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='fjpgt')

        self.diff_group = self.model.addVars(args.G, args.G, args.T, args.K, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_group = self.model.addVars(args.G, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        self.diff_region = self.model.addVars(args.J, args.J, args.T, args.K, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_region = self.model.addVars(args.J, args.J, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        self.diff_value = self.model.addVars(args.G, args.G, args.T, args.K, vtype=GRB.CONTINUOUS, name='qgt')
        self.abs_value = self.model.addVars(args.G, args.G, args.T, args.K, lb=0.0, vtype=GRB.CONTINUOUS, name='qgt')

        # pdb.set_trace()

        # Objective
        self.model.setObjective((1/(args.K))*quicksum((quicksum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t,k]*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
                                +quicksum(self.idata.O_p1[p]*self.f_p1[j,p,g,t,k] for j in range(args.J) for p in range(args.P1) for g in range(args.G) for t in range(args.T))
                                +quicksum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t,k] for g in range(args.G) for t in range(args.T) for j in range(args.J))
                                +quicksum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t,k] for w in range(args.W) for p in range(args.P) for t in range(args.T))
                                +quicksum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p,k] for i in range(args.I) for w in range(args.W) for p in range(args.P))
                                +args.dprate*quicksum(self.idata.O_p[p]*self.x[w,p] for w in range(args.W) for p in range(args.P))
                                -quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))) 
                                -quicksum(self.idata.A_H_flood_p1[a][p]*self.idata.Hd_weight[a][g]*self.f_p1[j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P1) for g in range(args.G) for t in range(args.T) for a in range(args.A)) 
                                for k in range(args.K)), GRB.MINIMIZE);

        
        # Policy
        if(args.model == "perfect"):
            for w in range(args.W):
                for k in range(args.K):
                    self.model.addConstr(quicksum(self.idata.u_p[p]*self.xk[w,p,k] for p in range(args.P)) <= self.idata.Cap_w[w])
            
        else:
            for w in range(args.W):
                self.model.addConstr(quicksum(self.idata.u_p[p]*self.x[w,p] for p in range(args.P)) <= self.idata.Cap_w[w])
           

        # Initail Inventory
        for k in range(args.K):
            for w in range(args.W):
                for p in range(args.P):
                    if(args.model == "perfect"):
                        self.model.addConstr(self.v[w,p,0,k] == self.xk[w,p,k])
                    else:
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
                    if(args.model == "perfect"):
                        if(k+1 == args.K):
                            self.model.addConstr(self.v[w,p,args.T-1,k] + quicksum(self.s[i,w,p,k] for i in range(args.I))  == self.xk[w,p,0])
                        else:
                            self.model.addConstr(self.v[w,p,args.T-1,k] + quicksum(self.s[i,w,p,k] for i in range(args.I))  == self.xk[w,p,k+1])
                    else:
                        self.model.addConstr(self.v[w,p,args.T-1,k] + quicksum(self.s[i,w,p,k] for i in range(args.I))  == self.x[w,p])

        # Recycle Inventory
        for k in range(args.K):
            for p in range(args.P):
                if(args.model == "perfect"):
                    self.model.addConstr(quicksum(self.r[w,p,k] for w in range(args.W)) == self.idata.R_p[p]*(quicksum(self.xk[w,p,k]-self.v[w,p,args.T-1,k] for w in range(args.W))))
                else:
                    self.model.addConstr(quicksum(self.r[w,p,k] for w in range(args.W)) == self.idata.R_p[p]*(quicksum(self.x[w,p]-self.v[w,p,args.T-1,k] for w in range(args.W))))

        
        if(args.model == "avg"):
            self.avg_demand = np.zeros((args.J,args.G,args.T))
            for j in range(args.J):
                for g in range(args.G):
                    for t in range(args.T):
                        for k in range(args.K):
                            self.avg_demand[j][g][t] = self.avg_demand[j][g][t] + self.idata.demand[k][j][g][t]

            self.avg_demand = self.avg_demand/args.K

            for j in range(args.J):
                for g in range(args.G):
                    for t in range(args.T):
                        for k in range(args.K):
                            self.idata.demand[k][j][g][t] = self.avg_demand[j][g][t]





        # Demand
        for k in range(args.K):
            for g in range(args.G):
                for t in range(1,args.T):
                    for j in range(args.J):
                        self.model.addConstr(quicksum(self.f[w,j,p,g,t,k] for w in range(args.W) for p in range(args.P)) + quicksum(self.f_p1[j,p,g,t,k] for p in range(args.P1)) + self.q[j,g,t,k] == self.idata.demand[k][j][g][t])

        for k in range(args.K):
            for p in range(args.P1):
                for t in range(args.T):
                    self.model.addConstr(quicksum(self.f_p1[j,p,g,t,k] for j in range(args.J) for g in range(args.G)) <= self.idata.S_p1[p])
        


        # Fair Constraint Group (Shoratge)
        if(args.fair_sw_group == 1):
            for k in range(args.K):
                for g1 in range(args.G):
                    for g2 in range(args.G):
                        for t in range(1,args.T):
                            if(sum(self.idata.demand[k][j][g1][t] for j in range(args.J)) >= 10e-5 and sum(self.idata.demand[k][j][g2][t] for j in range(args.J)) >= 10e-5):
                                self.model.addConstr(quicksum(self.q[j,g1,t,k] for j in range(args.J))/sum(self.idata.demand[k][j][g1][t] for j in range(args.J)) - quicksum(self.q[j,g2,t,k] for j in range(args.J))/sum(self.idata.demand[k][j][g2][t] for j in range(args.J)) == self.diff_group[g1,g2,t,k])
                                self.model.addConstr(self.abs_group[g1,g2,t,k] == gp.abs_(self.diff_group[g1,g2,t,k]))
                                self.model.addConstr(self.abs_group[g1,g2,t,k] <= args.fair)


        # Fair Constraint region (Shoratge)
        if(args.fair_sw_region == 1):
            for k in range(args.K):
                for j1 in range(args.J):
                    for j2 in range(args.J):
                        for t in range(1,args.T):
                            if(sum(self.idata.demand[k][j1][g][t] for g in range(args.G)) >= 10e-5 and sum(self.idata.demand[k][j2][g][t] for g in range(args.G)) >= 10e-5):
                                self.model.addConstr(quicksum(self.q[j1,g,t,k] for g in range(args.G))/sum(self.idata.demand[k][j1][g][t] for g in range(args.G)) - quicksum(self.q[j2,g,t,k] for g in range(args.G))/sum(self.idata.demand[k][j2][g][t] for g in range(args.G)) == self.diff_region[j1,j2,t,k])
                                self.model.addConstr(self.abs_region[j1,j2,t,k] == gp.abs_(self.diff_region[j1,j2,t,k]))
                                self.model.addConstr(self.abs_region[j1,j2,t,k] <= args.fair)

        
        


        # Fair Constraint group (Value) 
        if(args.fair_sw_value == 1):
            for k in range(args.K):
                for g1 in range(args.G):
                    for g2 in range(args.G):
                        for t in range(1,args.T):
                            G1 = quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g1]*self.f[w,j,p,g1,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for a in range(args.A)) + quicksum(self.idata.A_H_flood_p1[a][p]*self.idata.Hd_weight[a][g1]*self.f_p1[j,p,g1,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P1)for a in range(args.A)) 
                            
                            G2 = quicksum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g2]*self.f[w,j,p,g2,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for a in range(args.A)) + quicksum(self.idata.A_H_flood_p1[a][p]*self.idata.Hd_weight[a][g2]*self.f_p1[j,p,g2,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P1)for a in range(args.A)) 

                            self.model.addConstr(G1/self.idata.max_value_g[g1] - G2/self.idata.max_value_g[g2] == self.diff_value[g1,g2,t,k])
                            self.model.addConstr(self.abs_value[g1,g2,t,k] == gp.abs_(self.diff_value[g1,g2,t,k]))
                            self.model.addConstr(self.abs_value[g1,g2,t,k] <= args.fair)


    def run(self,args):

        os_path = "result_temp/{model}_{k}_FR{a}_FG{b}_FV{c}_SF{d}".format(model = args.model, k = str(args.K), a=str(args.fair_sw_group),b=str(args.fair_sw_region),c=str(args.fair_sw_value),d=str(int(args.s_factor*10)))
        os.mkdir(os_path) 

        self.model.update()
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print(self.model.ObjVal)
        else:
            print("Infeasible or unbounded!")

        if(args.model != "perfect"):
            inventory_level = np.zeros((args.W,args.P))
            for w in range(args.W):
                for p in range(args.P):
                    inventory_level[w][p] = self.x[w,p].x

        Used_time = np.zeros((args.T,args.K))

        for t in range(args.T):
            for k in range(args.K):
                Used_time[t][k] = sum(self.v[w,p,t,k].x for w in range(args.W) for p in range(args.P))


        used_nostore = np.zeros((args.K,args.T))
        for k in range(args.K):
            for t in range(args.T):
                for j in range(args.J):
                    for p in range(args.P1):
                        for g in range(args.G):
                            used_nostore[k][t] = used_nostore[k][t] + self.f_p1[j,p,g,t,k].x

        df = pd.DataFrame(used_nostore)
        name = "{path}/{model}used_nostore.csv".format(path = os_path, model = args.model)
        df.to_csv(name)



        cost_scen = np.zeros((args.K,5))

        for k in range(args.K):
            cost_scen[k][0] = sum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t,k].x*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T))
            cost_scen[k][1] = sum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t,k].x for w in range(args.W) for p in range(args.P) for t in range(args.T))
            cost_scen[k][2] = sum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t,k].x for g in range(args.G) for t in range(args.T) for j in range(args.J))
            cost_scen[k][3] = sum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p,k].x for i in range(args.I) for w in range(args.W) for p in range(args.P))
            cost_scen[k][4] = sum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k].x*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A))

        df = pd.DataFrame(cost_scen)
        name = "{path}/{model}_cost_scen.csv".format(path = os_path, model = args.model)
        df.to_csv(name)

        if(args.model == "perfect"):
            set_up_cost_total = quicksum(self.idata.O_p[p]*self.xk[w,p,1].x for w in range(args.W) for p in range(args.P))


        set_up_cost_total = args.dprate*sum(self.idata.O_p[p]*self.x[w,p].x for w in range(args.W) for p in range(args.P))*args.K
        cost_nostore = sum(self.idata.O_p1[p]*self.f_p1[j,p,g,t,k].x for j in range(args.J) for p in range(args.P1) for g in range(args.G) for t in range(args.T) for k in range(args.K))/args.K
        operation_cost_total = sum((self.idata.wj_dis[w][j] + self.idata.I_p[p]*self.idata.O_p[p])*self.f[w,j,p,g,t,k].x*args.t_cost for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for k in range(args.K))/args.K
        holding_cost_total = sum(self.idata.CH_p[p]*self.idata.O_p[p]*self.v[w,p,t,k].x for w in range(args.W) for p in range(args.P) for t in range(args.T) for k in range(args.K))/args.K
        unmet_cost_total = sum(args.s_factor*self.idata.CU_g[g]*self.q[j,g,t,k].x for g in range(args.G) for t in range(args.T) for k in range(args.K) for j in range(args.J))/args.K
        replenish_cost_total = sum((self.idata.O_p[p] + self.idata.iw_dis[i][w]*args.t_cost)*self.s[i,w,p,k].x for i in range(args.I) for w in range(args.W) for p in range(args.P)for k in range(args.K))/args.K
        group_value_cost = sum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k].x*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for g in range(args.G) for t in range(args.T) for a in range(args.A) for k in range(args.K))/args.K
        group_value_cost = group_value_cost + sum(self.idata.A_H_flood_p1[a][p]*self.idata.Hd_weight[a][g]*self.f_p1[j,p,g,t,k].x*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P1) for g in range(args.G) for t in range(args.T) for a in range(args.A) for k in range(args.K))/args.K
        
        value_group = np.zeros((args.G))
        for g in range(args.G):
            value_group = sum(self.idata.A_H_flood[a][p]*self.idata.Hd_weight[a][g]*self.f[w,j,p,g,t,k]*args.g_value for w in range(args.W) for j in range(args.J) for p in range(args.P) for t in range(args.T) for a in range(args.A) for k in range(args.K))


        if(args.model == "baseline"):
            
            for k in range(args.K):
                percentage = np.zeros((args.T,args.G))
                for t in range(1,args.T):
                    for g in range(args.G):
                        if(sum(self.idata.demand[k][j][g][t] for j in range(args.J)) <= 10e-5):
                            percentage[t][g] = -1
                        else:
                            percentage[t][g] = sum(self.q[j,g,t,k].x for j in range(args.J))/sum(self.idata.demand[k][j][g][t] for j in range(args.J))

                df = pd.DataFrame(percentage)
                name = "{path}/{model}_group_demand_percentage_{k}.csv".format(path = os_path, model = args.model, k = str(k))
                df.to_csv(name)

            
            for k in range(args.K):
                percentage = np.zeros((args.T,args.J))
                for t in range(1,args.T):
                    for j in range(args.J):
                        if(sum(self.idata.demand[k][j][g][t] for g in range(args.G)) <= 10e-5):
                            percentage[t][j] = -1
                        else:
                            percentage[t][j] = sum(self.q[j,g,t,k].x for g in range(args.G))/sum(self.idata.demand[k][j][g][t] for g in range(args.G))

                df = pd.DataFrame(percentage)
                name = "{path}/{model}_region_demand_percentage_{k}.csv".format(path=os_path, model = args.model, k = str(k))
                df.to_csv(name)


        df_name = ["OPT","Set_Up_cost","cost_nostore","Operation_Cost","Holding_Cost","Unmet_Cost","Replenish_Cost","Victim_value"]
        data = [[self.model.ObjVal,set_up_cost_total,cost_nostore,operation_cost_total,holding_cost_total,unmet_cost_total,replenish_cost_total,group_value_cost]]
        df = pd.DataFrame(data, columns=[df_name])
        name = "{path}/{model}_Cost_structure.csv".format(path=os_path, model = args.model)
        df.to_csv(name)

        if(args.model != "perfect"):
            df = pd.DataFrame(inventory_level)
            name = "{path}/{model}_Inventory_Policy.csv".format(path=os_path, model = args.model)
            df.to_csv(name)

        df = pd.DataFrame(Used_time)
        name = "{path}/{model}_Used_time.csv".format(path=os_path, model = args.model)
        df.to_csv(name)
     

