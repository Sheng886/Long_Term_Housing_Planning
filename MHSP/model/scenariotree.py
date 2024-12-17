import numpy as np
import pdb
from itertools import product



class ScenarioNode_red:
    def __init__(self, name, state, stage, demand, parent=None, root=False, to_node=1):
        self.stage = stage
        self.state = state
        self.name = name
        self.parent = parent
        self.prob_to_node = to_node
        self.children_red = []
        self.children_blackpath = []
        self.prob2Children_red = []
        self.prob2Children_black = []
        self.root = root
        self.demand = demand

    def add_child_red(self, child, prob2child):
        self.children_red.append(child)
        self.prob2Children_red.append(prob2child)

    def add_child_black(self, child, prob2child):
        self.children_blackpath.append(child)
        self.prob2Children_black.append(prob2child)


class ScenarioTree:
    def __init__(self, args, node_num, demand):

        self.node_all = [None for _ in range(node_num)]
        self.node_all[0] = ScenarioNode_red(name="Root (t=0, n=1)",state=args.initial_state, stage=0, root=True, demand=demand[args.initial_state])
        self.args = args


    def _build_tree_red(self, MC_tran, demand):
        
        queue = []
        queue.append(0)
        current_count = 0

        while(queue):

            node = queue[0]

            for n in range(self.args.N):
                current_count += 1
                stage_temp = self.node_all[node].stage + 1

                if(stage_temp < self.args.T):
                    queue.append(current_count)
                    
                self.node_all[node].add_child_red(current_count,MC_tran[self.node_all[node].stage][self.node_all[node].state][n])
                name = f"t={stage_temp},n={n}"
                temp_node = ScenarioNode_red(name=name,state=n, stage=stage_temp, parent=node,to_node=self.node_all[node].prob_to_node*MC_tran[self.node_all[node].stage][self.node_all[node].state][n], demand=demand[n])
                self.node_all[current_count] = temp_node

            queue.pop(0)
            
                

    def print_tree_red(self):
        for indx,node in enumerate(self.node_all):
            print(f"Node {indx}: Stage {node.stage}, State {node.state}, Name {node.name}, Parent {node.parent}, Child_red {node.prob2Children_red}, Prob_to_node {node.prob_to_node}")

    def print_tree_sce(self):
        for indx,node in enumerate(self.node_all):
            print(f"Node {indx}: State {node.state}:")
            print(node.demand)





            