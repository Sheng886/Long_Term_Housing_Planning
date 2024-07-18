import numpy as np
import pdb
from itertools import product


class Scenariopath_black:
    def __init__(self, demand1,demand2):
        self.demand = [demand1,demand2]


class ScenarioNode_red:
    def __init__(self, name, stage, parent=None, root=False, to_node=1):
        self.stage = stage
        self.name = name
        self.parent = parent
        self.prob_to_node = to_node
        self.children_red = []
        self.children_blackpath = []
        self.prob2Children_red = []
        self.prob2Children_black = []
        self.root = root

    def add_child_red(self, child, prob2child):
        self.children_red.append(child)
        self.prob2Children_red.append(prob2child)

    def add_child_black(self, child, prob2child):
        self.children_blackpath.append(child)
        self.prob2Children_black.append(prob2child)


class ScenarioTree:
    def __init__(self, node_num):

        self.node_all = [None for _ in range(node_num)]
        self.node_all[0] = ScenarioNode_red(name="Root (s=0, n=0)", stage=0, root=True)

    def _build_tree_red(self,Ad_matrix):
        
        queue = []
        queue.append(0)
        current_stage = 0
        while(queue):

            node = queue[0]

            for child_node,check in enumerate(Ad_matrix[node]):
                if(check != 0):
                    queue.append(child_node)

                    current_stage = self.node_all[node].stage + 1
                    name = f"s={current_stage},n={child_node}"
                    temp_node = ScenarioNode_red(name=name, stage=current_stage, parent=node,to_node=self.node_all[node].prob_to_node*Ad_matrix[node][child_node])
                    self.node_all[node].add_child_red(child_node,Ad_matrix[node][child_node])
                    self.node_all[child_node] = temp_node

            queue.pop(0)

    def _build_tree_black(self,scenario_matrix1,scenario_matrix2,scenario_prob):
        
        for indx,node in enumerate(self.node_all):
            for indx2,sce in enumerate(scenario_matrix1[indx]):
                temp_path = Scenariopath_black(sce,scenario_matrix2[indx][indx2])
                self.node_all[indx].add_child_black(temp_path,scenario_prob[indx][indx2])
                

    def print_tree_red(self):
        for indx,node in enumerate(self.node_all):
            print(f"Node {indx}: Stage {node.stage}, Name {node.name}, Parent {node.parent}, Child_red {node.prob2Children_red}, Prob_to_node {node.prob_to_node}")

    def print_tree_sce(self):
        for indx,node in enumerate(self.node_all):
            for month in node.children_blackpath:
                print(f"Node {indx}: Scen {month.demand}")





            