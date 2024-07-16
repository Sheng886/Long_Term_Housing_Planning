import numpy as np
import pdb
from itertools import product


# Each black sample path is a array, for example [100,0,0,0,100,0]

class ScenarioNode_red:
    def __init__(self, name, stage, parent=None, root=False):
        self.stage = stage
        self.name = name
        self.parent = parent
        self.children_red_path = []
        self.children_black = []
        self.prob2Children_red = []
        self.prob2Children_black = []
        self.root = root

    def add_child_red(self, child, prob2child):
        self.children_red_path.append(child)
        self.prob2Children_red.append(prob2child)

    def add_child_black(self, child, prob2child):
        self.children_black.append(child)
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
                    temp_node = ScenarioNode_red(name=name, stage=current_stage, parent=node)
                    self.node_all[node].add_child_red(child_node,Ad_matrix[node][child_node])
                    self.node_all[child_node] = temp_node

            queue.pop(0)

    def _build_tree_black(self,scenario_matrix,scenario_prob):
        
        for indx,node in enumerate(self.node_all):
            for sce in scenario_matrix[indx]:
                self.node_all[indx].add_child_black(sce,scenario_prob[indx][sce])

    def print_tree_red(self):
        for indx,node in enumerate(self.node_all):
            print(f"Node {indx}: Stage {node.stage}, Name {node.name}, Parent {node.parent}, Child_red {node.prob2Children_red}")





            