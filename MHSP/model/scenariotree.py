import numpy as np
from itertools import product

# Each black sample path is a array, for example [100,0,0,0,100,0]

class ScenarioNode__black_path:
    def __init__(self, name, state, parent=None, demand):
        self.name = name
        self.parent_red = parent
        self.demand = demand

class ScenarioNode_red:
    def __init__(self, name, state, parent=None):
        self.name = name
        self.parent = parent
        self.children_red = []
        self.children_black = []
        self.prob2Children_red = []
        self.prob2Children_black = []
        self.index = None
        self.root = None

    def add_child_red(self, child, prob2child):
        self.children_red.append(child)
        self.prob2Children_red.append(prob2child)

    def add_child_black(self, child, prob2child):
        self.children_black.append(child)
        self.prob2Children_black.append(prob2child)

class ScenarioTree:
    def __init__(self, hurricaneData, networkData):
        self.hurricaneData = hurricaneData;
        self.networkData = networkData;
        nodeLists = self.hurricaneData.nodeLists;
        T = len(nodeLists);

        rootState = nodeLists[0][0]
        self.root = ScenarioNode(stage=0, name="Root (t=0, n=0)", state=rootState)
        self.stages = T
        self.nodes = [self.root]
        self.leaves = []
        self.root.index = 0
        self.current_index = 1  # Start indexing from 1 since root is 0
        self._build_tree()

    def _build_tree(self):
        stage_node_counts = [0] * self.stages
        SCEN = self.networkData.SCEN;
        nodeLists = self.hurricaneData.nodeLists;
        P_joint = self.hurricaneData.P_joint;
        for current_stage in range(1, self.stages):
            print("# of scenarios = ", stage_node_counts[current_stage-1])
            current_nodes = [node for node in self.nodes if node.stage == current_stage - 1]
            for node in current_nodes:
                for b in range(len(nodeLists[current_stage])):
                    child_state = nodeLists[current_stage][b]
                    if P_joint[node.state][child_state] > self.hurricaneData.smallestTransProb:
                        child_name = f"t={current_stage},n={stage_node_counts[current_stage]}"
                        child = ScenarioNode(stage=current_stage, name=child_name, state=child_state, parent=node)
                        child.index = self.current_index
                        child.prob = P_joint[node.state][child_state]
                        self.current_index += 1
                        node.add_child(child,P_joint[node.state][child_state])
                        self.nodes.append(child)
                        stage_node_counts[current_stage] += 1
                        if current_stage == self.stages-1:
                            self.leaves.append(child)

    def get_all_nodes(self):
        """Retrieve all nodes in the scenario tree."""
        all_nodes = []
        self.traverse(action=lambda node: all_nodes.append(node))
        return all_nodes

    def traverse(self, node=None, action=lambda node: print(node.data)):
        if node is None:
            node = self.root
        action(node)
        for child in node.children:
            self.traverse(child, action)

    def get_scenarios(self):
        SCEN = self.networkData.SCEN;
        Nj = self.networkData.Nj;
        T = self.stages
        number_of_scenarios = len(self.leaves);
        demands_stj = np.zeros((number_of_scenarios,T,Nj))
        prob_s = np.ones(number_of_scenarios)
        for leaf_index in range(number_of_scenarios): # leaf_index = scenario index (starting from 0)
            node = self.leaves[leaf_index]
            t = node.stage
            while t > 0: # exclude t = 0, since no demand is considered for the root (i.e., keep its demand as all zeros)
                for j in range(Nj):
                    demands_stj[leaf_index][t][j] = SCEN[node.state][j]
                prob_s[leaf_index] = prob_s[leaf_index]*node.prob
                t = t - 1
                node = node.parent
        return demands_stj,prob_s
    
    def print_tree(self):
        for node in self.nodes:
            if node.parent:
                print(f"Node {node.index}: Stage {node.stage}, Name {node.name}, Parent {node.parent.name}, State {node.state}")
            else:
                print(f"Node {node.index}: Stage {node.stage}, Name {node.name}")

    def get_scenarios_with_same_history_up_to_stage(self, stage):
        # Dictionary to hold ancestor at the specified stage as the key, and the list of leaf nodes as values
        history_groups = {}

        leaf_indices = {node: idx for idx, node in enumerate(node for node in self.nodes if len(node.children) == 0)}

        # Iterate through all nodes to find leaf nodes
        for leaf_node, leaf_index in leaf_indices.items():
            # Find the common ancestor at the specified stage
            ancestor = self.get_ancestor_at_stage(leaf_node, stage)
            if ancestor not in history_groups:
                history_groups[ancestor] = []
            history_groups[ancestor].append(leaf_index)

        # Extract the grouped scenario indices
        scenario_groups = list(history_groups.values())
        return scenario_groups
    
    def get_scenarios_stage_state(self, stage, state):
        scenario_group = [];
        number_of_scenarios = len(self.leaves);
        for leaf_index in range(number_of_scenarios): # leaf_index = scenario index (starting from 0)
            leaf_node = self.leaves[leaf_index]
            ancestor = self.get_ancestor_at_stage(leaf_node, stage)
            if ancestor.state == state:
                scenario_group.append(leaf_index);
        return scenario_group

    def get_ancestor_at_stage(self, node, stage):
        """Traverse back to find the ancestor of a node at a specific stage."""
        while node.stage > stage:
            node = node.parent
        return node

            