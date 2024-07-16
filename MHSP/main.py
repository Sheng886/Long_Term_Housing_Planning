from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from model import baseline, inpu_data, scenariotree
from model import scenariotree
import time



if __name__ == '__main__':

    args = Arguments().parser().parse_args()


    Ad_matrix = [[0,0.5,0.5,0,0,0,0],[0,0,0,0.5,0.5,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0.5,0.5],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    scecn_matrix = np.zeros((7,5,5))
    scecn_prob = np.zeros((7,5))
    tree = scenariotree.ScenarioTree(7)
    tree._build_tree_red(Ad_matrix)
    tree._build_tree_black(scecn_matrix,scecn_prob)



