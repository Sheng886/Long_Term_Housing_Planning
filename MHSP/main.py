from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from model import MHSP_extend, inpu_data, scenariotree
from model import scenariotree
import time



if __name__ == '__main__':

    args = Arguments().parser().parse_args()


    Ad_matrix = [[0,0.5,0.5,0,0,0,0],[0,0,0,0.5,0.5,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0.5,0.5],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    scecn_matrix = [[[100,50],[100,20]],[[100,50],[100,50]],[[100,50],[100,50]],[[100,50],[100,50]],[[100,50],[100,50]],[[100,50],[100,50]],[[100,90],[100,50]]]
    scecn_prob = [[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]]
    tree = scenariotree.ScenarioTree(7)
    tree._build_tree_red(Ad_matrix)
    tree._build_tree_black(scecn_matrix,scecn_prob)
    # tree.print_tree_sce()

    input_data = inpu_data.input_data_class(args)
    MHSP_extend = MHSP_extend.baseline_class(args, input_data,tree)



