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

    input_data = inpu_data.input_data_class(args)
    MHSP_extend = MHSP_extend.baseline_class(args, input_data)
    MHSP_extend.run(args)



