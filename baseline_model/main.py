from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from model import baseline, data_class
import time



if __name__ == '__main__':

    args = Arguments().parser().parse_args()
    print("------------------------------------------")
    print("Model:", args.model)
    print("------------------------------------------")

    TSCC_re = TSCC_re.single_stage_chance(args)
    x,s,m,v = TSCC_re.run(args)
    input_data = inpu_data.input_data_class(args)
    baseline_model = baseline.baseline_class(args,input_data)
    baseline_model.run()


