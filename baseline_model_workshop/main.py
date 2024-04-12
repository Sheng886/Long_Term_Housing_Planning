from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from model import baseline, inpu_data, baseline_perfect,baseline_avg
import time



if __name__ == '__main__':

    args = Arguments().parser().parse_args()

    input_data = inpu_data.input_data_class(args)


    if(args.model == "avg"):
        baseline_model_avg = baseline_avg.baseline_class(args,input_data)
        x = baseline_model_avg.run(args)

        print("avg end")
        print("----------------")

        baseline_model = baseline.baseline_class(args,input_data,x)
        baseline_model.run(args)
        
    else:
        baseline_model = baseline.baseline_class(args,input_data)
        baseline_model.run(args)




