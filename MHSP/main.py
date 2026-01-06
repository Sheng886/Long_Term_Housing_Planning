from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from model import MHSP_extend, MHSP_Benders, inpu_data, scenariotree, MHSP_SDDP, MHSP_RS_SDDP
from model import scenariotree
import time



if __name__ == '__main__':

    args = Arguments().parser().parse_args()
    input_data = inpu_data.input_data_class(args)

    print("Model:",args.Model)

    start_time = time.time()
    if(args.Model == "Extend"):
        start_time_setup = time.time()
        MHSP_extend = MHSP_extend.baseline_class(args, input_data)
        end_time_setup = time.time()
        print("Model set up time:", end_time_setup-start_time_setup)
        MHSP_extend.run(args)
    elif(args.Model == "2SSP"):
        start_time_setup = time.time()
        MHSP_Benders_sub = MHSP_Benders.subporblem(args, input_data)
        MHSP_Benders = MHSP_Benders.Benders(args, input_data,MHSP_Benders_sub)
        end_time_setup = time.time()
        print("Model set up time:", end_time_setup-start_time_setup)
        MHSP_Benders.run(args)
    elif(args.Model == "SDDP"):
        start_time_setup = time.time()
        MHSP_SDDP = MHSP_SDDP.solve_SDDP(args, input_data)
        end_time_setup = time.time()
        MHSP_SDDP.run()
        print("Model set up time:", end_time_setup-start_time_setup)
    elif(args.Model == "MHSP_RS_SDDP"):
        print("Review Interval:",args.R)
        start_time_setup = time.time()
        MHSP_RS_SDDP = MHSP_RS_SDDP.solve_SDDP(args, input_data)
        end_time_setup = time.time()
        print("Model set up time:", end_time_setup-start_time_setup)
        MHSP_RS_SDDP.run()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.4f} seconds")

    # MHSP_SDDP_Benders = MHSP_SDDP_Benders.solve_SDDP(args, input_data)
    # MHSP_SDDP_Benders.run()



