from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from model import MHSP_extend, MHSP_Benders, inpu_data, scenariotree, MHSP_SDDP, MHSP_SDDP_Benders
from model import scenariotree
import time



if __name__ == '__main__':

    args = Arguments().parser().parse_args()
    input_data = inpu_data.input_data_class(args)


    MHSP_extend = MHSP_extend.baseline_class(args, input_data)
    MHSP_extend.run(args)

    # MHSP_Benders_sub = MHSP_Benders.subporblem(args, input_data)
    # MHSP_Benders = MHSP_Benders.Benders(args, input_data,MHSP_Benders_sub)
    # MHSP_Benders.run(args)

    # MHSP_SDDP = MHSP_SDDP.solve_SDDP(args, input_data)
    # MHSP_SDDP.run()

    MHSP_SDDP_Benders = MHSP_SDDP_Benders.solve_SDDP(args, input_data)
    MHSP_SDDP_Benders.run()



