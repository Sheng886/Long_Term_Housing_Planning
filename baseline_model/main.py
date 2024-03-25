from arguments import Arguments
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from model import TSCC
from model import SS
from model import TS, TSCC_F, TSCC_decomposition, TSCC_decomposition_constant,TSCC_constant,TSCC_decomposition_special_cut,TSCC_decomposition_special_cut_class_TIU, TSCC_re
from model import TSCC_decomposition_special_cut_class, Heuristic,TSCC_decomposition_special_cut_class_TIU_eval
import time



if __name__ == '__main__':

    args = Arguments().parser().parse_args()
    print("------------------------------------------")
    print("Model:", args.model)
    print("------------------------------------------")

    tic = time.perf_counter()

    if(args.model == "TSCC"):
        model_TSCC = TSCC.two_stage_chance(args)
        model_TSCC.run(args)

    # if(args.model == "TSCC_C"):
    #     model_TSCC =TSCC_constant.two_stage_chance(args)
    #     model_TSCC.run(args)

    # if(args.model == "SS"):
    #     model_SS = SS.single_stage_chance(args)
    #     x,s,m,v = model_SS.run(args)
    #     input_data = TSCC_decomposition_special_cut_class.input_data(args)
    #     sub_N = TSCC_decomposition_special_cut_class.sub_N(args,input_data)
    #     sub_R = TSCC_decomposition_special_cut_class.sub_R(args,input_data)
    #     model_TSCC_SS = TSCC_decomposition_special_cut_class.master(args,input_data,sub_N,sub_R)
    #     model_TSCC_SS.run_eval(args,s,x,m,v)

    if(args.model == "TSCC_re"):
        TSCC_re = TSCC_re.single_stage_chance(args)
        x,s,m,v = TSCC_re.run(args)
        input_data = TSCC_decomposition_special_cut_class.input_data(args)
        sub_N = TSCC_decomposition_special_cut_class.sub_N(args,input_data)
        sub_R = TSCC_decomposition_special_cut_class.sub_R(args,input_data)
        model_TSCC_TSCC_re = TSCC_decomposition_special_cut_class.master(args,input_data,sub_N,sub_R)
        model_TSCC_TSCC_re.run_eval(args,s,x,m,v)

    
    # if(args.model == "TS"):    sub_N = TSCC_decomposition_special_cut_class.sub_N(args,input_data)
        sub_R = TSCC_decomposition_special_cut_class.sub_R(args,input_data)
        model_TSCC_D = TSCC_decomposition_special_cut_class.master(args,input_data,sub_N,sub_R)
        model_TSCC_D.run_eval(args,s,x,m,v)

    #     model_TS = TS.two_stage(args)
    #     temp_x= model_TS.run(args)
    #     model_TS_eva = TSCC.two_stage_chance(args)
    #     model_TS_eva.run(args,temp_x,True,"TS.csv")

    if(args.model == "WS"):
        index = []
        input_data = TSCC_decomposition_special_cut_class.input_data(args)
        sub_N = TSCC_decomposition_special_cut_class.sub_N(args,input_data)
        sub_R = TSCC_decomposition_special_cut_class.sub_R(args,input_data)
        model_TSCC_D = TSCC_decomposition_special_cut_class.master(args,input_data,sub_N,sub_R,index)
        x,s,m,v,z = model_TSCC_D.run(args)

    # if(args.model == "TSCC_F"):
    #     model_TSCC_F = TSCC_F.two_stage_chance_F(args)
    #     model_TSCC_F.run(args,name="TSCC_F.csv")

    if(args.model == "TSCC_De"):
        model_TSCC_D = TSCC_decomposition_constant.two_stage_chance(args)
        x,s,m,v = model_TSCC_D.run(args)

    if(args.model == "TSCC_De_Sp"):
        model_TSCC_D = TSCC_decomposition_special_cut.two_stage_chance(args)
        x,s,m,v = model_TSCC_D.run(args)

    if(args.model == "TSCC_De_Sp_class" or args.model == "TSCC_without"):
        index = []
        
        if(args.Heu == 1):
            start = time.time()
            input_data = Heuristic.input_data(args)
            Heuristic_initial = Heuristic.master(args,input_data)
            index = Heuristic_initial.run(args)
            end = time.time()
            result_name = args.result_path + "/" + str(args.model) + "_k_" + str(args.K) + "e_" + str(args.error) + "_" + str(args.cut) + "_modular_" + str(args.modular) + "_" + str(args.UR) + "_" + str(args.output_name) + "_TIU_" + str(args.TIU) + ".txt"
            print((end - start))
            f = open(result_name,"w+")
            f.write("Heu-Diff process time: %s \r" % (end - start))
            print("----------------------------")
            print("Heuristic End")
            f.close()
            print("index/error:", index,"/",args.error)
            print("----------------------------")

        input_data = TSCC_decomposition_special_cut_class.input_data(args)
        sub_N = TSCC_decomposition_special_cut_class.sub_N(args,input_data)
        sub_R = TSCC_decomposition_special_cut_class.sub_R(args,input_data)
        model_TSCC_D = TSCC_decomposition_special_cut_class.master(args,input_data,sub_N,sub_R,index)
        x,s,m,v,z = model_TSCC_D.run(args)
        print("----------------------------")
        print("Model End")
        print("----------------------------")

        if(args.Large_Scenario_main == 1):
            args.Large_Scenario_data = 1 
            args.error = int(args.Num_Large_Scenario*(args.error/args.K))
            input_data = TSCC_decomposition_special_cut_class.input_data(args)
            sub_N = TSCC_decomposition_special_cut_class.sub_N(args,input_data)
            sub_R = TSCC_decomposition_special_cut_class.sub_R(args,input_data)
            model_TSCC_D = TSCC_decomposition_special_cut_class.master(args,input_data,sub_N,sub_R)
            model_TSCC_D.run_eval(args,s,x,m,v)


        if(args.TIU_eval == 1):
            temp = args.TIU
            args.TIU = 1
            input_data = TSCC_decomposition_special_cut_class_TIU_eval.input_data(args,temp)
            sub_N = TSCC_decomposition_special_cut_class_TIU_eval.sub_N(args,input_data)
            sub_R = TSCC_decomposition_special_cut_class_TIU_eval.sub_R(args,input_data)
            model_TSCC_D = TSCC_decomposition_special_cut_class_TIU_eval.master(args,input_data,sub_N,sub_R,index)
            model_TSCC_D.run(args,z,x,s,m,v)
            
            print("----------------------------")
            print("EVAL End")
            print("----------------------------")

    if(args.model == "TSCC_De_Sp_class_TIU"):
        input_data = TSCC_decomposition_special_cut_class_TIU.input_data(args)
        sub_N = TSCC_decomposition_special_cut_class_TIU.sub_N(args,input_data)
        sub_R = TSCC_decomposition_special_cut_class_TIU.sub_R(args,input_data)
        model_TSCC_D = TSCC_decomposition_special_cut_class_TIU.master(args,input_data,sub_N,sub_R)
        x,s,m,v = model_TSCC_D.run(args)



        if(args.Large_Scenario_main == 1):
            args.Large_Scenario_data = 1 
            args.error = int(args.Num_Large_Scenario*(args.error/args.K))
            input_data = TSCC_decomposition_special_cut_class_TIU.input_data(args)
            sub_N = TSCC_decomposition_special_cut_class_TIU.sub_N(args,input_data)
            sub_R = TSCC_decomposition_special_cut_class_TIU.sub_R(args,input_data)
            model_TSCC_D = TSCC_decomposition_special_cut_class_TIU.master(args,input_data,sub_N,sub_R)
            model_TSCC_D.run_eval(args,s,x,m,v)


    toc = time.perf_counter()
    print(f" finish in {toc - tic:0.4f} seconds")

