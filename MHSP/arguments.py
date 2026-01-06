import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Time_index_model')


        # Number of Elements in the Set
        self._parser.add_argument('--I', type=int, default=3, help='Number of Supplier')
        self._parser.add_argument('--W', type=int, default=2, help='Number of Staging Area')
        self._parser.add_argument('--J', type=int, default=8, help='Number of Study Region')
        self._parser.add_argument('--K', type=int, default=25, help='Scenario')
        self._parser.add_argument('--P', type=int, default=2, help='Housing Type')
        self._parser.add_argument('--G', type=int, default=2, help='Victims Group')
        self._parser.add_argument('--M', type=int, default=6, help='number of node in a scen path')
        self._parser.add_argument('--A', type=int, default=8, help='Number of Attribute')
        
        # Tree
        self._parser.add_argument('--T', type=int, default=10, help='Number of stage')
        self._parser.add_argument('--N', type=int, default=16, help='Number of state')

        self._parser.add_argument('--N_G', type=int, default=4, help='Number of state for Gulf')
        self._parser.add_argument('--N_A', type=int, default=4, help='Number of state for Atlantic')

        self._parser.add_argument('--TN', type=int, default=7, help='number of node in tree')
        self._parser.add_argument('--initial_state', type=int, default=0, help='initial_state_root')

        # Parameter
        self._parser.add_argument('--t_cost', type=int, default=10, help='Transportation cost')
        self._parser.add_argument('--g_value', type=int, default=5000, help='Mean of Group Value')

        # Scen Factor
        self._parser.add_argument('--s_factor', type=int, default=1, help='Shortage Factor')

        # Baseline deamand
        self._parser.add_argument('--DTrailer', type=int, default=500, help='Baseline deamand Trailer')
        self._parser.add_argument('--DMHU', type=int, default=50, help='Baseline deamand MHU')

        # Stop_Criteria
        self._parser.add_argument('--MAX_ITER', type=int, default=100000, help='Shortage Factor')
        self._parser.add_argument('--time_limit', type=int, default=10800, help='Shortage Factor')
        self._parser.add_argument('--CUTVIOL_MAXITER', type=float, default=100000, help='Shortage Factor')
        self._parser.add_argument('--STALL', type=int, default=200, help='Shortage Factor')
        self._parser.add_argument('--LB_TOL', type=float, default=1e-4, help='Shortage Factor')

        # Method
        self._parser.add_argument('--Strategic_node_sovling', type=int, default=1, help='0: Extend Formulation 1: Benders Decompostion')
        self._parser.add_argument('--Model', type=str, default="SDDP", help='SDDP/Extend/2SSP')

        # Debug
        self._parser.add_argument('--Cost_print', type=bool, default=False, help='0: Extend Formulation 1: Benders Decompostion')

        # Demand path
        self._parser.add_argument('--demand_path', type=str, default='demand_data/Demand_Stage_2_States_6_Study_278_month_6_K_2.csv', help='Demand')
        self._parser.add_argument('--MC_trans_path', type=str, default='demand_data/MC_trans_Stage_2_States_6.csv', help='MC_trans')


        self._parser.add_argument('--cut_pool', type=int, default=1, help='cut_pool')


        # Demand Generator Path
        self._parser.add_argument('--Atlantic_month_dis', type=str, default='generator_data/Atlantic_month_dis.xlsx', help='Atlantic_month_dis file path')
        self._parser.add_argument('--Atlantic_SS_dis', type=str, default='generator_data/Atlantic_SS_dis.xlsx', help='Atlantic_SS_dis file path')
        self._parser.add_argument('--Data_Atlantic', type=str, default='generator_data/Data_Atlantic_all_state.csv', help='Data_Atlantic file path')
        self._parser.add_argument('--Data_Gulf', type=str, default='generator_data/Data_Gulf_all_state.csv', help='Data_Gulf file path')
        self._parser.add_argument('--Frequency', type=str, default='generator_data/Frequency.xlsx', help='Frequency file path')
        self._parser.add_argument('--Gulf_month_dis', type=str, default='generator_data/Gulf_month_dis.xlsx', help='Gulf_month_dis file path')
        self._parser.add_argument('--Gulf_SS_dis', type=str, default='generator_data/Gulf_SS_dis.xlsx', help='Gulf_SS_dis file path')
        self._parser.add_argument('--Hurricane_landfall_Atlantic', type=str, default='generator_data/Hurricane_landfall_Atlantic.csv', help='Hurricane_landfall_Atlantic file path')
        self._parser.add_argument('--Hurricane_landfall_Gulf', type=str, default='generator_data/Hurricane_landfall_Gulf.csv', help='Hurricane_landfall_Gulf file path')
        self._parser.add_argument('--Regression_par', type=str, default='generator_data/Regression_par.xlsx', help='Regression_par file path')
        self._parser.add_argument('--dataall', type=str, default='generator_data/Data_all.csv', help='Regression_par file path')

        self._parser.add_argument('--tran_A', type=str, default='generator_data/trans_A_30.csv', help='tran_Atlantic')
        self._parser.add_argument('--tran_G', type=str, default='generator_data/trans_G_30.csv', help='tran_Gulf')

        # Factor
        self._parser.add_argument('--O_p_factor', type=float, default=1, help='Acquire Factor')
        
        # Based on O_p
        self._parser.add_argument('--H_p_factor', type=float, default=0.05, help='Holding Factor')
        self._parser.add_argument('--R_p_factor', type=float, default=0.6, help='Recycle Factor')
        self._parser.add_argument('--C_u_factor', type=float, default=5, help='Unmet Factor')

        self._parser.add_argument('--E_w_factor', type=float, default=0.2, help='Increasing Capacity Factor')
        
        # Help with SA (import from file)
        self._parser.add_argument('--B_i_factor', type=float, default=1, help='Production Capacity Factor')
        self._parser.add_argument('--P_p_factor', type=float, default=1, help='Lead time Factor')
        self._parser.add_argument('--Cp_w_factor', type=float, default=1, help='Initial Staging Area')
        self._parser.add_argument('--II_factor', type=float, default=0.25, help='Initial Invenotry')

        self._parser.add_argument('--price_strategic', type=float, default=0.8, help='Initial Invenotry')
        
        # Evaulation switch
        self._parser.add_argument('--evaluate_switch', type=bool, default=False, help='Acquire Factor')
        
        # Policy
        self._parser.add_argument('--Policy', type=str, default='', help='Policy')

        # Scale Down Factor
        self._parser.add_argument('--sc', type=float, default='1', help='Scale Down Factor')

        # Scale Down Factor
        self._parser.add_argument('--sample_path', type=str, default='MC', help='sample path method')

        # RS Policy
        self._parser.add_argument('--R', type=int, default='1', help='reviewed interval')






    def parser(self):
        return self._parser

# python main.py --K=10 --error=1 --cut=both --Demand_Trailer_path=test_data_3/trailer_50_3.txt --Demand_MHU_path=test_data_3/MHU_50_3.txt --UR=reset --model=TSCC_De_Sp_class_TIU --TIU_post=1 --TIU_pre=2
