import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Time_index_model')

        # model
        self._parser.add_argument('--model', type=str, default="baseline", help='Number of Supplier')

        # Number of Elements in the Set
        self._parser.add_argument('--I', type=int, default=7, help='Number of Supplier')
        self._parser.add_argument('--W', type=int, default=2, help='Number of Staging Area')
        self._parser.add_argument('--J', type=int, default=46, help='Number of Study Region')
        self._parser.add_argument('--K', type=int, default=10, help='Scenario')
        self._parser.add_argument('--P', type=int, default=4, help='Housing Type')
        self._parser.add_argument('--G', type=int, default=12, help='Victims Group')
        self._parser.add_argument('--T', type=int, default=6, help='Hurricane Season Count Month')
        self._parser.add_argument('--A', type=int, default=8, help='Number of Attribute')

        # Parameter
        self._parser.add_argument('--t_cost', type=int, default=10, help='Transportation cost')
        self._parser.add_argument('--g_value', type=int, default=5000, help='Mean of Group Value')
        self._parser.add_argument('--fair', type=float, default=0.5, help='Fairness Level')

        # Shortage Factor
        self._parser.add_argument('--s_factor', type=float, default=1, help='Shortage Factor')

        # Fairness_switch
        self._parser.add_argument('--fair_sw_group', type=int, default=1, help='fairness')
        self._parser.add_argument('--fair_sw_region', type=int, default=0, help='fairness')



        # Path
        # self._parser.add_argument('--Deprivation_Penalty_path', type=str, default='data/Deprivation_Penalty.xlsx', help='Deprivation Cost Penalty Parameter')
        # self._parser.add_argument('--House_Info_path', type=str, default='data/House_Info.xlsx', help='House Information')
        # self._parser.add_argument('--House_Match', type=str, default='data/House_Match.xlsx', help='House Match')
        # self._parser.add_argument('--Mismatch_Penalty_path', type=str, default='data/Mismatch_Penalty.xlsx', help='Mismatch Penalty Parameter')
        # self._parser.add_argument('--Staging_Area_path', type=str, default='data/Staging_Area.xlsx', help='Staging Area Information/Parameter')
        # self._parser.add_argument('--Study_Region_path', type=str, default='data/Study_Region.xlsx', help='Study Region Information/Parameter')
        # self._parser.add_argument('--Demand_Trailer_path', type=str, default='test_data_3/trailer_50_3.txt', help='Study Region Information/Parameter')
        # self._parser.add_argument('--Demand_MHU_path', type=str, default='test_data_3/MHU_50_1.txt', help='Study Region Information/Parameter')
        # self._parser.add_argument('--Supplier_path', type=str, default='data/Supplier.xlsx', help='Supplier Information/Parameter')
        # self._parser.add_argument('--Timepoint_path', type=str, default='data/Timepoint.xlsx', help='Three Time points')
        # self._parser.add_argument('--Unmet_Penalty_path', type=str, default='data/Unmet_Penalty.xlsx', help='Unmet Penalty Parameter')
        # self._parser.add_argument('--Unused_Inventory_Penalty_path', type=str, default='data/Unused_Inventory_Penalty.xlsx', help='Unused Inventory Penalty Penalty')






    def parser(self):
        return self._parser

# python main.py --K=10 --error=1 --cut=both --Demand_Trailer_path=test_data_3/trailer_50_3.txt --Demand_MHU_path=test_data_3/MHU_50_3.txt --UR=reset --model=TSCC_De_Sp_class_TIU --TIU_post=1 --TIU_pre=2
