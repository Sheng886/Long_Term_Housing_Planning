import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Time_index_model')


        # Number of Elements in the Set
        self._parser.add_argument('--I', type=int, default=9, help='Number of Supplier')
        self._parser.add_argument('--W', type=int, default=1, help='Number of Staging Area')
        self._parser.add_argument('--J', type=int, default=67, help='Number of Study Region')
        self._parser.add_argument('--K', type=int, default=5, help='Scenario')
        self._parser.add_argument('--P', type=int, default=3, help='Housing Type')
        self._parser.add_argument('--G', type=int, default=2, help='Victims Group')
        self._parser.add_argument('--T', type=int, default=1, help='Time')

        # Path
        self._parser.add_argument('--Deprivation_Penalty_path', type=str, default='data/Deprivation_Penalty.xlsx', help='Deprivation Cost Penalty Parameter')
        self._parser.add_argument('--House_Info_path', type=str, default='data/House_Info.xlsx', help='House Information')
        self._parser.add_argument('--House_Match', type=str, default='data/House_Match.xlsx', help='House Match')
        self._parser.add_argument('--Mismatch_Penalty_path', type=str, default='data/Mismatch_Penalty.xlsx', help='Mismatch Penalty Parameter')
        self._parser.add_argument('--Staging_Area_path', type=str, default='data/Staging_Area.xlsx', help='Staging Area Information/Parameter')
        self._parser.add_argument('--Study_Region_path', type=str, default='data/Study_Region.xlsx', help='Study Region Information/Parameter')
        self._parser.add_argument('--Demand_Trailer_path', type=str, default='test_data_3/trailer_50_3.txt', help='Study Region Information/Parameter')
        self._parser.add_argument('--Demand_MHU_path', type=str, default='test_data_3/MHU_50_1.txt', help='Study Region Information/Parameter')
        self._parser.add_argument('--Supplier_path', type=str, default='data/Supplier.xlsx', help='Supplier Information/Parameter')
        self._parser.add_argument('--Timepoint_path', type=str, default='data/Timepoint.xlsx', help='Three Time points')
        self._parser.add_argument('--Unmet_Penalty_path', type=str, default='data/Unmet_Penalty.xlsx', help='Unmet Penalty Parameter')
        self._parser.add_argument('--Unused_Inventory_Penalty_path', type=str, default='data/Unused_Inventory_Penalty.xlsx', help='Unused Inventory Penalty Penalty')

        #input Datat Demand
        self._parser.add_argument('--Data_input_path_trailer', type=str, default='test_data/trailer_50.txt')
        self._parser.add_argument('--Data_input_path_MHU', type=str, default='test_data/MHU_50.txt')

        # Number of Emergency Modality can be activated
        self._parser.add_argument('--error', type=int, default=1, help='# of emergency modality')

        # Transportation Cost
        self._parser.add_argument('--trans_cost', type=int, default=10, help='unit transition cost')

        # Emergency Acquistion Factor
        self._parser.add_argument('--emergency_price_factor', type=int, default=1.2, help='emergency price factor')

        # Model
        self._parser.add_argument('--model', type=str, default="TSCC_De_Sp_class", help='model')

        # Cut
        self._parser.add_argument('--cut', type=str, default="both", help='cut--bigM/both/Spe')

        # output_name
        self._parser.add_argument('--output_name', type=str, default="")

        # without_modular
        self._parser.add_argument('--modular', type=int, default=1, help='1: with modular 0: without modular')

        # deprivation_factor
        self._parser.add_argument('--dep_factor', type=float, default=1, help='ratio of dep_factor')
        # shortage_factor
        self._parser.add_argument('--short_factor', type=float, default=1, help='ratio of shortage_factor')
        # Mis_factor
        self._parser.add_argument('--mis_factor', type=float, default=1, help='ratio of mis_factor')
        # unused_factor
        self._parser.add_argument('--unused_factor', type=float, default=1, help='ratio of mis_factor')
        
        # update/reset
        self._parser.add_argument('--UR', type=str, default='reset', help='reset: model.reset() update: model.update()')

        # cut save location
        self._parser.add_argument('--cut_path', type=str, default='cut')


        # Time-index_unit
        self._parser.add_argument('--TIU', type=int, default=1, help='Time-index_unit')
        self._parser.add_argument('--TIU_post', type=int, default=1, help='Time-index_unit_post')
        self._parser.add_argument('--TIU_pre', type=int, default=1, help='Time-index_unit_pre')

        # Upper Bound
        self._parser.add_argument('--UB', type=int, default=None, help='Upper Bound')

        # Large Scenario
        self._parser.add_argument('--Large_Scenario_main', type=int, default=0, help='Large Scenario activation')
        self._parser.add_argument('--Large_Scenario_data', type=int, default=0, help='Large Scenario activation')
        self._parser.add_argument('--Num_Large_Scenario', type=int, default=1000, help='Number of Large Scenario sampled')
        self._parser.add_argument('--random_generate', type=int, default=0, help='random generate Scenario')

        self._parser.add_argument('--save_demand_flow', type=int, default=0, help='record demand flow')
        
        # threshold policy
        self._parser.add_argument('--threshold_policy', type=int, default=0, help='threshold policy')
        self._parser.add_argument('--threshold_policy_replication', type=int, default=10, help='threshold policy')
        
        # Result_file
        self._parser.add_argument('--result_path', type=str, default='result')

        # Heuristic solution
        self._parser.add_argument('--Heu', type=int, default=1)

        # Heuristic cut save
        self._parser.add_argument('--cut_save', type=int, default=0)





    def parser(self):
        return self._parser

# python main.py --K=10 --error=1 --cut=both --Demand_Trailer_path=test_data_3/trailer_50_3.txt --Demand_MHU_path=test_data_3/MHU_50_3.txt --UR=reset --model=TSCC_De_Sp_class_TIU --TIU_post=1 --TIU_pre=2
