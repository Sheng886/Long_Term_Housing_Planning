import numpy as np
import pandas as pd
import csv
import pdb

df_tract_county = pd.read_csv('Tract_to_county.csv')

dic = {}
county_names = df_tract_county['CountyName'].unique()


for count_name in county_names:
    df_temp = df_tract_county.loc[df_tract_county['CountyName'] == count_name]
    tract_temp = df_temp['Tract'].to_numpy()
    dic[count_name] = tract_temp

# --------------------------------------------------------------------------------

np.zeros((150,46,6))


for i in range(1,97):
    name = 'sel_storms_new_outputs_original/output_'+ str(i) + '.xlsx'

    df_output = pd.read_excel(name)


    Short_term_shelter_needs_household_Nh = []
    Long_term_shelter_needs_household_Lh = []
    demand = []

    for count_name in county_names:
        temp1 = 0
        temp2 = 0

        for tract_number in dic[count_name]:
            temp1 = df_output.loc[df_output['Tract_ID'] == tract_number]['Short_term_shelter_needs_household_Nh'].values[0] + temp1
            temp2 = df_output.loc[df_output['Tract_ID'] == tract_number]['Long_term_shelter_needs_household_Lh'].values[0] + temp2

        Short_term_shelter_needs_household_Nh.append(temp1)
        Long_term_shelter_needs_household_Lh.append(temp2)
        demand.append(temp1+temp2)

    # pdb.set_trace()
    dic_demand = {"County":county_names, "S_Shelter_housedhold": Short_term_shelter_needs_household_Nh,"L_Shelter_housedhold":Long_term_shelter_needs_household_Lh, "demand":demand}
    df = pd.DataFrame(dic_demand)
    name_output = 'county_level_original/output_'+str(i)+'.csv'
    df.to_csv(name_output, index=False)  


