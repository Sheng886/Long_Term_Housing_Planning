import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from scipy.stats import poisson
from scipy.stats import gaussian_kde
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import math
import csv
from arguments import Arguments
from scipy import stats
import pdb

###### Distacne Matrix ######

def haversine(lon1, lat1, lon2, lat2):
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371  # Radius of Earth in kilometers
    
    return c * r


def distribute_hurricane_month(n_hurricane, month_probabilities):

    month_boxes = len(month_probabilities)
    month_distribution = np.random.multinomial(n_hurricane, month_probabilities)
    
    return month_distribution

def inverse_value_from_reg(predictions,lambda_value,min_,std,mean):
    
    invers_values = []

    for prediction in predictions:
        # Calculate the expression
        value = lambda_value * prediction + 1
        
        # Apply the condition: if value <= 0, set to 0, otherwise perform the calculation
        if value <= 0:
            invers_values.append(0)
        else:
            # Perform the transformation and rounding
            transformed_value = ((value ** (1 / lambda_value) + min_) * std + mean).round()
            invers_values.append(transformed_value)

    return invers_values

def trans_output(data):
    year_mapping = {year: idx for idx, year in enumerate(sorted(data["Year"].unique()))}
    data["Year"] = data["Year"].map(year_mapping)

    # Get the unique years and categories for constructing transition matrices
    years = data["Year"].unique()
    categories = sorted(data["Category"].unique())
    n_categories = len(categories)

    # Build the transition matrix for each year
    matrix_all = np.zeros((len(years),n_categories, n_categories))
    for year in range(len(years)):
        # Filter data for the current year
        year_data = data[data["Year"] == year]
        
        # Initialize an empty matrix
        matrix = np.zeros((n_categories, n_categories))
        
        # Populate the matrix using the transition probabilities
        for _, row in year_data.iterrows():
            i = categories.index(row["Category"])
            j = categories.index(row["Next_Category"])
            matrix[i, j] = row["Transition_Prob"]
        
        # Store the matrix in the dictionary
        matrix_all[year] = matrix

    return matrix_all


def main_generator(args):

    # Data input ------------------------------------------------------------------

    Frequency = pd.read_excel(args.Frequency)
    Frequency = Frequency.to_dict()
    lambda_A = Frequency["Atlantic"][0]
    lambda_G = Frequency["Gulf"][0]


    # Altantic Transition Matrix

    data_A = pd.read_csv(args.tran_A)
    matrix_tran_A = trans_output(data_A)
    data_G = pd.read_csv(args.tran_G)
    matrix_tran_G = trans_output(data_G)

    Atlantic_month_dis = pd.read_excel(args.Atlantic_month_dis)
    Atlantic_month_dis = Atlantic_month_dis.to_numpy()[0]

    Atlantic_SS_dis = pd.read_excel(args.Atlantic_SS_dis)
    Atlantic_SS_dis = Atlantic_SS_dis.to_numpy()[0]

    Gulf_month_dis = pd.read_excel(args.Gulf_month_dis)
    Gulf_month_dis = Gulf_month_dis.to_numpy()[0]
    
    Gulf_SS_dis = pd.read_excel(args.Gulf_SS_dis)
    Gulf_SS_dis = Gulf_SS_dis.to_numpy()[0]

    # Study Region Data
    Data_Atlantic_df = pd.read_csv(args.Data_Atlantic)
    coord_A = Data_Atlantic_df[['lng','lat']]  
    
    Data_Gulf_df = pd.read_csv(args.Data_Gulf)
    coord_G = Data_Gulf_df[['lng','lat']]  

    # Regression   --------------------------------------------------------------

    df_regression = pd.read_csv(args.dataall)
    X = df_regression[['node1','SS_scale','population','SVI']]  
    
    y_1 = df_regression['type1']
    y_1_stand =  (y_1 - y_1.mean()) / y_1.std()
    min_y1 = min(y_1_stand)
    y_1_stand = y_1_stand - min_y1
    y_1_stand = y_1_stand.replace(0, 0.0001)

    y_2 = df_regression['type2']
    y_2_stand =  (y_2 - y_2.mean()) / y_2.std()
    min_y2 = min(y_2_stand)
    y_2_stand = y_2_stand - min_y2
    y_2_stand = y_2_stand.replace(0, 0.0001)

    
    data_boxcox1, lambda_value1 = boxcox(y_1_stand)
    reg1 = LinearRegression()
    reg1.fit(X, data_boxcox1)

    data_boxcox2, lambda_value2 = boxcox(y_2_stand)
    reg2 = LinearRegression()
    reg2.fit(X, data_boxcox2)

    
    # Landfall     --------------------------------------------------------------
    Hurricane_landfall_Atlantic_df = pd.read_csv(args.Hurricane_landfall_Atlantic)

    longitude = Hurricane_landfall_Atlantic_df['Longitude'].values
    latitude = Hurricane_landfall_Atlantic_df['Latitude'].values
    coords_Atlantic = np.vstack([longitude, latitude])
    kde_Atlantic = gaussian_kde(coords_Atlantic)

    Hurricane_landfall_Gulf_df = pd.read_csv(args.Hurricane_landfall_Gulf)
    longitude = Hurricane_landfall_Gulf_df['Longitude'].values
    latitude = Hurricane_landfall_Gulf_df['Latitude'].values
    coords_Gulf = np.vstack([longitude, latitude])
    kde_Gulf = gaussian_kde(coords_Gulf)

    Regression_par = pd.read_excel(args.Regression_par)
    Regression_par = Regression_par.to_dict()

    
    # State (A,G) ------------------------------------------------------------------

    states = []
    ###
    # H: 0~2
    # M: 3~5
    # H: 5up
    ###
    states_A = ["H","L"]
    states_G = ["H","M","L"]
    temp = []

    for n1 in states_A:
        for n2 in states_G:
            states.append([n1, n2])

    args.N = len(states_A) * len(states_G)

    # state range ------------------------------------------------------------------

    rangeL_A = range(3)       # 0 ≤ X ≤ 2
    rangeH_cutoff_A =range(3, 7)     # X ≥ 3 (complement of sum of P(X ≤ 6))


    rangeL_G = range(3)       # 0 ≤ X ≤ 2
    rangeM_G = range(3, 6)    # 3 ≤ X ≤ 5
    rangeH_cutoff_G = range(6, 11)      # X ≥ 6 (complement of sum of P(X ≤ 10))

    


    # Probability --------------------------------------------------------------

    # prob_A = []
    # prob_G = []
    # for i in range(rangeH_cutoff):
    #     prob_A.append(poisson.pmf(i, lambda_A))
    #     prob_G.append(poisson.pmf(i, lambda_G))

    # prob_0_to_2_A = sum(prob_A[n] for n in rangeL)
    # prob_3_to_5_A = sum(prob_A[n] for n in rangeM)
    # prob_6_and_above_A = 1 - sum(prob_A[n] for n in range(rangeH_cutoff))

    # prob_0_to_2_G = sum(prob_G[n] for n in rangeL)
    # prob_3_to_5_G = sum(prob_G[n] for n in rangeM)
    # prob_6_and_above_G = 1 - sum(prob_G[n] for n in range(rangeH_cutoff))

    # prob_A_state = [prob_0_to_2_A, prob_3_to_5_A, prob_6_and_above_A]
    # prob_G_state = [prob_0_to_2_G, prob_3_to_5_G, prob_6_and_above_G]

    # # conditional --------------------------------------------------------------
    
    # Probabilities for A
    prob_0_to_2_A_cond = np.array([76438, 19016, 3833]) / np.sum([76438, 19016, 3833])
    prob_3_to_up_A_cond = np.array([610, 94, 7, 2]) / np.sum([610, 94, 7, 2])

    # Probabilities for G
    prob_0_to_2_G_cond = np.array([90112, 95085, 62782]) / np.sum([90112, 95085, 62782])
    prob_3_to_5_G_cond = np.array([31560, 13343, 4812]) / np.sum([31560, 13343, 4812])
    prob_6_to_up_G_cond = np.array([1600, 502, 153, 35, 16]) / np.sum([1600, 502, 153, 35, 16])




    # MC Trans Matrix --------------------------------------------------------------

    MC_trans = np.zeros((args.T,args.N,args.N))
    state_state_pro = []
    for t in range(args.T):
        for n in range(args.N):
            for n_next in range(args.N): 
                # print(t, states[n][0],states[n][1], "->", states[n_next][0],states[n_next][1])
                
                if(states[n][0] == "H"):
                    n_A_temp = 1
                else:
                    n_A_temp = 0

                if(states[n_next][0] == "H"):
                    pro_A_temp = matrix_tran_A[t][n_A_temp][1]
                elif(states[n_next][0] == "L"):
                    pro_A_temp = matrix_tran_A[t][n_A_temp][0]

                if(states[n][1] == "H"):
                    n_G_temp = 2
                elif(states[n][1] == "M"):
                    n_G_temp = 1
                else:
                    n_G_temp = 0

                if(states[n_next][1] == "H"):
                    pro_G_temp = matrix_tran_G[t][n_G_temp][2]
                elif(states[n_next][1] == "M"):
                    pro_G_temp = matrix_tran_G[t][n_G_temp][1]
                else:
                    pro_G_temp = matrix_tran_G[t][n_G_temp][0]

                joint_prob  = pro_A_temp*pro_G_temp
                # print(pro_A_temp,pro_G_temp,joint_prob)
                MC_trans[t][n][n_next] = joint_prob
    


    
    # Demand Generator--------------------------------------------------------------
    # Step 1 distribute the numbers to the month by month distribution -------------


    args.J = len(pd.merge(Data_Atlantic_df, Data_Gulf_df, on='county_full', how='outer'))

    demand = np.zeros((args.N,args.K,args.M+1,args.J,args.G))

    for n_index,n in enumerate(states):
        for k in range(args.K):

            # Sample # of hurricane
            # Atlantic
            if(n[0] == "L"):
                freq_sample_path_A = np.random.choice(rangeL_A, size=1, p=prob_0_to_2_A_cond)[0]
            else:
                freq_sample_path_A = np.random.choice(rangeH_cutoff_A, size=1, p=prob_3_to_up_A_cond)[0]

            # Gulf
            if(n[1] == "L"):
                freq_sample_path_G = np.random.choice(rangeL_G, size=1, p=prob_0_to_2_G_cond)[0]
            elif(n[1]  == "M"):
                freq_sample_path_G = np.random.choice(rangeM_G, size=1, p=prob_3_to_5_G_cond)[0]
            else:
                freq_sample_path_G = np.random.choice(rangeH_cutoff_G, size=1, p=prob_6_to_up_G_cond)[0]

            
            # Distribute into month
            A_month_freq = distribute_hurricane_month(freq_sample_path_A, Atlantic_month_dis)
            G_month_freq = distribute_hurricane_month(freq_sample_path_G, Gulf_month_dis)

            # print(freq_sample_path_A,freq_sample_path_G)


            for m in range(args.M+1):

                total_demand_in_month_A = np.zeros((2,len(Data_Atlantic_df)))
                total_demand_in_month_G = np.zeros((2,len(Data_Gulf_df)))

                if(m != 0):

                    A_num_hurricane = A_month_freq[m-1]
                    G_num_hurricane = G_month_freq[m-1]

                    for h in range(A_num_hurricane):

                        # Calculate distance from landfall (node1)
                        Landfall = kde_Atlantic.resample(1)
                        disatnce_A = []
                        for index, row in coord_A.iterrows():
                            distance = haversine(Landfall[0][0],Landfall[1][0],row['lng'],row['lat'])
                            disatnce_A.append(distance)

                        Data_Atlantic_df['node1'] = disatnce_A

                        # SS sample
                        SS = np.random.choice([1,2,3,4,5], size=1, p=Atlantic_SS_dis)[0]
                        Data_Atlantic_df['SS_scale'] = SS

                        
                        # Generate Demand
                        X_A = Data_Atlantic_df[['node1','SS_scale','population','SVI']]

                        type1_demand = inverse_value_from_reg(reg1.predict(X_A),lambda_value1,min_y1,y_1.std(),y_1.mean())
                        type2_demand = inverse_value_from_reg(reg2.predict(X_A),lambda_value2,min_y2,y_2.std(),y_2.mean())

                        ## thredhold < 10 demand then 0 demand
                        type1_demand = [0 if value <= 10 else value for value in type1_demand]
                        type2_demand = [0 if value <= 10 else value for value in type2_demand]

                        total_demand_in_month_A[0] = total_demand_in_month_A[0] + type1_demand
                        total_demand_in_month_A[1] = total_demand_in_month_A[1] + type2_demand



                    for h in range(G_num_hurricane):

                        # Calculate distance from landfall (node1)
                        Landfall = kde_Gulf.resample(1)
                        disatnce_G = []
                        for index, row in coord_G.iterrows():
                            distance = haversine(Landfall[0][0],Landfall[1][0],row['lng'],row['lat'])
                            disatnce_G.append(distance)

                        Data_Gulf_df['node1'] = disatnce_G

                        # SS sample
                        SS = np.random.choice([1,2,3,4,5], size=1, p=Gulf_SS_dis)[0]
                        Data_Gulf_df['SS_scale'] = SS

                        # Generate Demand

                        X_G = Data_Gulf_df[['node1','SS_scale','population','SVI']]  

                        type1_demand = inverse_value_from_reg(reg1.predict(X_G),lambda_value1,min_y1,y_1.std(),y_1.mean())
                        type2_demand = inverse_value_from_reg(reg2.predict(X_G),lambda_value2,min_y2,y_2.std(),y_2.mean())
                        
                        ## thredhold < 10 demand then 0 demand
                        type1_demand = [0 if value <= 10 else value for value in type1_demand]
                        type2_demand = [0 if value <= 10 else value for value in type2_demand]

                        total_demand_in_month_G[0] = total_demand_in_month_G[0] + type1_demand
                        total_demand_in_month_G[1] = total_demand_in_month_G[1] + type2_demand


                # Combine the FL data

                study_region_demand_A = pd.DataFrame(columns=['county', 'type1', 'type2'])
                study_region_demand_G = pd.DataFrame(columns=['county', 'type1', 'type2'])
                study_region_demand = pd.DataFrame(columns=['county', 'type1', 'type2'])

                study_region_demand_A['county'] = Data_Atlantic_df['county_full']
                study_region_demand_A['type1'] = total_demand_in_month_A[0]
                study_region_demand_A['type2'] = total_demand_in_month_A[1]

                study_region_demand_G['county'] = Data_Gulf_df['county_full']
                study_region_demand_G['type1'] = total_demand_in_month_G[0]
                study_region_demand_G['type2'] = total_demand_in_month_G[1]

                study_region_demand = pd.merge(study_region_demand_A, study_region_demand_G, on='county', how='outer', suffixes=('_df1', '_df2'))

                study_region_demand['type1'] = study_region_demand['type1_df1'].fillna(0) + study_region_demand['type1_df2'].fillna(0)
                study_region_demand['type2'] = study_region_demand['type2_df2'].fillna(0) + study_region_demand['type2_df2'].fillna(0)

                study_region_demand.drop(columns=['type1_df1', 'type1_df2', 'type2_df2', 'type2_df2'], inplace=True)

                study_region_demand_np = study_region_demand[['type1','type2']].to_numpy()


                demand[n_index][k][m] = study_region_demand_np


    states_len = len(states)
    for n1 in range(states_len):
        for n2 in range(n1,states_len):
            demand1 = []
            demand2 = []
            for k in range(args.K):
                demand1.append(sum(demand[n1][k][m][j][g] for j in range(args.J) for m in range(args.M+1) for g in range(args.G)))
                demand2.append(sum(demand[n2][k][m][j][g] for j in range(args.J) for m in range(args.M+1) for g in range(args.G)))
            # print(n1,n2, demand1, demand2)
            statistic, p_value = stats.ks_2samp(demand1, demand2)
            print(n1,n2)
            print(f"KS Statistic: {statistic}")
            print(f"P-value: {p_value}")

    # pdb.set_trace()

    # demand1 = demand[t][n_index]

   
    # Get indices of all elements in the array
    indices = np.array(np.meshgrid(
        np.arange(args.N),
        np.arange(args.K),
        np.arange(args.M+1),
        np.arange(args.J),
        np.arange(args.G),
        indexing='ij'
    )).reshape(5, -1).T

    # Flatten the data array
    values = demand.flatten()
    # Combine indices and values
    result = np.column_stack((indices, values))
    # Save to CSV
    filename = "demand_data/Demand_Stage_" + str(args.T) + "_States_" + str(args.N) + "_Study_" + str(args.J) + "_month_" + str(args.M) + "_K_" + str(args.K) + ".csv"
    np.savetxt(filename, result, delimiter=",", header="State,Scenario,Month,Sutdy_Region,Group,Demand", comments="", fmt="%.6f")

    
    indices = np.array(np.meshgrid(
        np.arange(args.T),
        np.arange(args.N),
        np.arange(args.N),
        indexing='ij'
    )).reshape(3, -1).T

    values = MC_trans.flatten()
    # Save to CSV
    result = np.column_stack((indices, values))
    filename_mc= "demand_data/MC_trans_Stage_" + str(args.T) + "_States_" + str(args.N) + ".csv"
    np.savetxt(filename_mc, result, delimiter=",", header="Stage,state,state_next", comments="", fmt="%.6f")



if __name__ == '__main__':

    args = Arguments().parser().parse_args()
    main_generator(args)













