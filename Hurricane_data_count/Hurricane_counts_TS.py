import pandas as pd
import matplotlib.pyplot as plt
import pdb
from statsmodels.tsa.api import GARCH

# Load the uploaded files
file_g_path = 'G.csv'
file_a_path = 'A.csv'

data_g = pd.read_csv(file_g_path)
data_a = pd.read_csv(file_a_path)

# Display the first few rows to understand the structure
data_g.head(), data_a.head()

# Count the number of landfalls per year for each file
landfall_count_g = data_g.groupby('Year').size().reset_index(name='Count_G')
landfall_count_a = data_a.groupby('Year').size().reset_index(name='Count_A')

# Get the range of years from both datasets
all_years = pd.DataFrame({'Year': range(min(landfall_count_g['Year'].min(), landfall_count_a['Year'].min()),
                                        max(landfall_count_g['Year'].max(), landfall_count_a['Year'].max()) + 1)})

# Merge with all years to fill missing years with 0
landfall_count_g = all_years.merge(landfall_count_g, on='Year', how='left').fillna(0)
landfall_count_a = all_years.merge(landfall_count_a, on='Year', how='left').fillna(0)

# Convert counts to integers
landfall_count_g['Count_G'] = landfall_count_g['Count_G'].astype(int)
landfall_count_a['Count_A'] = landfall_count_a['Count_A'].astype(int)

# Combine the counts into one dataframe
combined_landfall_counts = all_years.merge(landfall_count_g, on='Year').merge(landfall_count_a, on='Year')

# Plot each time series in a separate plot

# Plot for File G
plt.figure(figsize=(12, 6))
plt.plot(combined_landfall_counts['Year'], combined_landfall_counts['Count_G'], label='File G', marker='o')
plt.xlabel('Year')
plt.ylabel('Number of Landfalls')
plt.title('Hurricane Landfall Counts Over Time (Gulf)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for File A
plt.figure(figsize=(12, 6))
plt.plot(combined_landfall_counts['Year'], combined_landfall_counts['Count_A'], label='File A', marker='x', color='orange')
plt.xlabel('Year')
plt.ylabel('Number of Landfalls')
plt.title('Hurricane Landfall Counts Over Time (Atlantic)')
plt.grid(True)
plt.tight_layout()
plt.show()

from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller Test for File G
adf_test_g = adfuller(combined_landfall_counts['Count_G'])
adf_result_g = {
    'ADF Statistic': adf_test_g[0],
    'p-value': adf_test_g[1],
    'Critical Values': adf_test_g[4],
    'Number of Lags Used': adf_test_g[2],
    'Number of Observations Used': adf_test_g[3]
}

# Perform Augmented Dickey-Fuller Test for File A
adf_test_a = adfuller(combined_landfall_counts['Count_A'])
adf_result_a = {
    'ADF Statistic': adf_test_a[0],
    'p-value': adf_test_a[1],
    'Critical Values': adf_test_a[4],
    'Number of Lags Used': adf_test_a[2],
    'Number of Observations Used': adf_test_a[3]
}



from arch import arch_model

# Fit GARCH model for File G
arch_model_g = arch_model(new_combined_landfall_counts['Count_G'], vol='Garch', p=1, q=1)
garch_result_g = arch_model_g.fit(disp='off')

# Fit GARCH model for File A
arch_model_a = arch_model(new_combined_landfall_counts['Count_A'], vol='Garch', p=1, q=1)
garch_result_a = arch_model_a.fit(disp='off')

# Summarize the GARCH results
garch_summary_g = garch_result_g.summary()
garch_summary_a = garch_result_a.summary()

print(garch_summary_g)
print(garch_summary_a)