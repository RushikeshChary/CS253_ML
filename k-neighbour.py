import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Function to convert monetary values to numeric
def convert_to_float(crore_str):
    crore_str = crore_str.strip().lower()  
    if isinstance(crore_str, float) and np.isnan(crore_str):
        return np.nan  
    try:
        if crore_str.endswith(' crore+'):
            return float(crore_str.replace(' crore+', '')) * 10000000
        elif crore_str.endswith(' lac+'):
            return float(crore_str.replace(' lac+', '')) * 100000
        elif crore_str.endswith(' thou+'):
            return float(crore_str.replace(' thou+', '')) * 1000
        elif crore_str.endswith(' hund+'):
            return float(crore_str.replace(' hund+', '')) * 100
        else:
            return float(crore_str)  
    except ValueError:
        return np.nan  

# Load the dataset
data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Explore the data
print(data.head())

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()
data_test.columns = data_test.columns.str.strip()

# Apply conversion function to monetary columns (handling potential errors)
try:
    data['Total Assets'] = data['Total Assets'].apply(convert_to_float)
    data['Liabilities'] = data['Liabilities'].apply(convert_to_float)
    data_test['Total Assets'] = data_test['Total Assets'].apply(convert_to_float)
    data_test['Liabilities'] = data_test['Liabilities'].apply(convert_to_float)
except:
    print("Error occurred during conversion. Check data format!")

# # Plot representing the percentage of candidates with criminal records.*********************************************************

# Filter the dataset to include only candidates with criminal records
criminal_records_data = data[data['Criminal Case'] > 0]

# Group the data by party and count the number of candidates in each party
party_counts = criminal_records_data['Party'].value_counts()

# Calculate the percentage of candidates in each party
percentage_distribution = (party_counts / party_counts.sum()) * 100

# Plot the percentage distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=percentage_distribution.values, y=percentage_distribution.index, palette='viridis')
plt.xlabel('Percentage of Candidates with Criminal Records')
plt.ylabel('Party')
plt.title('Percentage Distribution of Parties with Candidates Having Criminal Records')
plt.show()
# *******************************************************************************************************************************
# Plot representing the party's percentage wealth.***************************************************************************
wealthy_candidates_data = data[data['Total Assets'] > 0]

# Group the data by party and calculate the total declared wealth for candidates in each party
party_wealth = wealthy_candidates_data.groupby('Party')['Total Assets'].sum()

total_wealth = party_wealth.sum()
percentage_distribution = (party_wealth / total_wealth) * 100

plt.figure(figsize=(10, 6))
sns.barplot(x=percentage_distribution.values, y=percentage_distribution.index, palette='magma')
plt.xlabel('Percentage of Total Wealth')
plt.ylabel('Party')
plt.title('Percentage Distribution of Parties with the Most Wealthy Candidates')
plt.show()