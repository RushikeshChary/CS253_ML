import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


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
plt.figure(figsize=(15, 12))
sns.barplot(x=percentage_distribution.values, y=percentage_distribution.index, palette='viridis')
plt.xlabel('Percentage of Candidates with Criminal Records')
plt.ylabel('Party')
plt.title('Percentage Distribution of Parties with Candidates Having Criminal Records')
plt.show()
# *******************************************************************************************************************************
# # Plot representing the party's percentage wealth.***************************************************************************
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
# ***********************************************************************************************************************************
# Drop columns (if desired)
data['tot_revenue'] = data['Total Assets'] - data['Liabilities']
data_test['tot_revenue'] = data_test['Total Assets'] - data_test['Liabilities']
drop_cols = ['ID', 'Candidate', 'Constituency ∇', 'Education']  # Example columns to drop
X = data.drop(columns=drop_cols)
X_test = data_test.drop(columns=['ID', 'Candidate', 'Constituency ∇'])
y = data['Education']

# Encode categorical variables
X = pd.get_dummies(X, columns=['Party', 'state'])
X_test = pd.get_dummies(X_test, columns=['Party', 'state'])

# Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

best_params = best_params ={'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
# Train model with best parameters on entire training data
best_rf_classifier = RandomForestClassifier(random_state=42, **best_params)
best_rf_classifier.fit(X, y)

# Predictions on the test data
y_pred = best_rf_classifier.predict(X_test)

# Convert predictions to a DataFrame with 'ID' column from the test dataset
predictions_df = pd.DataFrame({'ID': np.arange(len(y_pred)), 'Education': y_pred})

# Write predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)

# Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
