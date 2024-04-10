import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load the dataset
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Remove leading/trailing spaces from column names
data_train.columns = data_train.columns.str.strip()
data_test.columns = data_test.columns.str.strip()

# Function to convert monetary values to numeric
def convert_crore_to_float(crore_str):
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

# Apply conversion function to monetary columns 
try:
    data_train['Total Assets'] = data_train['Total Assets'].apply(convert_crore_to_float)
    data_train['Liabilities'] = data_train['Liabilities'].apply(convert_crore_to_float)
    data_test['Total Assets'] = data_test['Total Assets'].apply(convert_crore_to_float)
    data_test['Liabilities'] = data_test['Liabilities'].apply(convert_crore_to_float)
except:
    print("Error occurred during conversion. Check data format!")

# Create new feature: Ratio of Total Assets to Liabilities
# data_train['Asset_Liability_Ratio'] = np.where(data_train['Liabilities'] != 0, data_train['Total Assets'] / data_train['Liabilities'], 0)
# data_test['Asset_Liability_Ratio'] = np.where(data_test['Liabilities'] != 0, data_test['Total Assets'] / data_test['Liabilities'], 0)

# Encode categorical variables
data_train = pd.get_dummies(data_train, columns=['Party', 'state'])
data_test = pd.get_dummies(data_test, columns=['Party', 'state'])

# Drop columns (if desired)
data_train['tot_revenue'] = data_train['Total Assets'] - data_train['Liabilities']
data_test['tot_revenue'] = data_test['Total Assets'] - data_test['Liabilities']
drop_cols = ['ID', 'Candidate', 'Constituency ∇', 'Education']  
X_train = data_train.drop(columns=drop_cols)
y_train = data_train['Education']
X_test = data_test.drop(columns=['ID', 'Candidate', 'Constituency ∇'])

# Define parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto','sqrt', 'log2']
}

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Perform grid search using GridSearchCV with 'accuracy' scoring
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train model with best parameters on entire training data
best_rf_classifier = RandomForestClassifier(random_state=42, **best_params)
best_rf_classifier.fit(X_train, y_train)

# Predictions on the test data
y_pred = best_rf_classifier.predict(X_test)

# Convert predictions to a DataFrame with 'ID' column from the test dataset
predictions_df = pd.DataFrame({'ID': data_test['ID'], 'Education': y_pred})

# Write predictions to a CSV file
predictions_df.to_csv('predictions_RF.csv', index=False)
