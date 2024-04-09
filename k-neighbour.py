import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to convert monetary values to numeric
def convert_crore_to_float(crore_str):
    """
    This function converts a string in crore format (e.g., '32 Crore+') to a float value.

    Args:
        crore_str: The string representation of a number in crore format.

    Returns:
        The float value of the number in crore, or NaN if the format is invalid.
    """
    if isinstance(crore_str, float) and np.isnan(crore_str):
        return np.nan  # Return NaN if already NaN
    crore_str = crore_str.strip().lower()  # Remove leading/trailing spaces and convert to lowercase
    if crore_str == '0':
        return 0.0
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
            return float(crore_str)  # If no unit is specified, assume it's already in crore
    except ValueError:
        return np.nan  # Return NaN for invalid formats

# Load the dataset
data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()
data_test.columns = data_test.columns.str.strip()

# Encode categorical variables
data = pd.get_dummies(data, columns=['Party', 'state'])
data_test = pd.get_dummies(data_test, columns=['Party', 'state'])

# Apply conversion function to monetary columns (handling potential errors)
try:
    data['Total Assets'] = data['Total Assets'].apply(convert_crore_to_float)
    data['Liabilities'] = data['Liabilities'].apply(convert_crore_to_float)
    data_test['Total Assets'] = data_test['Total Assets'].apply(convert_crore_to_float)
    data_test['Liabilities'] = data_test['Liabilities'].apply(convert_crore_to_float)
except:
    print("Error occurred during conversion. Check data format!")

# Drop columns (if desired)
drop_cols = ['Candidate', 'Constituency ∇', 'Education']  # Example columns to drop
X = data.drop(columns=drop_cols)
X_test = data_test.drop(columns=[ 'Candidate', 'Constituency ∇'])
y = data['Education']

# Define hyperparameters to tune
param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier()

# Perform grid search
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X, y)

# Get best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Predictions
y_pred = grid_search.predict(X_test)

# Convert y_pred to a DataFrame with 'ID' column from the test dataset
predictions_df = pd.DataFrame({'ID': data_test['ID'], 'Education': y_pred})

# Write predictions to a CSV file
predictions_df.to_csv('predictions_k-neighbour.csv', index=False)
