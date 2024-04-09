import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Function to convert monetary values to numeric
def convert_crore_to_float(crore_str):
    """
    This function converts a string in crore format (e.g., '32 Crore+') to a float value.

    Args:
        crore_str: The string representation of a number in crore format.

    Returns:
        The float value of the number in crore, or NaN if the format is invalid.
    """
    crore_str = crore_str.strip().lower()  # Remove leading/trailing spaces and convert to lowercase
    if isinstance(crore_str, float) and np.isnan(crore_str):
        return np.nan  # Return NaN if already NaN
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

# Apply conversion function to monetary columns (handling potential errors)
try:
    data['Total Assets'] = data['Total Assets'].apply(convert_crore_to_float)
    data['Liabilities'] = data['Liabilities'].apply(convert_crore_to_float)
    data_test['Total Assets'] = data_test['Total Assets'].apply(convert_crore_to_float)
    data_test['Liabilities'] = data_test['Liabilities'].apply(convert_crore_to_float)
except:
    print("Error occurred during conversion. Check data format!")

# Encode categorical variables
data = pd.get_dummies(data, columns=['Party', 'state'])
data_test = pd.get_dummies(data_test, columns=['Party', 'state'])

# Drop columns (if desired)
drop_cols = ['ID', 'Candidate', 'Constituency ∇', 'Education']  # Example columns to drop
X = data.drop(columns=drop_cols)
X_test = data_test.drop(columns=['ID', 'Candidate', 'Constituency ∇'])
y = data['Education']

# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

# Set the best parameters obtained from grid search
best_params ={'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 100}

# Set the best estimator with the best parameters
rf_classifier.set_params(**best_params)

# Fit the model on the entire training data
rf_classifier.fit(X, y)

# Predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Convert predictions to a DataFrame with 'ID' column from the test dataset
predictions_df = pd.DataFrame({'ID': data_test['ID'], 'Education': y_pred})

# Write predictions to a CSV file
predictions_df.to_csv('predictions_randomforest.csv', index=False)
