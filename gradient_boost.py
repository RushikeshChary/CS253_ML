import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Load the dataset
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Remove leading/trailing spaces from column names
data_train.columns = data_train.columns.str.strip()
data_test.columns = data_test.columns.str.strip()

# Apply conversion function to monetary columns 
try:
    data_train['Total Assets'] = data_train['Total Assets'].apply(convert_crore_to_float)
    data_train['Liabilities'] = data_train['Liabilities'].apply(convert_crore_to_float)
    data_test['Total Assets'] = data_test['Total Assets'].apply(convert_crore_to_float)
    data_test['Liabilities'] = data_test['Liabilities'].apply(convert_crore_to_float)
except:
    print("Error occurred during conversion. Check data format!")

# Encode categorical variables
data_train = pd.get_dummies(data_train, columns=['Party', 'state'])
data_test = pd.get_dummies(data_test, columns=['Party', 'state'])

# Drop columns (if desired)
drop_cols = ['ID', 'Candidate', 'Constituency ∇', 'Education']
X_train = data_train.drop(columns=drop_cols)
y_train = data_train['Education']
X_test = data_test.drop(columns=['ID', 'Candidate', 'Constituency ∇'])

# Define parameter grid for grid search
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.4],
    'max_depth': [3, 5, 7],
}

# Initialize Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid_gb, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best classifier
best_classifier = grid_search.best_estimator_
print(f"Best parameters for Gradient Boosting: {grid_search.best_params_}")

# Make predictions on the test set
y_pred = best_classifier.predict(X_test)

# Save predictions to a CSV file
predictions_df = pd.DataFrame({'ID': data_test['ID'], 'Education': y_pred})
predictions_df.to_csv('predictions_test.csv', index=False)
