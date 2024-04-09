import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer  # For missing value imputation


# Function to convert monetary values to numeric
def convert_crore_to_float(crore_str):
    """
    This function converts a string in crore format (e.g., '32 Crore+') to a float value.

    Args:
        crore_str: The string representation of a number in crore format.

    Returns:
        The float value of the number in crore, or NaN if the format is invalid.
    """
    if crore_str == '0':
        return 0
    crore_str = crore_str.strip()  # Remove leading/trailing spaces
    try:
        number = float(crore_str.split(" ")[0])  # Extract the number before 'Crore'
        unit = crore_str.split(" ")[-1].lower()  # Get the unit (Crore or Crore+)

        unit_multiplier = {
            "crore": 10000000,
            "lac": 10000,
            "thou": 1000,
            "hund": 1000,
        }
        return number * unit_multiplier.get(unit, np.nan)  # Use get with default of NaN
    except ValueError:
        return np.nan  # Return NaN for invalid formats


# Load the dataset
data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
# Explore the data
print(data.head())

# Check for missing values
# print(data.isnull().sum())

# # Handle missing values (example using SimpleImputer)
# imputer = SimpleImputer(strategy='mean')  # Replace missing values with mean
# data = pd.DataFrame(imputer.fit_transform(data))  # Impute missing values
# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()
data_test.columns = data_test.columns.str.strip()
# Encode categorical variables
data = pd.get_dummies(data, columns=['Party', 'state'])
data_test = pd.get_dummies(data_test, columns=['Party', 'state'])
# Split features and target variable

# Apply conversion function to monetary columns (handling potential errors)
try:
    data['Total Assets'] = data['Total Assets'].apply(convert_crore_to_float)
    data['Liabilities'] = data['Liabilities'].apply(convert_crore_to_float)
    data_test['Total Assets'] = data_test['Total Assets'].apply(convert_crore_to_float)
    data_test['Liabilities'] = data_test['Liabilities'].apply(convert_crore_to_float)
except:
    print("Error occurred during conversion. Check data format!")

# Drop columns (if desired)
drop_cols = ['ID', 'Candidate', 'Constituency ∇', 'Education']  # Example columns to drop
X_train = data.drop(columns=drop_cols)
X_test = data_test.drop(columns=['ID', 'Candidate', 'Constituency ∇'])
y_train = data['Education']
# Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

# Define parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
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
predictions_df.to_csv('predictions3.csv', index=False)

# Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# Visualize feature importance
# plt.figure(figsize=(10, 6))
# feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns)
# feature_importance.nlargest(10).plot(kind='barh')
# plt.xlabel('Feature Importance')
# plt.ylabel('Features')
# plt.title('Top 10 Important Features')
# plt.show()
