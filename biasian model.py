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

# Drop columns (if desired)
data['tot_revenue'] = data['Total Assets'] + data['Liabilities']
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

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

# Define parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [0.1,0.25, 0.5,0.75, 1.0],  # Laplace smoothing parameter
    # 'fit_prior': [True, False],  # Whether to learn class prior probabilities or not
    # 'class_prior': [None, [0.5, 0.5], [0.3, 0.7]],  # Class prior probabilities
    # 'normalize': [True, False]  # Whether to normalize feature counts or not
    # Add more hyperparameters to tune as needed
}

# Instantiate Multinomial Naive Bayes classifier
mnb_classifier = MultinomialNB()

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=mnb_classifier, param_grid=param_grid, scoring='f1_weighted', cv=cv)

# Fit GridSearchCV to training data
grid_search.fit(X, y)

# Get the best estimator
best_mnb_classifier = grid_search.best_estimator_

# Predictions on the test data using the best estimator
y_pred = best_mnb_classifier.predict(X_test)


predictions_df = pd.DataFrame({'ID': np.arange(len(y_pred)), 'Education': y_pred})

# Write predictions to a CSV file
predictions_df.to_csv('predictions_bayes.csv', index=False)

# Calculate F1 score for Multinomial Naive Bayes
# f1_mnb = f1_score(y_test, y_pred_mnb, average='weighted')

# print("Best hyperparameters:", grid_search.best_params_)
# print("F1 Score (Multinomial Naive Bayes with GridSearchCV):", f1_mnb)
