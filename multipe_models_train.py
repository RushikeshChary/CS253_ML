import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

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
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [30, 50, 70],
}

# param_grid_svc = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf', 'poly'],
# }

# Initialize classifiers
rf_classifier = RandomForestClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)
# svc_classifier = SVC(probability=True, random_state=42)

classifiers = {
    'Random Forest': (rf_classifier, param_grid_rf),
    'Gradient Boosting': (gb_classifier, param_grid_gb),
    # 'Support Vector Classifier': (svc_classifier, param_grid_svc),
}

from sklearn.metrics import accuracy_score

# Split the training dataset into new training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Perform grid search for each classifier using the new training and validation sets
best_classifiers = {}
for name, (classifier, param_grid) in classifiers.items():
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_split, y_train_split)
    best_classifiers[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")

# Evaluate the accuracy on the validation set for each classifier
for name, classifier in best_classifiers.items():
    y_pred_val = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Accuracy for {name}: {accuracy}")

# Retrain each classifier using the entire training dataset
for name, classifier in best_classifiers.items():
    classifier.fit(X_train, y_train)  # Retrain on the entire training dataset

# Make predictions on the original test dataset using the retrained classifiers
predictions = []
for name, classifier in best_classifiers.items():
    predictions.append(classifier.predict_proba(X_test))

# Average the predictions
avg_predictions = np.mean(predictions, axis=0)

# Convert predictions to classes
y_pred = np.argmax(avg_predictions, axis=1)

# Map class indices to class labels
class_mapping = {
    0: '8th Pass',
    1: '12th Pass',
    2: 'Post Graduate',
    3: 'Graduate Professional',
    4: 'Graduate',
    5: '10th Pass',
    6: 'Others',
    7: 'Doctorate',
    8: 'Literate',
    9: '5th Pass'
}

# Convert class indices to class labels
predicted_labels = [class_mapping[idx] for idx in y_pred]

# Calculate overall accuracy
accuracy = accuracy_score(y_val, y_pred_val)
print(f"Overall Accuracy: {accuracy}")
