import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB

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
# print(data.head())

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
# data['tot_revenue'] = data['Total Assets'] + data['Liabilities']
# data_test['tot_revenue'] = data_test['Total Assets'] + data_test['Liabilities']
drop_cols = ['ID', 'Candidate', 'Constituency ∇', 'Education', 'Total Assets', 'Liabilities']  # Example columns to drop
X = data.drop(columns=drop_cols)
# X_test = data_test.drop(columns=['ID', 'Candidate', 'Constituency ∇','Total Assets', 'Liabilities'])
y = data['Education']

# Encode categorical variables
X = pd.get_dummies(X, columns=['Party', 'state'])
# X_test = pd.get_dummies(X_test, columns=['Party', 'state'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)


model = BernoulliNB(alpha=0.5, binarize=0.1,fit_prior=True,class_prior=None)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# predictions_df = pd.DataFrame({'ID': np.arange(len(y_pred)), 'Education': y_pred})

# Write predictions to a CSV file
# predictions_df.to_csv('predictions_bayes_1.csv', index=False)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
# Calculate F1 score
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
# Latests scores
# Accuracy: 0.22572815533980584
# F1 Score: 0.20070918218498388

# Scores without totassets, liabilities and tort_rev.
# Accuracy: 0.22815533980582525
# F1 Score: 0.20146484382535346