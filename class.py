import pandas as pd
import numpy as np

# Load the original training dataset
data_train = pd.read_csv('train.csv')

# Display unique values in the 'Education' column
unique_classes = data_train['Education'].unique()
print("Unique classes and their corresponding IDs:")
for idx, class_label in enumerate(unique_classes):
    print(f"ID {idx}: {class_label}")


# ID 0: 8th Pass
# ID 1: 12th Pass
# ID 2: Post Graduate
# ID 3: Graduate Professional
# ID 4: Graduate
# ID 5: 10th Pass
# ID 6: Others
# ID 7: Doctorate
# ID 8: Literate
# ID 9: 5th Pass