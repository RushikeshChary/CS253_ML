import pandas as pd

def compare_column(file1, file2, column_name):
    # Read CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if column exists in both DataFrames
    if column_name not in df1.columns or column_name not in df2.columns:
        print("Column '{}' not found in both files.".format(column_name))
        return

    # Extract the column data
    column1 = df1[column_name]
    column2 = df2[column_name]

    # Find similarities
    similarities = column1[column1.isin(column2)]

    if similarities.empty:
        print("No similarities found in column '{}'.".format(column_name))
    else:
        print("Similarities found in column '{}':".format(column_name))
        print(similarities)

# Example usage
compare_column('train.csv', 'test.csv', 'Candidate')
