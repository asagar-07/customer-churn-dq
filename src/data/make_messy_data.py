import pandas as pd
import numpy as np

df = pd.read_csv('/Users/abila/Documents/Coding Projects/my ML Engineering Projects 2025/Project 1 - Customer Churn Prediction + Data Quality/customer-churn-dq/data/raw/telco-Customer-Churn.csv')


# function to filter random rows based on a given percentage
def filter_random_rows(df, percentage, random_state=42):
    n_rows = int(len(df) * percentage / 100)
    return df.sample(n=n_rows, random_state=random_state).reset_index(drop=True)

# function to introduce missing values randomly on a random_state in a dataframe based on a given percentage
def introduce_missing_values(df, col_name, percentage, random_state=42):
    df_copy = df.copy()
    n_rows = len(df_copy)
    np.random.seed(random_state)

    col_mask = np.random.choice([True, False], size=n_rows, p=[percentage / 100, 1 - percentage / 100])
    df_copy.loc[col_mask, col_name] = np.nan

    return df_copy

# function to introduce UNKNOWN randomly on a random_state in a dataframe based on a given percentage
unknown_values = ['UNKNOWN', 'Unknown', 'N/A', 'na', 'n/a','NOT APPLICABLE', 'Not Applicable','Not provided']

def introduce_unknown_values(df, col_name, unknown_values, percentage, random_state=42):
    np.random.seed(random_state)
    # Get the indices of the rows to modify
    df_copy = df.copy()
    n = df_copy.shape[0]
    n_modify = int(n * percentage/100)
    modify_indices = np.random.choice(n, size=n_modify, replace=False)

    # Introduce unknown values
    for idx in modify_indices:
        df_copy.at[idx, col_name] = np.random.choice(unknown_values)

    return df_copy

# dictionary for introducting inconsistent, typos values to each column
inconsistent_values = {
    'Female': ['F', 'f', '0'],
    'Male': ['M', 'm', '1'],
    'Yes': ['Y', 'y', '1'],
    'No': ['N', 'n', '0'],
    'No phone service': ['No Phone Service', 'no phone service', 'NPS'],
    'No internet service': ['No Internet Service', 'no internet service', 'NIS'],
    'Month-to-month': ['Monthly', 'month to month','m2m' ,'M2M'],
    'One year': ['1 year', 'One Yr', '1 Yr'],
    'Two year': ['2 year', 'Two Yr', '2 Yr'],
    'Fiber optic': ['Fiber', 'fiber', 'FO', 'Fiber-Optik'],
    'Electronic check': ['E-check', 'E Check', 'Electronic Payment'],
    'Mailed check': ['Mailed Check', 'Check by Mail','Paper Check'],
    'Bank transfer (automatic)': ['Bank Transfer', 'Direct Debit'],
    'Credit card (automatic)': ['Credit Card', 'CC']
}
   
#function to introduce inconsistent values randomly on a random_state in a dataframe based on a given percentage
def introduce_inconsistent_values(df, col_name, inconsistent_values, percentage, random_state=42):
    np.random.seed(random_state)
    df_copy = df.copy()
    n = df_copy.shape[0]
    n_modify = int(n * percentage / 100)
    modify_indices = np.random.choice(n, size=n_modify, replace=False)

    for idx in modify_indices:
        original_value = df_copy.at[idx, col_name]
        if original_value in inconsistent_values:
            new_value = np.random.choice(inconsistent_values[original_value])
            df_copy.at[idx, col_name] = new_value

    return df_copy

#function to add white spaces to cell value either suffix or prefix on a random_state in a dataframe based on a given percentage
def introduce_whitespaces(df, col_name, percentage, random_state=42):
    np.random.seed(random_state)
    df_copy = df.copy()
    n = df_copy.shape[0]
    n_modify = int(n * percentage / 100)
    modify_indices = np.random.choice(n, size=n_modify, replace=False)

    for idx in modify_indices:
        original_value = df_copy.at[idx, col_name]
        if isinstance(original_value, str):
            rand = np.random.rand()
            if rand < 0.33:
                new_value = ' ' + original_value  # Prefix whitespace
            elif rand < 0.66:
                new_value = original_value + ' '  # Suffix whitespace
            else:
                new_value = ' ' + original_value + ' '  # Both prefix and suffix whitespace
            df_copy.at[idx, col_name] = new_value

    return df_copy

#function to introduce numeric outliers randomly on a random_state in a dataframe based on a given percentage
wrong_numeric_values = [-9999, 9999, -1, 1000000,-30, 1e+309, '100', "Nan", 0.0, 0.39, 0.000001, "ten", "thirty"]

def introduce_numeric_outliers(df, col_name, percentage, random_state=42):
    np.random.seed(random_state)
    # Get the indices of the rows to modify
    df_copy = df.copy()
    n = df_copy.shape[0]
    n_modify = int(n * percentage/100)
    df_copy[col_name] = df_copy[col_name].astype('object')
    modify_indices = np.random.choice(n, size=n_modify, replace=False)
    
    # Introduce outlier+invalid values
    for idx in modify_indices:
        df_copy.at[idx, col_name] = np.random.choice(wrong_numeric_values)
    
    return df_copy

#function to introduce_label_noise randomly on a random_state in a dataframe based on a given percentage
def introduce_label_noise(df, col_name, percentage, random_state=42):
    np.random.seed(random_state)
    df_copy = df.copy()
    n = df_copy.shape[0]
    n_modify = int(n * percentage / 100)
    modify_indices = np.random.choice(n, size=n_modify, replace=False)

    for idx in modify_indices:
        original_value = df_copy.at[idx, col_name]
        if original_value == 'Yes':
            df_copy.at[idx, col_name] = 'No'
        elif original_value == 'No':
            df_copy.at[idx, col_name] = 'Yes'

    return df_copy


#evaluate the impact of the introduced noise.
def evaluate_noise_impact(original_df, noisy_df, col_name):
    original_counts = original_df[col_name].value_counts()
    noisy_counts = noisy_df[col_name].value_counts()
    comparison_df = pd.DataFrame({'Original': original_counts, 'Noisy': noisy_counts})
    comparison_df.fillna(0, inplace=True)
    return comparison_df

#function to introduce logical_inconsistencies randomly in a dependent column on a random_state in a dataframe based on a given percentage
def introduce_logical_inconsistencies(df, col_name, dependent_col_name, percentage, random_state=42):
    np.random.seed(random_state)
    df_copy = df.copy()
    n = df_copy.shape[0]
    no_indices = [idx for idx in range(n) if df_copy.at[idx, col_name] == 'No']

    if not no_indices:
        return df_copy  # No 'No' values to modify
    
    n_modify = int(len(no_indices) * percentage / 100)
    if n_modify == 0:
        return df_copy  # No rows to modify based on the percentage
    
    modify_indices = np.random.choice(len(no_indices), size=n_modify, replace=False)

    for idx in modify_indices:
        row_idx = no_indices[idx]              # actual row index in df_copy
        df_copy.at[row_idx, dependent_col_name] = 'Yes'

    return df_copy


#Messiness Parameters for both Duplicate Rows DataFrame and Raw DataFrame
duplicate_rows_percentage = 15
whitespace_percentage_dup = 10
whitespace_percentage_w = 2
outliers_percentage_dup = 10
outliers_percentage_w = 3
logical_inconsistent_percentage_dup = 10
logical_inconsistent_percentage_w = 3
label_noise_percentage_dup = 10
label_noise_percentage_w = 2
NaN_values_percentage_w = 5
NaN_values_percentage_dup = 20
unknown_values_percentage_w = 3
unknown_values_percentage_dup = 15
drop_customer_id_percentage_w = 2
drop_customer_id_percentage_dup = 10
inconsistent_values_percentage_w = 2
inconsistent_values_percentage_dup = 20

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')

df_minus_pred_ID = df.drop(columns=['customerID','Churn']).copy()

num_columns = df_minus_pred_ID.select_dtypes(include=['number']).columns.tolist()
cat_columns = df_minus_pred_ID.select_dtypes(include=['object']).columns.tolist()
#print("num_columns, length:", num_columns, len(num_columns))
#print("cat_columns, length:", cat_columns, len(cat_columns))

dependent_column_pair = {
    'InternetService': ['StreamingTV', 'StreamingMovies', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport'],
    'PhoneService': ['MultipleLines']
}
    
#Apply Messiness to Duplicate Rows DataFrame
duplicate_df = filter_random_rows(df, duplicate_rows_percentage, random_state=21)
duplicate_df_minus_pred_ID = duplicate_df.drop(columns=['customerID','Churn']).copy()

base_whitespace_seed = 1
for i, col in enumerate(cat_columns):
    col_seed = base_whitespace_seed + i
    duplicate_df_minus_pred_ID = introduce_whitespaces(duplicate_df_minus_pred_ID, col, whitespace_percentage_dup, random_state=col_seed)

base_numeric_outliers_seed = 101
for j, col in enumerate(num_columns):
    col_seed = base_numeric_outliers_seed + j
    duplicate_df_minus_pred_ID = introduce_numeric_outliers(duplicate_df_minus_pred_ID, col, outliers_percentage_dup, random_state=col_seed)


base_logical_inconsistencies_seed = 201
for i, (key, val) in enumerate(dependent_column_pair.items()):
    col_seed = base_logical_inconsistencies_seed + i
    for v in val:
        duplicate_df_minus_pred_ID = introduce_logical_inconsistencies(duplicate_df_minus_pred_ID, key, v, logical_inconsistent_percentage_dup, random_state=col_seed)

base_inconsistent_values_seed = 301
for k, col in enumerate(cat_columns):
    col_seed = base_inconsistent_values_seed + k
    duplicate_df_minus_pred_ID = introduce_inconsistent_values(duplicate_df_minus_pred_ID, col, inconsistent_values, inconsistent_values_percentage_dup, random_state=col_seed)

base_missing_values_seed = 401  
for i, col in enumerate(num_columns):
    col_seed = base_missing_values_seed + i
    duplicate_df_minus_pred_ID = introduce_missing_values(duplicate_df_minus_pred_ID, col, NaN_values_percentage_dup, random_state=col_seed)

base_unknown_values_seed = 501
for j, col in enumerate(cat_columns):
    col_seed = base_unknown_values_seed + j
    duplicate_df_minus_pred_ID = introduce_unknown_values(duplicate_df_minus_pred_ID, col, unknown_values, unknown_values_percentage_dup, random_state=col_seed)

#reingesting customer ID column and Churn column for duplicate_df_minus_pred_ID
duplicate_df_minus_pred_ID['customerID'] = duplicate_df['customerID']
duplicate_df_minus_pred_ID['Churn'] = duplicate_df['Churn']

base_label_noise_seed = 601
duplicate_df_minus_pred_ID = introduce_label_noise(duplicate_df_minus_pred_ID, 'Churn', label_noise_percentage_dup, random_state=base_label_noise_seed) 

base_label_inconsistent_values_seed = 701
duplicate_df_minus_pred_ID = introduce_inconsistent_values(duplicate_df_minus_pred_ID, 'Churn', inconsistent_values, inconsistent_values_percentage_dup, random_state=base_label_inconsistent_values_seed)


# Apply Messiness to Raw DataFrame
base_whitespace_seed = 5
for i, col in enumerate(cat_columns):
    col_seed = base_whitespace_seed + i
    df_minus_pred_ID = introduce_whitespaces(df_minus_pred_ID, col, whitespace_percentage_w, random_state=col_seed)

base_numeric_outliers_seed = 51
for j, col in enumerate(num_columns):
    col_seed = base_numeric_outliers_seed + j
    df_minus_pred_ID = introduce_numeric_outliers(df_minus_pred_ID, col, outliers_percentage_w, random_state=col_seed)

base_logical_inconsistencies_seed = 151
for i, (key, val) in enumerate(dependent_column_pair.items()):
    col_seed = base_logical_inconsistencies_seed + i
    for v in val:
        df_minus_pred_ID = introduce_logical_inconsistencies(df_minus_pred_ID, key, v, logical_inconsistent_percentage_w, random_state=col_seed)

base_inconsistent_values_seed = 251
for k, col in enumerate(cat_columns):
    col_seed = base_inconsistent_values_seed + k
    df_minus_pred_ID = introduce_inconsistent_values(df_minus_pred_ID, col, inconsistent_values, inconsistent_values_percentage_w, random_state=col_seed)

base_missing_values_seed = 351
for i, col in enumerate(num_columns):
    col_seed = base_missing_values_seed + i
    df_minus_pred_ID = introduce_missing_values(df_minus_pred_ID, col, NaN_values_percentage_w, random_state=col_seed)

base_unknown_values_seed = 451
for j, col in enumerate(cat_columns):
    col_seed = base_unknown_values_seed + j
    df_minus_pred_ID = introduce_unknown_values(df_minus_pred_ID, col, unknown_values, unknown_values_percentage_w, random_state=col_seed)

# reingesting customer ID column and Churn column for df_minus_pred_ID
df_minus_pred_ID['customerID'] = df['customerID']
df_minus_pred_ID['Churn'] = df['Churn']

base_label_noise_seed = 551
df_minus_pred_ID = introduce_label_noise(df_minus_pred_ID, 'Churn', label_noise_percentage_w, random_state=base_label_noise_seed)

base_label_inconsistent_values_seed = 651
df_minus_pred_ID = introduce_inconsistent_values(df_minus_pred_ID, 'Churn', inconsistent_values, inconsistent_values_percentage_w, random_state=base_label_inconsistent_values_seed)


#Final Messy DataFrame
df_messy = pd.concat([df_minus_pred_ID, duplicate_df_minus_pred_ID], ignore_index=True)
df_messy.to_csv('/Users/abila/Documents/Coding Projects/my ML Engineering Projects 2025/Project 1 - Customer Churn Prediction + Data Quality/customer-churn-dq/data/processed/telco-Customer-Churn-messy-data.csv', index=False)