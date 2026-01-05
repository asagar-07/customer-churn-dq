import pandas as pd
import numpy as np

cleansed_file_path = 'data/processed/customers_clean_for_model.csv'

df = pd.read_csv(cleansed_file_path)
print(df.head())

numeric_cols = df.select_dtypes(include=["number"]).columns.to_list()
print(numeric_cols)

for col in numeric_cols: 
    print(f"{col}: {'Infinities Present' if np.isinf(df[col]).any() else 'No Infinities'}")

print("Number of duplicate rows", df['is_duplicate'][df['is_duplicate']== 1].sum())
print("Number of unique rows", df['customerID'].nunique())

print(df.info())