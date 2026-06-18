print("---STARTING THE INSPECTION OF THE DATASET---")

import pandas as pd
from datetime import datetime

#loading the dataset
df = pd.read_csv("github_data.csv")
print(f"Original shape:{df.shape}")

#converting has_wiki into 1 and 0 to make it easier for the machine learning model to understand 
df["has_wiki"] = df["has_wiki"].astype(int)

#creating a column to convert into actual date_time python object
df['created_at'] = pd.to_datetime(df['created_at'])

#getting today's date
today = pd.Timestamp.now(tz='UTC')

# Calculate the difference in days
df['age_days'] = (today - df['created_at']).dt.days

# 4. Let's see our new columns
print("\nFirst 3 rows of our engineered dataset:")
print(df[['name', 'stars', 'forks', 'has_wiki', 'age_days']].head(3))

# 5. Save this cleaned version to a new file so we don't overwrite the raw data
df.to_csv("github_data_cleaned.csv", index=False)
print("\nSuccess! Cleaned dataset saved to github_data_cleaned.csv")
