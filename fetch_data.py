print("--- STARTING ADVANCED DATA FETCH ---")

import requests
import pandas as pd
import os
from dotenv import load_dotenv

# 1. Load the token from your .env file
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"}

url = "https://api.github.com/search/repositories"

# 2. We change per_page to 100 and look for highly-starred Python repos
params = {
    "q": "language:python stars:>1000", 
    "sort": "stars",
    "order": "desc",
    "per_page": 100 
} 

print("Sending request to GitHub API for 100 records...")
response = requests.get(url, headers=HEADERS, params=params)

if response.status_code == 200:
    data = response.json()
    items = data["items"]
    
    # 3. Extracting 8 deep features for machine learning
    repo_list = []
    for repo in items:
        repo_list.append({
            "name": repo["name"],
            "language": repo["language"],
            "stars": repo["stargazers_count"],     # Our Target Variable
            "forks": repo["forks_count"],
            "open_issues": repo["open_issues_count"],
            "size_kb": repo["size"],
            "has_wiki": repo["has_wiki"],
            "created_at": repo["created_at"]
        })
    
    # 4. Turn into a DataFrame and overwrite the old CSV file
    df = pd.DataFrame(repo_list)
    df.to_csv("github_data.csv", index=False)
    print("Success! Advanced dataset saved to github_data.csv")
    print(f"Dataset shape: {df.shape} (Rows, Columns)")
else:
    print(f"Something went wrong! Error code: {response.status_code}")