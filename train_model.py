print("--- STARTING MACHINE LEARNING TRAINING ---")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load our engineered dataset
df = pd.read_csv("github_data_cleaned.csv")

# 2. Define Features (X) and Target (y)
# X contains the clues, y contains what we want to predict (stars)
features = ['forks', 'open_issues', 'size_kb', 'has_wiki', 'age_days']
X = df[features]
y = df['stars']

# 3. Split data into Training Set (80%) and Testing Set (20%)
# This allows us to test the model on data it hasn't seen before!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training with {len(X_train)} samples, Testing with {len(X_test)} samples.")

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete!")

# 5. Evaluate the model using the Test Set
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- MODEL PERFORMANCE METRICS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} stars")
print(f"R-squared Score (R²): {r2:.2f}")
print("Note: An R² closer to 1.0 means the model is highly accurate.")

# 6. Let's Use Our Model to Predict a New, Hypothetical Repository!
# Imagine you just launched a repository with:
# 5,000 forks, 50 open issues, 8,000 KB size, has a wiki (1), and is 365 days old
hypothetical_repo = [[5000, 50, 8000, 1, 365]]
predicted_stars = model.predict(hypothetical_repo)

print("\n--- MAKING A NEW PREDICTION ---")
print(f"For a repo with 5,000 forks & 50 open issues, our AI predicts it will reach: {int(predicted_stars[0]):,} stars!")