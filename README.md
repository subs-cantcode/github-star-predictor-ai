# GitHub Star Predictor AI 🚀

A machine learning pipeline that fetches live repository data from the GitHub API, engineers dataset features, and trains a Linear Regression model to predict repository popularity (stars).

## Project Structure
* `fetch_data.py`: Connects to GitHub API and downloads the top 100 repositories.
* `clean_data.py`: Cleans raw strings and engineers a feature calculating repository age in days.
* `train_model.py`: Splits data, trains a Scikit-Learn Linear Regression model, and tests its accuracy.
* `app.py`: Launches an interactive Streamlit UI web app to test custom inputs against the AI model live.

## Performance
* **Model Type:** Linear Regression
* **R² Accuracy:** 0.58 (Explains 58% of the variance)
* **Sample Prediction:** A repo with 5,000 forks is predicted to reach ~61,000 stars!
