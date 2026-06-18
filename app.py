import streamlit as pd
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Page Configuration & Styling
st.set_page_config(page_title="GitHub Star Predictor AI", page_icon="🚀", layout="centered")

st.title("🚀 GitHub Star Predictor AI")
st.write("Adjust the sliders below to see our trained Machine Learning model predict how many stars a repository will get in real-time!")
st.write("---")

# 2. Behind the Scenes: Quick-Train the Model on Startup
@st.cache_data # This keeps the app running instantly fast
def load_and_train():
    df = pd.read_csv("github_data_cleaned.csv")
    features = ['forks', 'open_issues', 'size_kb', 'has_wiki', 'age_days']
    X = df[features]
    y = df['stars']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

model = load_and_train()

# 3. Create the Interactive UI Inputs (Sliders & Toggles)
st.subheader("📊 Repository Characteristics")

col1, col2 = st.columns(2)

with col1:
    forks = st.slider("Number of Forks (Copies)", min_value=0, max_value=100000, value=5000, step=250)
    open_issues = st.slider("Open Issues / Bugs", min_value=0, max_value=5000, value=50, step=10)
    size_kb = st.slider("Codebase Size (in KB)", min_value=100, max_value=50000, value=8000, step=500)

with col2:
    age_years = st.slider("Repository Age (in Years)", min_value=0.5, max_value=15.0, value=2.0, step=0.5)
    age_days = int(age_years * 365) # Convert back to days for our mathematical formula
    
    st.write("") # Just empty spacing to look clean
    has_wiki_bool = st.toggle("Repository Has Documentation Wiki?", value=True)
    has_wiki = 1 if has_wiki_bool else 0

st.write("---")

# 4. Use the Model to Predict Living UI Outputs!
# Match the order of features: ['forks', 'open_issues', 'size_kb', 'has_wiki', 'age_days']
user_features = [[forks, open_issues, size_kb, has_wiki, age_days]]
predicted_stars = model.predict(user_features)[0]

# Ensure the model doesn't predict negative stars for low parameters
if predicted_stars < 0:
    predicted_stars = 0

# 5. Display the Final Prediction Beautifully
st.subheader("🔮 AI Prediction Result")
st.metric(label="Predicted Star Count", value=f"{int(predicted_stars):,} Stars")

# Give the user context feedback based on what they picked
if predicted_stars > 100000:
    st.success("🔥 Wow! The AI predicts this repository will become an absolute viral superstar!")
elif predicted_stars > 25000:
    st.info("⭐ Strong metrics! This looks like a highly successful open-source project.")
else:
    st.warning("📈 Keep building! Increasing forks and maintaining active codebase age will raise its predicted popularity.")