import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Crypto Market Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('crypto_market_daily_snapshot.csv')
df = load_data()

st.title("Crypto Market Dashboard")

st.sidebar.header("Controls")
target_col = st.sidebar.selectbox("Select Prediction Target", df.columns)

features = st.sidebar.multiselect(
"Select Feature Columns",
[c for c in df.columns if c != target_col],
default=[c for c in df.columns if c != target_col][:3]
)

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Basic Statistics")
st.write(df.describe())

st.subheader("Market Trend Visualization")
plot_col = st.selectbox("Choose Column to Plot", df.columns)

fig, ax = plt.subplots()
ax.plot(df[plot_col])
ax.set_title(plot_col)
st.pyplot(fig)

st.subheader("Train AI Model")

if len(features) > 0:
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.success("Model trained successfully!")
    st.write("MAE:", mean_absolute_error(y_test, preds))
    st.write("R2 Score:", r2_score(y_test, preds))

    # Feature importance
    st.subheader("Feature Importance")
    imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))

# Prediction Interface
st.subheader("Make Live Prediction")

input_data = {}
for col in features:
    input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted {target_col}: {prediction}")

st.markdown("---")
st.caption("Built with Python + Streamlit + Machine Learning")