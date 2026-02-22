import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Diabetes Risk Estimator", layout="wide")

st.title("🩺 Diabetes Risk Estimator")
st.write("Predicting diabetes progression using Machine Learning")

# -------------------------
# Load Dataset
# -------------------------
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider("Test Size (%)", 10, 40, 25)

model_option = st.sidebar.selectbox(
    "Choose Model",
    ("Linear Regression", "Ridge Regression")
)

alpha = 1.0
if model_option == "Ridge Regression":
    alpha = st.sidebar.slider("Ridge Alpha", 0.1, 10.0, 1.0)

# -------------------------
# Train Model Button
# -------------------------
if st.sidebar.button("Train Model"):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    if model_option == "Linear Regression":
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # -------------------------
    # Display Metrics
    # -------------------------
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")

    # -------------------------
    # Visualization (Only ONE plot to make it different)
    # -------------------------
    st.subheader("📈 Actual vs Predicted")

    fig, ax = plt.subplots()

    ax.scatter(y_test, predictions, alpha=0.7)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted Diabetes Progression")

    st.pyplot(fig)

    # -------------------------
    # Feature Importance Table
    # -------------------------
    st.subheader("🔎 Feature Coefficients")

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    st.dataframe(coef_df)
