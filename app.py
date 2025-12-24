import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Adult Income Prediction App")
st.write("Predict whether a person's income is **>50K or <=50K** using a trained Decision Tree model.")

# -------------------------------
# Load Dataset (for encoders & scaler)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

X = df.drop(columns=["income"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# -------------------------------
# Fit Encoders & Scaler
# -------------------------------
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# -------------------------------
# Load Trained Model
# -------------------------------
with open("decision_tree_model-3.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


# -------------------------------
# User Input Section
# -------------------------------
st.header("🧑 Enter Person Details")

user_input = {}

for col in cat_cols:
    user_input[col] = st.selectbox(
        f"{col.replace('-', ' ').title()}",
        label_encoders[col].classes_
    )

for col in num_cols:
    user_input[col] = st.number_input(
        f"{col.replace('-', ' ').title()}",
        value=float(df[col].mean())
    )

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔍 Predict Income"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical features
    for col in cat_cols:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale numerical features
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Reorder columns to match training
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]

    if prediction == ">50K":
        st.success("💰 Income is predicted to be Greater than 50K")
    else:
        st.info("📉 Income is predicted to be Less than or Equal to 50K")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Machine Learning Project | Decision Tree Classifier")
