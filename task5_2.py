import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train.csv")

print("Missing Values")
print(df.isnull().sum())

print("Filling Missing Value of Age with its median ")
df['Age']=df["Age"].fillna(df["Age"].median())
print(df.isnull().sum())

print("Filling Missing Embarked Value with Mode since it is a categorical data")
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin because of too many missing values
df= df.drop(columns=["Cabin"])
print(df.isnull().sum())


import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Select useful features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
target = "Survived"

X = df[features]
y = df[target]


# Separate numerical and categorical columns
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]


# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Model
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)


# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),("model", model)
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train
pipeline.fit(X_train, y_train)


# Evaluate
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save entire pipeline
joblib.dump(pipeline, "titanic_model.pkl")

print("\nModel saved as titanic_model.pkl")

# ============================================================
# 2. streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import joblib


# Load saved model
model = joblib.load("titanic_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢"
)


st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger information to predict whether the passenger "
    "would have survived the Titanic disaster."
)


# ============================================================
# INPUT FIELDS
# ============================================================

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("🟢 Prediction: Passenger would likely SURVIVE")
    else:
        st.error("🔴 Prediction: Passenger would likely NOT SURVIVE")

    st.write(
        f"Survival Probability: {probability[1] * 100:.2f}%"
    )

    st.write(
        f"Non-Survival Probability: {probability[0] * 100:.2f}%"
    )