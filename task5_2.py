import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# TRAIN MODEL IF IT DOES NOT ALREADY EXIST
# ============================================================

MODEL_FILE = "titanic_model.pkl"


def train_model():

    # Load Titanic dataset
    # IMPORTANT: This CSV must also be uploaded to GitHub
    df = pd.read_csv("train.csv")

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Drop Cabin
    df = df.drop(columns=["Cabin"])

    # Features and target
    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]

    target = "Survived"

    X = df[features]
    y = df[target]

    # Numerical features
    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    # Categorical features
    categorical_features = [
        "Sex",
        "Embarked"
    ]

    # Numerical preprocessing
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Complete pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
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

    # Save model
    joblib.dump(pipeline, MODEL_FILE)

    return pipeline


# ============================================================
# LOAD OR TRAIN MODEL
# ============================================================

if os.path.exists(MODEL_FILE):

    model = joblib.load(MODEL_FILE)

else:

    model = train_model()


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢"
)

st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger information to predict whether "
    "the passenger would have survived."
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

        st.success(
            "🟢 Passenger would likely SURVIVE"
        )

    else:

        st.error(
            "🔴 Passenger would likely NOT SURVIVE"
        )

    st.write(
        f"Survival Probability: "
        f"{probability[1] * 100:.2f}%"
    )

    st.write(
        f"Non-Survival Probability: "
        f"{probability[0] * 100:.2f}%"
    )