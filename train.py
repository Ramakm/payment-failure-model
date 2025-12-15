import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# -----------------------
# MLflow Setup
# -----------------------
print("Setting MLflow experiment...")
mlflow.set_experiment("payment_failure_prediction")
print("Enabling autolog...")
mlflow.sklearn.autolog()
print("MLflow setup done.")

# -----------------------
# Load data
# -----------------------
with open("userdata.json", "r") as f:
    raw_data = json.load(f)

df = pd.json_normalize(raw_data)

# -----------------------
# Select required fields
# -----------------------
df = df[
    [
        "occupation",
        "purposeOfTransaction",
        "sourceOfFunds",
        "countryOfBirth",
        "nationality",
        "idVerificationStatus",
        "receiver.address.countryCode",
        "dateOfBirth"
    ]
]

# -----------------------
# Feature engineering
# -----------------------
current_year = datetime.now().year

df["age"] = current_year - df["dateOfBirth"].str[:4].astype(int)
df["id_verified"] = df["idVerificationStatus"].map({"Y": 1, "N": 0})
df["cross_border"] = (
    df["countryOfBirth"] != df["receiver.address.countryCode"]
).astype(int)

df.drop(columns=["dateOfBirth", "idVerificationStatus"], inplace=True)

# -----------------------
# Create target variable
# -----------------------
def label_failure(row):
    if row["id_verified"] == 0 and row["sourceOfFunds"] == "Cash":
        return 1
    if row["cross_border"] == 1 and row["sourceOfFunds"] == "Cash":
        return 1
    if row["occupation"] == "worker" and row["id_verified"] == 0:
        return 1
    return 0

df["payment_failed"] = df.apply(label_failure, axis=1)

# -----------------------
# Train / Test split
# -----------------------
X = df.drop(columns=["payment_failed"])
y = df["payment_failed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Preprocessing + Model
# -----------------------
categorical_features = [
    "occupation",
    "purposeOfTransaction",
    "sourceOfFunds",
    "countryOfBirth",
    "nationality",
    "receiver.address.countryCode"
]

numeric_features = ["age", "id_verified", "cross_border"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

# -----------------------
# Train model
# -----------------------
with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    # -----------------------
    # Evaluate
    # -----------------------
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # -----------------------
    # Save model
    # -----------------------
    # MLflow autolog saves the model, but we also stick to the local file for the app
    joblib.dump(pipeline, "payment_failure_model.pkl")
    print("Model saved as payment_failure_model.pkl")
