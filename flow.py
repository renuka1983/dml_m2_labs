from prefect import flow, task
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
@task
def ingest_data():
    df = pd.DataFrame({
        "age": [25, 32, 40, 50, 60],
        "bp": [120, 130, 140, 150, 160],
        "label": [0, 0, 1, 1, 1]
    })
    return df
@task
def validate_data(df: pd.DataFrame):
    if df.isnull().any().any():
        raise ValueError("❌ Missing values detected")

    if (df["age"] <= 0).any():
        raise ValueError("❌ Invalid age detected")
    print("✅ Validation passed")
    return df
@task
def data_preprocessing(df: pd.DataFrame):
    df = df.dropna()
    df['age'] = df['age'].fillna(df['age'].median())
    return df
@task
def train_model(df: pd.DataFrame):
    X = df[["age", "bp"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    return model, X_test, y_test
@task
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model Accuracy: {acc:.2f}")
    return acc
@task
def model_retraining_needed(acc, threshold=0.85):
    return acc < threshold
@flow
def ml_pipeline():
    df = ingest_data()
    df = validate_data(df)
    df = data_preprocessing(df)
    df = validate_data(df)
    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)
    acc = evaluate_model(model, X_test, y_test)
    if model_retraining_needed(acc):
        print("🔄 Retraining model...")
        model, X_test, y_test = train_model(df)
        evaluate_model(model, X_test, y_test)
    else: print("✅ Model is performing well, no retraining needed.")
if __name__ == "__main__":
    ml_pipeline()
