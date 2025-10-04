from prefect import flow, task
import pandas as pd

@task
def ingest_data():
       df = pd.DataFrame({
           "age": [25, 32, 5, 40],
           "bp": [120, 130, 120, 140]
       })
       return df

@task
def validate_data(df: pd.DataFrame):
       if (df["age"] <= 0).any():
           raise ValueError("❌ Invalid age detected")
       if df.isnull().any().any():
           raise ValueError("❌ Missing values detected")
       print("✅ Data validation passed")
       return df

@flow
def validation_pipeline():
       df = ingest_data()
       validate_data(df)

if __name__ == "__main__":
       validation_pipeline()