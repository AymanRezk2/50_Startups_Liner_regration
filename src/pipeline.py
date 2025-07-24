import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import joblib

def main():
    # 1. Load dataset
    df = pd.read_csv("data/50_Startups.csv")
    
    # 2. Split features and target
    X = df.drop("Profit", axis=1)
    y = df[["Profit"]]  # keep as DataFrame
    
    # 3. Split train/test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Identify numeric & categorical columns
    num_cols = x_train.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = x_train.select_dtypes(include=["object"]).columns
    
    # 5. Build pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
    
    # 6. Train model
    pipeline.fit(x_train, y_train)
    
    # 7. Predict and evaluate
    y_pred = pipeline.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    print("Mean Absolute Error:", mae)

    # 8. Save the pipeline
    import joblib
    joblib.dump(pipeline, "./model_pipeline.pkl")
    print("Pipeline saved as model_pipeline.pkl")

if __name__ == "__main__":
    main()
