# train_income_model.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    # 1. Load dataset
    df = pd.read_csv('IBM_Attrition.csv')

    # 2. Drop useless/constant columns and target Attrition, YearsAtCompany
    df.drop([
        'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours',
        'Attrition', 'YearsAtCompany'
    ], axis=1, inplace=True)

    # 3. Define target and feature set
    target = 'MonthlyIncome'
    features = [
        'OverTime',
        'Age',
        'TotalWorkingYears',
        'StockOptionLevel',
        'WorkLifeBalance',
        'JobRole',
        'EducationField'
    ]
    X = df[features]
    y = df[target]

    # 4. Encode categorical features
    cat_cols = ['OverTime', 'JobRole', 'EducationField']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # 5. Scale numerical features
    num_cols = ['Age', 'TotalWorkingYears', 'StockOptionLevel', 'WorkLifeBalance']
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # 6. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 8. Evaluate model
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R² : {r2:.3f}")

    # 9. Save model and preprocessors
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/income_rf_model.pkl')
    joblib.dump(scaler, 'models/income_scaler.pkl')
    joblib.dump(label_encoders, 'models/income_label_encoders.pkl')
    print("Models and preprocessors saved to ./models/")

if __name__ == "__main__":
    main()
