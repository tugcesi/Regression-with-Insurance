"""
save_model.py - Insurance Premium Model Training
Run: python save_model.py
Output: model.joblib, feature_columns.joblib, num_medians.joblib,
        cat_modes.joblib, ohe_categories.joblib
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def preprocess(df: pd.DataFrame, num_medians: dict, cat_modes: dict,
               prev_claims_median: float, nominal_cols: list,
               ohe_categories: dict) -> pd.DataFrame:
    df = df.copy()

    # 2. Missing value flag
    df['Previous_Claims_Missing'] = df['Previous Claims'].isna().astype(int)

    # 3. Numeric imputation
    num_cols = list(num_medians.keys())
    for col in num_cols:
        df[col] = df[col].fillna(num_medians[col])

    # 4. Categorical imputation
    df['Previous Claims'] = df['Previous Claims'].fillna(prev_claims_median)
    df['Marital Status'] = df['Marital Status'].fillna(cat_modes['Marital Status'])
    df['Customer Feedback'] = df['Customer Feedback'].fillna(cat_modes['Customer Feedback'])
    df['Occupation'] = df['Occupation'].fillna('Unknown')

    # 5. Policy Start Date → date features, then drop
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Policy_Year'] = df['Policy Start Date'].dt.year
    df['Policy_Month'] = df['Policy Start Date'].dt.month
    df['Policy_DayOfWeek'] = df['Policy Start Date'].dt.dayofweek
    df['Policy_Age_Days'] = (pd.Timestamp('2026-04-19') - df['Policy Start Date']).dt.days
    df = df.drop(columns=['Policy Start Date'])

    # 6. Feature engineering
    df['Log_Annual_Income'] = np.log1p(df['Annual Income'])
    df['Income_Per_Person'] = df['Annual Income'] / (df['Number of Dependents'] + 1)
    df['Claims_Per_Year'] = df['Previous Claims'] / df['Insurance Duration']
    df['Age_Health_Interaction'] = df['Age'] * df['Health Score']
    df['Income_Credit_Ratio'] = df['Annual Income'] / (df['Credit Score'] + 1)
    df['Total_Risk_Duration'] = df['Vehicle Age'] + df['Insurance Duration']

    # 7. Binary encoding
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Smoking Status'] = df['Smoking Status'].map({'No': 0, 'Yes': 1})

    # 8. Ordinal encoding
    education = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    feedback = {'Poor': 0, 'Average': 1, 'Good': 2}
    exercise = {'Rarely': 0, 'Monthly': 1, 'Weekly': 2, 'Daily': 3}
    df['Education Level'] = df['Education Level'].map(education)
    df['Customer Feedback'] = df['Customer Feedback'].map(feedback)
    df['Exercise Frequency'] = df['Exercise Frequency'].map(exercise)

    # 9. One-Hot Encoding
    for col in nominal_cols:
        cats = ohe_categories[col]
        for cat in cats[1:]:  # drop_first=True equivalent
            df[f'{col}_{cat}'] = (df[col] == cat).astype(int)
        df = df.drop(columns=[col])

    return df


def main():
    print("Loading data...")
    df = pd.read_csv("train.csv")
    print(f"Shape: {df.shape}")

    # 2. Missing value flag
    df['Previous_Claims_Missing'] = df['Previous Claims'].isna().astype(int)

    # 3. Numeric imputation — compute and save medians
    num_cols = ['Age', 'Annual Income', 'Number of Dependents',
                'Health Score', 'Vehicle Age', 'Credit Score', 'Insurance Duration']
    num_medians = df[num_cols].median().to_dict()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 4. Categorical imputation — compute and save modes/median
    prev_claims_median = df['Previous Claims'].median()
    df['Previous Claims'] = df['Previous Claims'].fillna(prev_claims_median)

    marital_mode = df['Marital Status'].mode()[0]
    feedback_mode = df['Customer Feedback'].mode()[0]
    df['Marital Status'] = df['Marital Status'].fillna(marital_mode)
    df['Customer Feedback'] = df['Customer Feedback'].fillna(feedback_mode)
    df['Occupation'] = df['Occupation'].fillna('Unknown')

    cat_modes = {
        'Marital Status': marital_mode,
        'Customer Feedback': feedback_mode,
        'Occupation': 'Unknown',
        'prev_claims_median': prev_claims_median,
    }

    # 5. Policy Start Date → date features, then drop
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Policy_Year'] = df['Policy Start Date'].dt.year
    df['Policy_Month'] = df['Policy Start Date'].dt.month
    df['Policy_DayOfWeek'] = df['Policy Start Date'].dt.dayofweek
    df['Policy_Age_Days'] = (pd.Timestamp('2026-04-19') - df['Policy Start Date']).dt.days
    df = df.drop(columns=['Policy Start Date'])

    # 6. Feature engineering
    df['Log_Annual_Income'] = np.log1p(df['Annual Income'])
    df['Income_Per_Person'] = df['Annual Income'] / (df['Number of Dependents'] + 1)
    df['Claims_Per_Year'] = df['Previous Claims'] / df['Insurance Duration']
    df['Age_Health_Interaction'] = df['Age'] * df['Health Score']
    df['Income_Credit_Ratio'] = df['Annual Income'] / (df['Credit Score'] + 1)
    df['Total_Risk_Duration'] = df['Vehicle Age'] + df['Insurance Duration']

    # 7. Binary encoding
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Smoking Status'] = df['Smoking Status'].map({'No': 0, 'Yes': 1})

    # 8. Ordinal encoding
    education = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    feedback = {'Poor': 0, 'Average': 1, 'Good': 2}
    exercise = {'Rarely': 0, 'Monthly': 1, 'Weekly': 2, 'Daily': 3}
    df['Education Level'] = df['Education Level'].map(education)
    df['Customer Feedback'] = df['Customer Feedback'].map(feedback)
    df['Exercise Frequency'] = df['Exercise Frequency'].map(exercise)

    # 9. One-Hot Encoding (drop_first=True) — save categories first
    nominal_cols = ['Marital Status', 'Occupation', 'Location', 'Policy Type', 'Property Type']
    ohe_categories = {col: sorted(df[col].dropna().unique().tolist()) for col in nominal_cols}
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    # 10. Define X and y
    drop_cols = ['Policy_Year', 'Policy_Age_Days', 'Policy_Month', 'Policy_DayOfWeek',
                 'Annual Income', 'Premium Amount', 'id', 'Income_Per_Person']
    X = df.drop(columns=drop_cols)
    y = df['Premium Amount']
    y_log = np.log1p(y)
    feature_columns = X.columns.tolist()

    # 11a. Evaluate on 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    eval_model = XGBRegressor(n_jobs=-1, random_state=42, verbosity=0)
    eval_model.fit(X_train, y_train)
    preds_log = eval_model.predict(X_test)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)
    rmse = mean_squared_error(y_true, preds) ** 0.5
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    print(f"\n--- 80/20 Evaluation (original scale) ---")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # 11b. Refit on full data
    print("\nRefitting on full data...")
    model = XGBRegressor(n_jobs=-1, random_state=42, verbosity=0)
    model.fit(X, y_log)

    # 12. Save artifacts
    joblib.dump(model, "model.joblib")
    joblib.dump(feature_columns, "feature_columns.joblib")
    joblib.dump(num_medians, "num_medians.joblib")
    joblib.dump(cat_modes, "cat_modes.joblib")
    joblib.dump(ohe_categories, "ohe_categories.joblib")

    print("\n✅ Saved: model.joblib, feature_columns.joblib, num_medians.joblib, "
          "cat_modes.joblib, ohe_categories.joblib")


if __name__ == "__main__":
    main()
