"""
app.py - Insurance Premium Predictor Streamlit App
Run: streamlit run app.py
"""

import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="🛡️ Insurance Premium Predictor",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path("model.joblib")
FEATURES_PATH = Path("feature_columns.joblib")
NUM_MEDIANS_PATH = Path("num_medians.joblib")
CAT_MODES_PATH = Path("cat_modes.joblib")
OHE_CATEGORIES_PATH = Path("ohe_categories.joblib")


@st.cache_resource
def load_artifacts():
    paths = [MODEL_PATH, FEATURES_PATH, NUM_MEDIANS_PATH, CAT_MODES_PATH, OHE_CATEGORIES_PATH]
    if not all(p.exists() for p in paths):
        return None, None, None, None, None
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    num_medians = joblib.load(NUM_MEDIANS_PATH)
    cat_modes = joblib.load(CAT_MODES_PATH)
    ohe_categories = joblib.load(OHE_CATEGORIES_PATH)
    return model, feature_columns, num_medians, cat_modes, ohe_categories


model, feature_columns, num_medians, cat_modes, ohe_categories = load_artifacts()

if model is None:
    st.error(
        "⚠️ Model files not found. Please run `python save_model.py` first to train and save the model."
    )
    st.stop()

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛡️ Insurance Premium Predictor")
st.markdown("Enter your details in the sidebar and click **🔍 Tahmin Et** to get your premium estimate.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("👤 Kişisel Bilgiler")
age = st.sidebar.slider("Age", 18, 64, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox(
    "Marital Status",
    sorted(ohe_categories.get("Marital Status", ["Married", "Single", "Divorced"]))
)
num_dependents = st.sidebar.slider("Number of Dependents", 0, 10, 2)
education_level = st.sidebar.selectbox(
    "Education Level",
    ["High School", "Bachelor's", "Master's", "PhD"]
)
occupation = st.sidebar.selectbox(
    "Occupation",
    sorted(ohe_categories.get("Occupation", ["Employed", "Self-Employed", "Unemployed", "Unknown"]))
)

st.sidebar.header("🏥 Sağlık & Yaşam")
health_score = st.sidebar.slider("Health Score", 1.0, 100.0, 50.0, step=0.1)
smoking_status = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
exercise_frequency = st.sidebar.selectbox(
    "Exercise Frequency",
    ["Rarely", "Monthly", "Weekly", "Daily"]
)
customer_feedback = st.sidebar.selectbox("Customer Feedback", ["Poor", "Average", "Good"])

st.sidebar.header("💰 Finansal Bilgiler")
annual_income = st.sidebar.number_input(
    "Annual Income", min_value=0.0, value=50000.0, step=1000.0
)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
previous_claims_unknown = st.sidebar.checkbox("Previous Claims unknown?", value=False)
if previous_claims_unknown:
    previous_claims = None
    st.sidebar.caption("Will use dataset median for Previous Claims.")
else:
    previous_claims = st.sidebar.slider("Previous Claims", 0, 15, 1)

st.sidebar.header("🚗 Araç & Sigorta")
vehicle_age = st.sidebar.slider("Vehicle Age", 0, 20, 5)
insurance_duration = st.sidebar.slider("Insurance Duration", 1, 10, 3)
policy_type = st.sidebar.selectbox(
    "Policy Type",
    sorted(ohe_categories.get("Policy Type", ["Basic", "Comprehensive", "Premium"]))
)
policy_start_date = st.sidebar.date_input("Policy Start Date", value=datetime.date.today())
location = st.sidebar.selectbox(
    "Location",
    sorted(ohe_categories.get("Location", ["Rural", "Suburban", "Urban"]))
)
property_type = st.sidebar.selectbox(
    "Property Type",
    sorted(ohe_categories.get("Property Type", ["Apartment", "Condo", "House"]))
)

predict_btn = st.sidebar.button("🔍 Tahmin Et", use_container_width=True)


# ── Build Input ───────────────────────────────────────────────────────────────
def build_input() -> pd.DataFrame:
    nominal_cols = ['Marital Status', 'Occupation', 'Location', 'Policy Type', 'Property Type']

    prev_claims_val = (
        cat_modes['prev_claims_median']
        if previous_claims is None
        else float(previous_claims)
    )
    prev_claims_missing = 1 if previous_claims is None else 0

    policy_dt = pd.Timestamp(policy_start_date)
    policy_year = policy_dt.year
    policy_month = policy_dt.month
    policy_dow = policy_dt.dayofweek
    policy_age_days = (pd.Timestamp('2026-04-19') - policy_dt).days

    log_annual_income = np.log1p(annual_income)
    income_per_person = annual_income / (num_dependents + 1)
    claims_per_year = prev_claims_val / insurance_duration
    age_health = age * health_score
    income_credit = annual_income / (credit_score + 1)
    total_risk = vehicle_age + insurance_duration

    education_map = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    feedback_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    exercise_map = {'Rarely': 0, 'Monthly': 1, 'Weekly': 2, 'Daily': 3}

    row = {
        'Age': float(age),
        'Gender': 0 if gender == 'Male' else 1,
        'Number of Dependents': float(num_dependents),
        'Education Level': education_map[education_level],
        'Health Score': float(health_score),
        'Previous Claims': prev_claims_val,
        'Vehicle Age': float(vehicle_age),
        'Credit Score': float(credit_score),
        'Insurance Duration': float(insurance_duration),
        'Customer Feedback': feedback_map[customer_feedback],
        'Smoking Status': 0 if smoking_status == 'No' else 1,
        'Exercise Frequency': exercise_map[exercise_frequency],
        'Previous_Claims_Missing': prev_claims_missing,
        'Log_Annual_Income': log_annual_income,
        'Claims_Per_Year': claims_per_year,
        'Age_Health_Interaction': age_health,
        'Income_Credit_Ratio': income_credit,
        'Total_Risk_Duration': total_risk,
    }

    # One-Hot Encoding using ohe_categories
    for col, cats in ohe_categories.items():
        col_val = {
            'Marital Status': marital_status,
            'Occupation': occupation,
            'Location': location,
            'Policy Type': policy_type,
            'Property Type': property_type,
        }[col]
        for cat in cats[1:]:  # drop_first=True equivalent
            row[f'{col}_{cat}'] = 1 if col_val == cat else 0

    df_input = pd.DataFrame([row])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    return df_input


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    try:
        X_input = build_input()
        pred_log = model.predict(X_input)
        pred = float(np.expm1(pred_log[0]))

        col1, col2 = st.columns([1, 1])

        with col1:
            st.success(f"### 💰 Predicted Premium Amount\n# **{pred:,.2f} TRY / $**")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"suffix": " $", "valueformat": ",.0f"},
                title={"text": "Predicted Premium Amount"},
                gauge={
                    "axis": {"range": [0, 5000]},
                    "bar": {"color": "royalblue"},
                    "steps": [
                        {"range": [0, 1000], "color": "#d4edda"},
                        {"range": [1000, 2500], "color": "#fff3cd"},
                        {"range": [2500, 5000], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": pred,
                    },
                },
            ))
            fig_gauge.update_layout(height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Input summary table
            st.markdown("### 📋 Input Summary")
            summary_data = {
                "Field": [
                    "Age", "Gender", "Marital Status", "Number of Dependents",
                    "Education Level", "Occupation", "Health Score",
                    "Smoking Status", "Exercise Frequency", "Customer Feedback",
                    "Annual Income", "Credit Score", "Previous Claims",
                    "Vehicle Age", "Insurance Duration", "Policy Type",
                    "Policy Start Date", "Location", "Property Type",
                ],
                "Value": [
                    age, gender, marital_status, num_dependents,
                    education_level, occupation, health_score,
                    smoking_status, exercise_frequency, customer_feedback,
                    f"{annual_income:,.0f}", credit_score,
                    "Unknown" if previous_claims is None else previous_claims,
                    vehicle_age, insurance_duration, policy_type,
                    str(policy_start_date), location, property_type,
                ],
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Feature Importance
        st.markdown("### 📊 Feature Importance")
        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({"Feature": feature_columns, "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(20)
        )
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker_color="steelblue",
        ))
        fig_fi.update_layout(
            title="Top 20 Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

else:
    st.info("👈 Enter your details in the sidebar and click **🔍 Tahmin Et** to get your premium estimate.")

# ── About ─────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this App"):
    st.markdown("""
    ### 🛡️ Insurance Premium Predictor

    This app predicts **insurance premium amounts** using a machine learning model
    trained on the **Kaggle Playground Series S5E8 – Regression with an Insurance Dataset**.

    **Model:** XGBRegressor (log1p-transformed target)

    **Pipeline:**
    - Missing value imputation (median/mode)
    - Date feature extraction from Policy Start Date
    - Feature engineering (log income, claims per year, risk interactions)
    - Binary & ordinal encoding
    - One-hot encoding for nominal features

    **Usage:**
    ```bash
    pip install -r requirements.txt
    python save_model.py   # train and save model artifacts
    streamlit run app.py   # launch the app
    ```
    """)
