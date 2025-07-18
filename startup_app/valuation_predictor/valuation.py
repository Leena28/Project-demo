# import joblib
# import numpy as np
# import pandas as pd

# # 🔹 Load trained model & preprocessing artifacts
# valuation_model = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/valuation_model.joblib")

# scaler_amount = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/scaler_log_amount.joblib")
# scaler_growth_rate = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/scaler_growth_rate.joblib")
# scaler_target = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/scaler_log_target.joblib")

# industry_mapping = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/industry_encoding_map.joblib")
# country_mapping = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/Country_encoding_map.joblib")

# # 🔹 Final features used by the model
# final_features = [
#     'scaled_log_amount',
#     'scaled_growth_rate',
#     'log_amt_to_val_ratio',
#     'Industry_encoded',
#     'Country_encoded'
# ]

# # 🔧 Preprocess user input for model
# def preprocess_input(amount, growth_rate, industry, country):
#     # 1. Log transform & scale amount
#     log_amount = np.log1p(amount)
#     scaled_amount = scaler_amount.transform([[log_amount]])[0][0]

#     # 2. Scale growth rate
#     scaled_growth_rate = scaler_growth_rate.transform([[growth_rate]])[0][0]

#     # 3. Calculate ratio
#     log_amt_to_val_ratio = log_amount / (growth_rate + 1e-5)

#     # 4. Encode categoricals
#     industry_encoded = industry_mapping.get(industry, 0)
#     country_encoded = country_mapping.get(country, 0)

#     # 5. Create final input
#     final_input = pd.DataFrame([[ 
#         scaled_amount,
#         scaled_growth_rate,
#         log_amt_to_val_ratio,
#         industry_encoded,
#         country_encoded
#     ]], columns=final_features)

#     return final_input

# # 💡 Predict valuation based on processed input
# def predict_valuation(amount, growth_rate, industry, country, unit="millions"):
#     input_df = preprocess_input(amount, growth_rate, industry, country)
#     prediction_scaled_log = valuation_model.predict(input_df)

#     # Inverse scale & log
#     log_valuation = scaler_target.inverse_transform(prediction_scaled_log.reshape(-1, 1)).flatten()
#     valuation = np.expm1(log_valuation)[0]

#     # Convert to millions or billions
#     if unit == "billions":
#         valuation = valuation / 1e9
#     else:
#         valuation = valuation / 1e6

#     return round(valuation, 2)
import joblib
import numpy as np
import pandas as pd

# Declare globals
valuation_model = None
scaler_amount = None
scaler_growth_rate = None
scaler_target = None
industry_mapping = None
country_mapping = None

# 🔹 Final features used by the model
final_features = [
    'scaled_log_amount',
    'scaled_growth_rate',
    'log_amt_to_val_ratio',
    'Industry_encoded',
    'Country_encoded'
]

# 🔧 Load all artifacts (run this once from main.py)
def load_artifacts():
    global valuation_model, scaler_amount, scaler_growth_rate, scaler_target, industry_mapping, country_mapping

    valuation_model = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/valuation_model.joblib")
    scaler_amount = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/scaler_log_amount.joblib")
    scaler_growth_rate = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/scaler_growth_rate.joblib")
    scaler_target = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/scaler_log_target.joblib")
    industry_mapping = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/industry_encoding_map.joblib")
    country_mapping = joblib.load(r"C:/Users/CG-DTE/Desktop/Project/startup_app/valuation_predictor/models/Country_encoding_map.joblib")

# 🔧 Preprocess user input for model
def preprocess_input(amount, growth_rate, industry, country):
    log_amount = np.log1p(amount)
    scaled_amount = scaler_amount.transform([[log_amount]])[0][0]
    scaled_growth_rate = scaler_growth_rate.transform([[growth_rate]])[0][0]
    log_amt_to_val_ratio = log_amount / (growth_rate + 1e-5)
    industry_encoded = industry_mapping.get(industry, 0)
    country_encoded = country_mapping.get(country, 0)

    final_input = pd.DataFrame([[ 
        scaled_amount,
        scaled_growth_rate,
        log_amt_to_val_ratio,
        industry_encoded,
        country_encoded
    ]], columns=final_features)

    return final_input

# 💡 Predict valuation
def predict_valuation(amount, growth_rate, industry, country, unit="millions"):
    input_df = preprocess_input(amount, growth_rate, industry, country)
    prediction_scaled_log = valuation_model.predict(input_df)
    log_valuation = scaler_target.inverse_transform(prediction_scaled_log.reshape(-1, 1)).flatten()
    valuation = np.expm1(log_valuation)[0]

    if unit == "billions":
        valuation = valuation / 1e9
    else:
        valuation = valuation / 1e6

    #return round(valuation, 2)
    return float(round(valuation, 2))

