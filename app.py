import streamlit as st
import random
import pandas as pd
import joblib
st.set_page_config(
    page_title="CodeX Beverage: Price Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Load model and feature columns
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("lightgbm_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model_and_columns()
age_group_mapping = {
    '18-25': 0,
    '26-35': 1,
    '36-45': 2,
    '46-55': 3,
    '56-70': 4,
    '70+': 4  # 70+ treated as 56-70 because model didn't have 70+
}
income_mapping = {
    'Not Reported': 0,
    '<10L': 1,
    '10L - 15L': 2,
    '16L - 25L': 3,
    '26L - 35L': 4,
    '> 35L': 5
}

consume_freq_mapping = {
    '0-2 times': 0,
    '3-4 times': 1,
    '5-7 times': 2
}
pref_size_mapping = {
    'Small (250 ml)': 0,
    'Medium (500 ml)': 1,
    'Large (1 L)': 2
}

health_mapping = {
    'Low (Not very concerned)': 0,
    'Medium (Moderately health-conscious)': 1,
    'High (Very health-conscious)': 2
}
zone_score_mapping = {
    'Rural': 1,
    'Semi-Urban': 2,
    'Urban': 3,
    'Metro': 4
}
frequency_score_mapping = {
    '0-2 times': 1,
    '3-4 times': 2,
    '5-7 times': 3
}

awareness_score_mapping = {
    '0 to 1': 1,
    '2 to 4': 2,
    'above 4': 3
}
income_score_mapping = income_mapping  # same as income
price_range_mapping = {
    0: '50-100',
    1: '100-150',
    2: '150-200',
    3: '200-250'
}
# --- Centered Header ---
st.markdown("<h1 style='text-align: center;'>CodeX Beverage: Price Prediction</h1>", unsafe_allow_html=True)

# --- Row 1: Age, Gender, Zone, Occupation ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**Age**")
    age = st.number_input("", min_value=18, max_value=80, value=30, step=1, label_visibility="collapsed")
with col2:
    st.write("**Gender**")
    gender = st.selectbox("", ["M", "F"], index=0, label_visibility="collapsed")
with col3:
    st.write("**Zone**")
    zone = st.selectbox("", ["Urban", "Metro", "Rural", "Semi-Urban",], index=0, label_visibility="collapsed")
with col4:
    st.write("**Occupation**")
    occupation = st.selectbox("", ["Working Professional", "Student", "Entrepreneur", "Retired"], index=0, label_visibility="collapsed")

# --- Row 2 ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**Income Level (In L)**")
    income_level = st.selectbox(
        "",
        ["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L"],
        index=3, label_visibility="collapsed"
    )
with col2:
    st.write("**Consume Frequency (weekly)**")
    consume_frequency = st.selectbox(
        "",
        ["0-2 times", "3-4 times", "5-7 times"],
        index=2, label_visibility="collapsed"
    )
with col3:
    st.write("**Current Brand**")
    current_brand = st.selectbox(
        "",
        ["Newcomer", "Established"],
        index=0, label_visibility="collapsed"
    )
with col4:
    st.write("**Preferable Consumption Size**")
    preferable_size = st.selectbox(
        "",
        ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"],
        index=1, label_visibility="collapsed"
    )

# --- Row 3 ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**Awareness of other brands**")
    awareness = st.selectbox(
        "",
        ["0 to 1", "2 to 4", "Above 4"],
        index=1, label_visibility="collapsed"
    )
with col2:
    st.write("**Reasons for choosing brands**")
    reasons = st.selectbox(
        "",
        ["Price", "Quality", "Availability", "Brand Reputation"],
        index=0, label_visibility="collapsed"
    )
with col3:
    st.write("**Flavor Preference**")
    flavor = st.selectbox(
        "",
        ["Traditional", "Exotic"],
        index=0, label_visibility="collapsed"
    )
with col4:
    st.write("**Purchase Channel**")
    purchase_channel = st.selectbox(
        "",
        ["Online", "Retail Store"],
        index=0, label_visibility="collapsed"
    )

# --- Row 4 (only 3 items) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**Packaging Preference**")
    packaging = st.selectbox(
        "",
        ["Simple", "Premium", "Eco-friendly"],
        index=0, label_visibility="collapsed"
    )
with col2:
    st.write("**Health Concerns**")
    health_concern = st.selectbox(
        "",
        ['Low (Not very concerned)', 'Medium (Moderately health-conscious)',
       'High (Very health-conscious)'],
        index=0, label_visibility="collapsed"
    )
with col3:
    st.write("**Typical Consumption Situations**")
    consumption_situation = st.selectbox(
        "",
        ["Active (eg. Sports, gym)", "Casual (eg. At home)",
         "Social (eg. Parties)"],
        index=0, label_visibility="collapsed"
    )
# col4 remains empty

# --- Left-aligned Calculate Button ---
calculate = st.button("Calculate Price Range", use_container_width=False)

# --- Result Display (simple box) ---
if calculate:
    # 1. Preprocess inputs
    # Age group
    if age <= 25:
        age_group = '18-25'
    elif age <= 35:
        age_group = '26-35'
    elif age <= 45:
        age_group = '36-45'
    elif age <= 55:
        age_group = '46-55'
    elif age <= 70:
        age_group = '56-70'
    else:
        age_group = '70+'
    age_group_encoded = age_group_mapping[age_group]

    income_score = income_mapping[income_level]

    # Zone score
    zone_score = zone_score_mapping[zone]
    # Frequency score
    freq_score = frequency_score_mapping[consume_frequency]

    # Awareness score
    aware_score = awareness_score_mapping[awareness]

    # cf_ab_score
    cf_ab = round(freq_score / (freq_score + aware_score), 2)

    # zas_score
    zas = zone_score * income_score

    # bsi
    bsi = 1 if (current_brand != "Established" and reasons in ["Price", "Quality"]) else 0

    # Encode categoricals that were mapped (not one-hot)
    consume_freq_encoded = consume_freq_mapping[consume_frequency]
    pref_size_encoded = pref_size_mapping[preferable_size]
    health_encoded = health_mapping[health_concern]
    # Now create one-hot encoded features
    # We'll create a dictionary with all feature names from feature_columns set to 0
    input_dict = {col: 0 for col in feature_columns}

    # Fill numeric features
    input_dict['income_levels'] = income_score
    input_dict['consume_frequency(weekly)'] = consume_freq_encoded
    input_dict['preferable_consumption_size'] = pref_size_encoded
    input_dict['health_concerns'] = health_encoded
    input_dict['age_group'] = age_group_encoded
    input_dict['frequency_score'] = freq_score
    input_dict['awareness_score'] = aware_score
    input_dict['cf_ab_score'] = cf_ab
    input_dict['zone_score'] = zone_score
    input_dict['income_score'] = income_score
    input_dict['zas_score'] = zas
    input_dict['bsi'] = bsi

    # One-hot encoded columns: we need to set the appropriate ones to 1
    # gender_M: if gender == 'M' then 1 else 0 (since drop_first=True, gender_F is dropped)
    if gender == 'M':
        input_dict['gender_M'] = 1

    # zone: one-hot for Rural, Semi-Urban, Urban (Metro is dropped)
    if zone == 'Rural':
        input_dict['zone_Rural'] = 1
    elif zone == 'Semi-Urban':
        input_dict['zone_Semi-Urban'] = 1
    elif zone == 'Urban':
        input_dict['zone_Urban'] = 1
    # Metro is base, so all zeros

    # occupation: one-hot for Retired, Student, Working Professional (Entrepreneur is dropped)
    if occupation == 'Retired':
        input_dict['occupation_Retired'] = 1
    elif occupation == 'Student':
        input_dict['occupation_Student'] = 1
    elif occupation == 'Working Professional':
        input_dict['occupation_Working Professional'] = 1
    # Entrepreneur is base

    # current_brand: Newcomer is the only one-hot (Established is dropped)
    if current_brand == 'Newcomer':
        input_dict['current_brand_Newcomer'] = 1

    # awareness_of_other_brands: one-hot for '2 to 4' and 'above 4' (0 to 1 is dropped)
    if awareness == '2 to 4':
        input_dict['awareness_of_other_brands_2 to 4'] = 1
    elif awareness == 'above 4':
        input_dict['awareness_of_other_brands_above 4'] = 1

    # reasons_for_choosing_brands: one-hot for Brand Reputation, Price, Quality (Availability is dropped)
    if reasons == 'Brand Reputation':
        input_dict['reasons_for_choosing_brands_Brand Reputation'] = 1
    elif reasons == 'Price':
        input_dict['reasons_for_choosing_brands_Price'] = 1
    elif reasons == 'Quality':
        input_dict['reasons_for_choosing_brands_Quality'] = 1

    # flavor_preference: one-hot for Traditional (Exotic is dropped)
    if flavor == 'Traditional':
        input_dict['flavor_preference_Traditional'] = 1

    # purchase_channel: one-hot for Retail Store (Online is dropped)
    if purchase_channel == 'Retail Store':
        input_dict['purchase_channel_Retail Store'] = 1

    # packaging_preference: one-hot for Premium and Simple (Eco-Friendly is dropped)
    if packaging == 'Premium':
        input_dict['packaging_preference_Premium'] = 1
    elif packaging == 'Simple':
        input_dict['packaging_preference_Simple'] = 1

    # typical_consumption_situations: one-hot for Casual and Social (Active is dropped)
    if consumption_situation == 'Casual (eg. At home)':
        input_dict['typical_consumption_situations_Casual (eg. At home)'] = 1
    elif consumption_situation == 'Social (eg. Parties)':
        input_dict['typical_consumption_situations_Social (eg. Parties)'] = 1

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure columns are in the same order as feature_columns
    input_df = input_df[feature_columns]

    # Predict
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    confidence = pred_proba[pred_class] * 100

    price_range = price_range_mapping[pred_class]

    # Display result
    st.markdown(f"""
            Estimated Price Range: {price_range} INR

            
        """, unsafe_allow_html=True)

