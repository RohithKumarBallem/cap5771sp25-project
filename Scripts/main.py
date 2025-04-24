import streamlit as st


st.set_page_config(layout="wide", page_title="Car Recommendation Engine")
st.markdown("""
    <style>
    /*  MAIN CONTENT AREA (right side) */
    .main .block-container {
        background-color: #fffdd0;  /* Light yellow */
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }

    /*  SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #fdfaf6;  /* Light cream */
    }

    /*  Input labels in sidebar */
    label {
        font-size: 1.1rem;
        font-weight: 500;
    }

    /*  Buttons */
    .stButton>button {
        background-color: #ffa64d;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        border: none;
        border-radius: 8px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #ff944d;
    }

    /*  Input fields and dropdowns */
    .stNumberInput input, .stSelectbox div {
        font-size: 1.05rem;
    }

    /* General headers */
    .stApp h1 {
        color: #d35400;
        font-size: 2.8rem;
        font-weight: bold;
    }

    h2, h3, h4 {
        color: #b34700;
    }
    </style>
""", unsafe_allow_html=True)

import numpy as np
import joblib
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

import hashlib
import json


lrmodel = joblib.load('models/logistic_regression_model.pkl')
knnmodel = joblib.load('models/knn_model.pkl')
rfcmodel = joblib.load('models/rf_model.pkl')
scaler =  joblib.load('models/scaler.pkl')

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.title("Car Recommendation Engine")


st.sidebar.markdown("# Enter Car Specifications")

def get_cluster_for_predicted_car(predicted_label, car_data):
 
    company_model = predicted_label
    predicted_car_row = car_data[car_data['company_model'] == company_model]

    if predicted_car_row.empty:
        raise ValueError(f"No car found in dataset matching '{predicted_label}'")

    predicted_car_cluster = predicted_car_row['Cluster'].values[0]
    return predicted_car_cluster, company_model


def recommend_similar_cars(predicted_label, car_data, top_n=5):
    
    predicted_cluster, company_model = get_cluster_for_predicted_car(predicted_label, car_data)
    
    similar_cars = car_data[car_data['Cluster'] == predicted_cluster].copy()

    feature_cols = [
        'Annual Income', 'Transmission', 'Color', 'Price ($)', 'Sale_Year',
        'Body_type', 'number_of_seats', 'wheelbase_mm', 'full_weight_kg',
        'max_trunk_capacity_l', 'injection_type', 'engine_type',
        'valves_per_cylinder', 'stroke_cycle_mm', 'turnover_of_maximum_torque_rpm',
        'engine_hp_rpm', 'drive_wheels', 'turning_circle_m', 'transmission',
        'fuel_tank_capacity_l', 'max_speed_km_per_h', 'fuel_grade',
        'back_suspension', 'rear_brakes', 'front_brakes', 'front_suspension',
        'avg_rating', 'max_rating', 'min_rating', 'popularity_score',
        'fuel_eff_score', 'avg_track_mm', 'displacement_per_cylinder',
        'power_to_weight', 'volume_mm3'
    ]

    similar_cars_scaled = similar_cars[feature_cols]
    similar_cars['score'] = similar_cars_scaled.sum(axis=1)

    recommended_cars = similar_cars.sort_values(by='score', ascending=False)
    return recommended_cars[['company_model', 'score']].drop_duplicates('company_model').head(top_n), company_model

def get_cluster_for_predicted_car_gmm(predicted_label, car_data):
  
    company_model = predicted_label
    predicted_car_row = car_data[car_data['company_model'] == company_model]

    if predicted_car_row.empty:
        raise ValueError(f"No car found in dataset matching '{predicted_label}'")

    predicted_car_cluster = predicted_car_row['GMM_Cluster'].values[0]
    return predicted_car_cluster, company_model


def recommend_similar_cars_gmm(predicted_label, car_data, top_n=5):

    predicted_cluster, company_model = get_cluster_for_predicted_car_gmm(predicted_label, car_data)

    similar_cars = car_data[car_data['GMM_Cluster'] == predicted_cluster].copy()
    similar_cars = similar_cars[similar_cars['company_model'] != company_model]

    feature_cols = [
        'Annual Income', 'Transmission', 'Color', 'Price ($)', 'Sale_Year',
        'Body_type', 'number_of_seats', 'wheelbase_mm', 'full_weight_kg',
        'max_trunk_capacity_l', 'injection_type', 'engine_type',
        'valves_per_cylinder', 'stroke_cycle_mm', 'turnover_of_maximum_torque_rpm',
        'engine_hp_rpm', 'drive_wheels', 'turning_circle_m', 'transmission',
        'fuel_tank_capacity_l', 'max_speed_km_per_h', 'fuel_grade',
        'back_suspension', 'rear_brakes', 'front_brakes', 'front_suspension',
        'avg_rating', 'max_rating', 'min_rating', 'popularity_score',
        'fuel_eff_score', 'avg_track_mm', 'displacement_per_cylinder',
        'power_to_weight', 'volume_mm3'
    ]

    similar_cars_scaled = similar_cars[feature_cols]
    similar_cars['score'] = similar_cars_scaled.sum(axis=1)

    recommended_cars = similar_cars.sort_values(by='score', ascending=False)
    return recommended_cars[['company_model', 'score']].drop_duplicates('company_model').head(top_n), company_model

def get_cluster_for_predicted_car_agg(predicted_label, car_data):
  
    company_model = predicted_label
    predicted_car_row = car_data[car_data['company_model'] == company_model]

    if predicted_car_row.empty:
        raise ValueError(f"No car found in dataset matching '{predicted_label}'")

    predicted_car_cluster = predicted_car_row['Agglo_Cluster'].values[0]
    return predicted_car_cluster, company_model

def recommend_similar_cars_agg(predicted_label, car_data, top_n=5):
   
    predicted_cluster, company_model = get_cluster_for_predicted_car_agg(predicted_label, car_data)

    similar_cars = car_data[car_data['Agglo_Cluster'] == predicted_cluster].copy()
    similar_cars = similar_cars[similar_cars['company_model'] != company_model]
    
    feature_cols = [
        'Annual Income', 'Transmission', 'Color', 'Price ($)', 'Sale_Year',
        'Body_type', 'number_of_seats', 'wheelbase_mm', 'full_weight_kg',
        'max_trunk_capacity_l', 'injection_type', 'engine_type',
        'valves_per_cylinder', 'stroke_cycle_mm', 'turnover_of_maximum_torque_rpm',
        'engine_hp_rpm', 'drive_wheels', 'turning_circle_m', 'transmission',
        'fuel_tank_capacity_l', 'max_speed_km_per_h', 'fuel_grade',
        'back_suspension', 'rear_brakes', 'front_brakes', 'front_suspension',
        'avg_rating', 'max_rating', 'min_rating', 'popularity_score',
        'fuel_eff_score', 'avg_track_mm', 'displacement_per_cylinder',
        'power_to_weight', 'volume_mm3'
    ]

    similar_cars_scaled = similar_cars[feature_cols]
    similar_cars['score'] = similar_cars_scaled.sum(axis=1)

    recommended_cars = similar_cars.sort_values(by='score', ascending=False)
    return recommended_cars[['company_model', 'score']].drop_duplicates('company_model').head(top_n), company_model

    
st.sidebar.title("Car Feature Input")

col1, col2 = st.sidebar.columns(2)

with col1:
    price = st.slider('Price ($)', min_value=1000, max_value=100000, value=22500)  
    min_rating = st.slider('Min Rating', min_value=1.0, max_value=5.0, value=1.0)  
    max_trunk_capacity = st.slider('Max Trunk Capacity (L)', min_value=100, max_value=5000, value=900)  
    max_torque_rpm = st.slider('Torque RPM', min_value=1000, max_value=8000, value=4400) 
    
    num_seats = st.slider('Number of Seats', min_value=2, max_value=9, value=5)  
    annual_income = st.slider('Annual Income ($)', min_value=10000, max_value=90000000, value=1025000)  
    Transmission = st.selectbox('Speed Transmission', ['Auto', 'Manual'], index=0)  
    Color = st.selectbox('Car Color', ['Black', 'Red', 'Pale White'], index=2)  
    
    
    wheelbase = st.slider('Wheelbase (mm)', min_value=1000, max_value=5000, value=2765)  
    full_weight = st.slider('Full Weight (kg)', min_value=800, max_value=4000, value=1950)  
    
    injection_type = st.selectbox('Injection Type', [
        'direct injection', 'Multi-point fuel injection',
        'distributed injection (multipoint)', 'direct injection (direct)',
        'Injector', 'Common Rail'], index=2)  
    engine_type = st.selectbox('Engine Type', ['Gasoline', 'petrol', 'Diesel'], index=1)  
    valves_per_cylinder = st.slider('Valves per Cylinder', min_value=2, max_value=8, value=4)  
    stroke_cycle = st.slider('Stroke Cycle (mm)', min_value=50, max_value=100, value=83)  
    body_type = st.selectbox('Body Type', ['Crossover', 'Sedan', 'Wagon', 'Coupe', 'Pickup', 'Cabriolet',
       'Minivan', 'Hatchback', 'Targa'], index=1)  
    drive_wheels = st.selectbox('Drive Wheels', [
        'Rear wheel drive', 'All wheel drive (AWD)', 'Front wheel drive', 
        'full', 'Four wheel drive (4WD)'], index=2)  
    sale_year = st.slider('Sale Year', min_value=2020, max_value=2025, value=2023) 
    turning_circle = st.slider('Turning Circle (m)', min_value=5, max_value=20, value=11)  
    
with col2:
    engine_hp_rpm = st.slider('Engine HP RPM', min_value=1000, max_value=9000, value=6400) 
    avg_track = st.slider('Average Track (mm)', min_value=1000.0, max_value=2000.0, value=1507.5)  
    popularity_score = st.slider('Popularity Score', min_value=0.0, max_value=50.0, value=21.5, step=0.01)  
    max_rating = st.slider('Max Rating', min_value=1.0, max_value=5.0, value=5.0) 
    avg_rating = st.slider('Average Rating', min_value=1.0, max_value=5.0, value=3.7, step=0.01)  
    power_to_weight = st.slider('Power to Weight Ratio', min_value=0.01, max_value=1.0, value=0.1754, step=0.0001)  
    volume = st.slider('Volume (m³)', min_value=1.0, max_value=30.0, value=13.42)  
    volume = volume * 1e9
    displacement_per_cylinder = st.slider('Displacement per Cylinder (L)', min_value=200.0, max_value=1000.0, value=600.83)  
    fuel_eff_score = st.slider('Fuel Efficiency Score', min_value=0.0, max_value=20.0, value=10.25, step=0.1)  
    fuel_tank_capacity = st.slider('Fuel Tank Capacity (L)', min_value=30, max_value=150, value=64)  
    max_speed = st.slider('Max Speed (km/h)', min_value=100, max_value=350, value=191)  
    fuel_grade = st.slider('Fuel Grade', min_value=80, max_value=110, value=95)  
    back_suspension = st.selectbox('Back Suspension Type', [
        'Independent, Dampers, Helical springs',
        'Multi wishbone, Dampers, spring', 
        'Independent, spring', 
        'Independent, Multi wishbone, spring, Stabilizer bar',
        'Multi wishbone', 
        'Torsion beam, spring', 
        'Multi wishbone, Stabilizer bar', 
        'Helical springs', 
        'Independent, A-shaped lever, Helical springs, Stabilizer bar',
        'Semi-independent, spring', 
        'Dependent, Spring, Dampers', 
        'Independent, Multi wishbone, Stabilizer bar',
        'Multi wishbone, Helical springs, Stabilizer bar', 
        'Independent, Multi wishbone, Dampers, Helical springs, Stabilizer bar',
        'Semi-dependent, Torsion beam', 
        'Double wishbone', 
        'Independent, Multi wishbone', 
        'Independent', 
        'Axle, Dampers, Helical springs, Stabilizer bar', 
        'Strut', 
        'Torsion, Dampers, Helical springs',
        'Independent, Multi wishbone, Dampers, Helical springs', 
        'Spring', 
        'Independent, Lever, Dampers, Helical springs', 
        'Independent, Double wishbone, Dampers, Helical springs', 
        'McPherson Struts, spring, Stabilizer bar',
        'Independent, McPherson Struts, spring, Stabilizer bar',
        'Torsion beam', 
        'Trailing arms', 
        'Semi-dependent', 
        'Independent, Double wishbone, Stabilizer bar', 
        'Dependent', 
        'Independent, Double wishbone, spring, Stabilizer bar',
        'Multi wishbone, Dampers, Stabilizer bar',
        'Semi-dependent, Stabilizer bar', 
        'Solid axle', 
        'Dependent, spring, Stabilizer bar', 
        'Axle, Dampers, Helical springs', 
        'Multi wishbone, Trailing arms, Stabilizer bar', 
        'Wishbone', 
        'Spring, Dampers', 
        'A few levers and rods', 
        'Stabilizer bar'
    ], index=2)  
    
    rear_brakes = st.selectbox('Rear Brakes', ['ventilated disc', 'Disc', 'drum'], index=1)  
    front_brakes = st.selectbox('Front Brakes', ['ventilated disc', 'Disc ventilated', 'Disc'], index=1)  
    front_suspension = st.selectbox('Front Suspension Type', [
        'Independent, Dampers, Helical springs', 
        'Lever, Dampers, spring',
        'Independent, spring', 
        'Multi wishbone',
        'Independent, McPherson Struts, spring',
        'Independent, McPherson Struts, Stabilizer bar',
        'Independent, McPherson Struts', 
        'Wishbone',
        'Independent, McPherson Struts, Dampers, Helical springs, Stabilizer bar',
        'Drive axle, Helical springs, Stabilizer bar',
        'Independent, Double wishbone, Stabilizer bar',
        'Independent, McPherson Struts, spring, Stabilizer bar',
        'Double wishbone', 
        'McPherson Struts',
        'Independent, Wishbone, Dampers, Helical springs, Stabilizer bar',
        'Dependent, spring, Stabilizer bar',
        'Independent, Dampers, Helical springs, Stabilizer bar', 
        'Strut',
        'Independent, Double wishbone, spring, Stabilizer bar',
        'Independent, Multi wishbone, Stabilizer bar',
        'McPherson Struts, Stabilizer bar', 
        'Helical springs',
        'Independent, Double wishbone, Dampers, Helical springs, Stabilizer bar',
        'Independent',
        'Independent, Double wishbone, Dampers, Stabilizer bar',
        'Axle, Dampers, Helical springs',
        'Multi wishbone, spring, Stabilizer bar',
        'Independent, Double wishbone, Dampers, spring', 
        'Stabilizer bar'
    ], index=2)  
     
    transmission = st.selectbox('Gear Transmission Type', ['Automatic', 'Manual'], index=0)  
    

# create dataframe
temp = pd.DataFrame({
    'Annual Income': [annual_income],
    'Transmission': [Transmission],
    'Color': [Color],
    'Price ($)': [price],
    'Sale_Year': [sale_year],
    'Body_type': [body_type],
    'number_of_seats': [num_seats],
    'wheelbase_mm': [wheelbase],
    'full_weight_kg': [full_weight],
    'max_trunk_capacity_l': [max_trunk_capacity],
    'injection_type': [injection_type],
    'engine_type': [engine_type],
    'valves_per_cylinder': [valves_per_cylinder],
    'stroke_cycle_mm': [stroke_cycle],
    'turnover_of_maximum_torque_rpm': [max_torque_rpm],
    'engine_hp_rpm': [engine_hp_rpm],
    'drive_wheels': [drive_wheels],
    'turning_circle_m': [turning_circle],
    'transmission': [transmission],
    'fuel_tank_capacity_l': [fuel_tank_capacity],
    'max_speed_km_per_h': [max_speed],
    'fuel_grade': [fuel_grade],
    'back_suspension': [back_suspension],
    'rear_brakes': [rear_brakes],
    'front_brakes': [front_brakes],
    'front_suspension': [front_suspension],
    'avg_rating': [avg_rating],
    'max_rating': [max_rating],
    'min_rating': [min_rating],
    'popularity_score': [popularity_score],
    'fuel_eff_score': [fuel_eff_score],
    'avg_track_mm': [avg_track],
    'displacement_per_cylinder': [displacement_per_cylinder],
    'power_to_weight': [power_to_weight],
    'volume_mm3': [volume]
})

st.write("## Updated Car Specifications:")
st.dataframe(temp)
sidebar_inputs = {
    'price': price,
    'annual_income': annual_income,
    'Transmission': Transmission,
    'Color': Color,
    
    'body_type': body_type,
    'num_seats': num_seats,
    'wheelbase': wheelbase,
    'full_weight': full_weight,
    'max_trunk_capacity': max_trunk_capacity,
    'injection_type': injection_type,
    'engine_type': engine_type,
    'valves_per_cylinder': valves_per_cylinder,
    'stroke_cycle': stroke_cycle,
    'max_torque_rpm': max_torque_rpm,
    'engine_hp_rpm': engine_hp_rpm,
    'drive_wheels': drive_wheels,
    'turning_circle': turning_circle,
    'fuel_tank_capacity': fuel_tank_capacity,
    'max_speed': max_speed,
    'fuel_grade': fuel_grade,
    'back_suspension': back_suspension,
    'rear_brakes': rear_brakes,
    'front_brakes': front_brakes,
    'front_suspension': front_suspension,
    'avg_rating': avg_rating,
    'max_rating': max_rating,
    'min_rating': min_rating,
    'popularity_score': popularity_score,
    'fuel_eff_score': fuel_eff_score,
    'transmission': transmission,
    'avg_track': avg_track,
    'displacement_per_cylinder': displacement_per_cylinder,
    'power_to_weight': power_to_weight,
    'volume': volume,
    'sale_year': sale_year
}

# unique hash
current_input_hash = hashlib.md5(json.dumps(sidebar_inputs, sort_keys=True).encode()).hexdigest()


if 'last_input_hash' not in st.session_state:
    st.session_state.last_input_hash = current_input_hash

# Clear model prediction if inputs changed
if current_input_hash != st.session_state.last_input_hash:
    st.session_state.last_model = None
    st.session_state.last_prediction = None
    st.session_state.last_input_hash = current_input_hash


categorical_cols = [
    'Transmission', 'Color', 'Body_type', 'injection_type', 
    'engine_type', 'drive_wheels', 'back_suspension', 
    'rear_brakes', 'front_brakes', 'front_suspension','transmission'
]


# Label encoding

for col in categorical_cols:
   
    le = label_encoders[col]
    
    
    temp[col] = le.transform(temp[col].astype(str))

# Define columns for log transformation and scaling
numeric_cols = ['Annual Income',
 'Transmission',
 'Color',
 'Price ($)',
 'Sale_Year',
 'Body_type',
 'number_of_seats',
 'wheelbase_mm',
 'full_weight_kg',
 'max_trunk_capacity_l',
 'injection_type',
 'engine_type',
 'valves_per_cylinder',
 'stroke_cycle_mm',
 'turnover_of_maximum_torque_rpm',
 'engine_hp_rpm',
 'drive_wheels',
 'turning_circle_m',
 'transmission',
 'fuel_tank_capacity_l',
 'max_speed_km_per_h',
 'fuel_grade',
 'back_suspension',
 'rear_brakes',
 'front_brakes',
 'front_suspension',
 'avg_rating',
 'max_rating',
 'min_rating',
 'popularity_score',
 'fuel_eff_score',
 'avg_track_mm',
 'displacement_per_cylinder',
 'power_to_weight',
 'volume_mm3']






# Apply log1p transformation and scaling

temp[numeric_cols] = temp[numeric_cols].apply(lambda col: np.log1p(col))
temp[numeric_cols] = scaler.transform(temp[numeric_cols])

if 'last_model' not in st.session_state:
    st.session_state.last_model = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Model selection
st.title("Set the car specifications in left panel and choose a model below to predict the car :")
col1, col2, col3 = st.columns(3)


logistic_clicked = col1.button("Logistic Regression")
knn_clicked = col2.button("K Nearest Neighbor")
rfc_clicked = col3.button("Random Forest Classifier")

# Save prediction to session state
if logistic_clicked:
    prediction = lrmodel.predict(temp)
    st.session_state.last_model = 'logistic'
    st.session_state.last_prediction = prediction[0]

if knn_clicked:
    prediction = knnmodel.predict(temp)
    st.session_state.last_model = 'knn'
    st.session_state.last_prediction = prediction[0]

if rfc_clicked:
    prediction = rfcmodel.predict(temp)
    st.session_state.last_model = 'rfc'
    st.session_state.last_prediction = prediction[0]


if st.session_state.last_prediction is not None:
    
    try:
        company, model = st.session_state.last_prediction.split('_', 1)
    except ValueError:
        company, model = st.session_state.last_prediction, "Unknown"

    # Get model name formatted
    model_used = st.session_state.last_model.upper() if st.session_state.last_model else "Unknown Model"
    model_name_map = {
        'LOGISTIC': 'Logistic Regression',
        'KNN': 'K-Nearest Neighbors',
        'RFC': 'Random Forest Classifier'
    }
    model_pretty = model_name_map.get(model_used, model_used)

    
    st.markdown(
        f"""
        <div style='
            background-color: #fff3e0;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #4e342e;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        '>
            <div style='font-size: 26px; margin-bottom: 10px;'>🚗 Predicted Car</div>
            <span style='color:#d84315; font-size: 24px;'>{company}</span>
            <span style='color:#3e2723; font-size: 24px;'>{model}</span>
            <div style='margin-top: 15px; font-size: 18px; color: #6d4c41;'>
                From model: <strong>{model_pretty}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.title("Set Number of Additional Cars")
num_additional_cars = st.slider("- - -", 
                                min_value=1, 
                                max_value=20, 
                                value=5, 
                                step=1)
st.info("This will affect how many cars are compared or clustered with your selected car.")
st.header(f"You selected {num_additional_cars} additional cars for analysis.")

# csv is read

car_full_analysis = pd.read_csv("models/clusteringanddata.csv")

st.title(" Click a clustering method below to get more similar recommendations:")

c1, c2, c3 = st.columns(3)

model_map = {
    'LOGISTIC': 'Logistic Regression',
    'KNN': 'K-Nearest Neighbors',
    'RFC': 'Random Forest Classifier'
}

#kmeans
with c1:
    if st.button("KMeans Clustering"):
        if st.session_state.last_prediction:
            model_used = st.session_state.last_model.upper()
            model_pretty = model_map.get(model_used, model_used)

            st.success(f"Running KMeans clustering based on previous model: {model_pretty}")
            try:
                recommended_top5, model_name = recommend_similar_cars(
                    st.session_state.last_prediction, car_full_analysis, top_n=num_additional_cars
                )

                try:
                    company_display, model_display = model_name.split('_', 1)
                except ValueError:
                    company_display, model_display = model_name, "Unknown"

                st.markdown(f"""
                    <div style='
                        background-color: #e3f2fd;
                        padding: 14px 24px;
                        margin-bottom: 24px;
                        border-left: 6px solid #2196f3;
                        border-radius: 10px;
                        font-size: 22px;
                        font-weight: 700;
                        color: #0d47a1;
                        text-transform: capitalize;
                    '>
                        {company_display} <span style='color:#0d47a1;'>{model_display}</span>
                    </div>
                """, unsafe_allow_html=True)

                for _, row in recommended_top5.iterrows():
                    try:
                        company, model = row['company_model'].split('_', 1)
                    except ValueError:
                        company, model = row['company_model'], "Unknown"
                    score = round(row['score'], 2)
                    st.markdown(f"""
                        <div style='background-color: #fff8e1; padding: 16px; border-radius: 10px;
                                     margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-transform: capitalize;'>
                            <div style='font-size: 22px; font-weight: bold; color: #4e342e;'>{company} <span style='color:#bf360c;'>{model}</span></div>
                            <div style='margin-top: 4px; font-size: 16px; color: #6d4c41;'>Score: <strong>{score}</strong></div>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during recommendation: {e}")
        else:
            st.warning("Please select and predict a car category first using a model!")

# Agglomerative Clustering
with c2:
    if st.button("Agglomerative Clustering"):
        if st.session_state.last_prediction:
            model_used = st.session_state.last_model.upper()
            model_pretty = model_map.get(model_used, model_used)

            st.success(f"Running Agglomerative clustering based on previous model: {model_pretty}")
            try:
                recommended_top5_agg, model_name_agg = recommend_similar_cars_agg(
                    st.session_state.last_prediction, car_full_analysis, top_n=num_additional_cars
                )

                try:
                    company_display, model_display = model_name_agg.split('_', 1)
                except ValueError:
                    company_display, model_display = model_name_agg, "Unknown"

                st.markdown(f"""
                    <div style='
                        background-color: #e3f2fd;
                        padding: 14px 24px;
                        margin-bottom: 24px;
                        border-left: 6px solid #2196f3;
                        border-radius: 10px;
                        font-size: 22px;
                        font-weight: 700;
                        color: #0d47a1;
                        text-transform: capitalize;
                    '>
                        {company_display} <span style='color:#0d47a1;'>{model_display}</span>
                    </div>
                """, unsafe_allow_html=True)

                for _, row in recommended_top5_agg.iterrows():
                    try:
                        company, model = row['company_model'].split('_', 1)
                    except ValueError:
                        company, model = row['company_model'], "Unknown"
                    score = round(row['score'], 2)
                    st.markdown(f"""
                        <div style='background-color: #fff8e1; padding: 16px; border-radius: 10px;
                                     margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-transform: capitalize;'>
                            <div style='font-size: 22px; font-weight: bold; color: #4e342e;'>{company} <span style='color:#bf360c;'>{model}</span></div>
                            <div style='margin-top: 4px; font-size: 16px; color: #6d4c41;'>Score: <strong>{score}</strong></div>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during Agglomerative recommendation: {e}")
        else:
            st.warning("Please select and predict a car category first using a model!")

# GMM Clustering
with c3:
    if st.button("GMM Clustering"):
        if st.session_state.last_prediction:
            model_used = st.session_state.last_model.upper()
            model_pretty = model_map.get(model_used, model_used)

            st.success(f"Running GMM clustering based on previous model: {model_pretty}")
            try:
                recommended_top5_gmm, model_name_gmm = recommend_similar_cars_gmm(
                    st.session_state.last_prediction, car_full_analysis, top_n=num_additional_cars
                )

                try:
                    company_display, model_display = model_name_gmm.split('_', 1)
                except ValueError:
                    company_display, model_display = model_name_gmm, "Unknown"

                st.markdown(f"""
                    <div style='
                        background-color: #e3f2fd;
                        padding: 14px 24px;
                        margin-bottom: 24px;
                        border-left: 6px solid #2196f3;
                        border-radius: 10px;
                        font-size: 22px;
                        font-weight: 700;
                        color: #0d47a1;
                        text-transform: capitalize;
                    '>
                        {company_display} <span style='color:#0d47a1;'>{model_display}</span>
                    </div>
                """, unsafe_allow_html=True)

                for _, row in recommended_top5_gmm.iterrows():
                    try:
                        company, model = row['company_model'].split('_', 1)
                    except ValueError:
                        company, model = row['company_model'], "Unknown"
                    score = round(row['score'], 2)
                    st.markdown(f"""
                        <div style='background-color: #fff8e1; padding: 16px; border-radius: 10px;
                                     margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-transform: capitalize;'>
                            <div style='font-size: 22px; font-weight: bold; color: #4e342e;'>{company} <span style='color:#bf360c;'>{model}</span></div>
                            <div style='margin-top: 4px; font-size: 16px; color: #6d4c41;'>Score: <strong>{score}</strong></div>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during GMM recommendation: {e}")
        else:
            st.warning("Please select and predict a car category first using a model!")


# supervised models performance

st.title(" Supervised Models Performance")

st.image("images/Supervised_EVAL.png", 
         caption="Comparison of Accuracy, F1, Recall, and Precision across models", 
         use_container_width=True)

# Clustering models Evaluation

st.title("Clustering Models Evaluation")

st.image("images/Clustering_EVAL.png", 
         caption="Comparison of Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index (ARI), and Cluster Purity across clustering methods", 
         use_container_width=True)



# Feature Importance Visualization

st.title("Feature Importance Visualization - Supervised Models")

col1, col2, col3 = st.columns(3)

if 'supervised_model' not in st.session_state:
    st.session_state.supervised_model = None

with col1:
    if st.button("Logistic Regression", key="btn_lr"):
        st.session_state.supervised_model = "logistic_regression"

with col2:
    if st.button("K nearest neighbor", key="btn_knn"):
        st.session_state.supervised_model = "knn"

with col3:
    if st.button("Random Forest Classifier", key="btn_rf"):
        st.session_state.supervised_model = "random_forest"


if st.session_state.supervised_model == "logistic_regression":
    st.image("images/LR_FI.png", caption="Logistic Regression - Top 10 Feature Importances", use_container_width=True)

elif st.session_state.supervised_model == "knn":
    st.image("images/KNN_FI.png", caption="KNN - Top 10 Feature Importances", use_container_width=True)

elif st.session_state.supervised_model == "random_forest":
    st.image("images/RF_FI.png", caption="Random Forest - Top 10 Feature Importances", use_container_width=True)


# Clustering Results Visualization

st.title("Clustering Results Visualization")

c1, c2, c3 = st.columns(3)

if 'clustering_method' not in st.session_state:
    st.session_state.clustering_method = None

with c1:
    if st.button("KMeans Clustering", key="btn_kmeans"):
        st.session_state.clustering_method = "kmeans"

with c2:
    if st.button("Agglomerative Clustering", key="btn_agglo"):
        st.session_state.clustering_method = "agglo"

with c3:
    if st.button("GMM Clustering", key="btn_gmm"):
        st.session_state.clustering_method = "gmm"


if st.session_state.clustering_method == "kmeans":
    st.image("images/KM_CL.png", caption="KMeans Clustering Results", use_container_width=True)

elif st.session_state.clustering_method == "agglo":
    st.image("images/AG_CL.png", caption="Agglomerative Clustering Results", use_container_width=True)

elif st.session_state.clustering_method == "gmm":
    st.image("images/GMM_CL.png", caption="GMM Clustering Results", use_container_width=True)
