import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time

# Load model
model_file = st.sidebar.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])
if model_file is not None:
    model = joblib.load(model_file)
else:
    st.warning("⚠️ Please upload your trained model file to continue.")
    st.stop()

# Emojis
condition_emojis = {
    'arthritis': '🦴',
    'asthma': '😮‍💨',
    'cancer': '🎗️',
    'diabetes': '🍬',
    'hypertension': '💓',
    'obesity': '⚖️'
}

# App config
st.set_page_config(page_title="Medical Condition Predictor", page_icon="💉", layout="wide")

# Minty green sidebar styling (CSS override)
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #DCEFEA;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Inputs
with st.sidebar:
    st.header("📝 Enter Patient Information")
    age = st.slider("Select Age", 0, 100, 30)
    room_number = st.number_input("Room Number", min_value=100, max_value=499, step=1)
    gender = st.selectbox("Gender", ["male", "female"])
    blood_type = st.selectbox("Blood Type", ["a+", "a-", "ab+", "ab-", "b+", "b-", "o+", "o-"])
    insurance = st.selectbox("Insurance Provider", ["aetna", "blue cross", "cigna", "medicare", "unitedhealthcare"])
    admission_type = st.selectbox("Admission Type", ["elective", "emergency", "urgent"])
    predict = st.button("Predict Medical Condition")

# Prediction mode
if predict:
    input_data = {
        "age": [age],
        "room number": [room_number],
        "gender_male": [1 if gender == "male" else 0],
        "gender_female": [1 if gender == "female" else 0]
    }

    for bt in ["a+", "a-", "ab+", "ab-", "b+", "b-", "o+", "o-"]:
        input_data[f"blood type_{bt}"] = [1 if blood_type == bt else 0]

    for p in ["aetna", "blue cross", "cigna", "medicare", "unitedhealthcare"]:
        input_data[f"insurance provider_{p}"] = [1 if insurance == p else 0]

    for t in ["elective", "emergency", "urgent"]:
        input_data[f"admission type_{t}"] = [1 if admission_type == t else 0]

    input_df = pd.DataFrame(input_data)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Spinner
    with st.spinner("Predicting the medical condition..."):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0]

    # Progress bar
    my_bar = st.progress(0, text="Loading prediction confidence...")
    for percent_complete in range(100):
        time.sleep(0.005)
        my_bar.progress(percent_complete + 1, text="Loading prediction confidence...")
    my_bar.empty()

    # Animation
    if prediction[0] in ['cancer', 'diabetes', 'hypertension']:
        st.balloons()
    else:
        st.snow()

    # Result display
    st.markdown("### <span style='color:green; font-weight:bold;'>✅ Prediction Complete!</span>", unsafe_allow_html=True)

    emoji = condition_emojis.get(prediction[0], "")
    st.markdown(f"### 🗂️ Predicted Medical Condition:")
    st.success(f"{emoji} {prediction[0].capitalize()}")

    # Charts
    proba_df = pd.DataFrame({
        'Condition': model.classes_,
        'Probability': proba
    }).sort_values(by='Probability', ascending=True)

    col_bar, col_pie = st.columns(2)

    with col_bar:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.barh(proba_df['Condition'], proba_df['Probability'], color='skyblue', edgecolor='black')
        ax.set_xlabel('Probability')
        ax.set_xlim(0, 1)
        ax.set_title('Prediction Confidence')
        ax.bar_label(bars, fmt='%.2f', label_type='edge')
        st.pyplot(fig)

    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        colors = plt.cm.Paired(range(len(proba_df)))
        ax_pie.pie(proba_df['Probability'], labels=proba_df['Condition'], autopct='%1.1f%%', startangle=140, colors=colors)
        ax_pie.axis('equal')
        ax_pie.set_title("Confidence Breakdown")
        st.pyplot(fig_pie)

# Intro screen (only when not predicting)
else:
    st.image("header.jpg", use_container_width=True)
    st.markdown("<h1 style='text-align: center;'>🏥 Medical Condition Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This app predicts a patient’s medical condition based on their profile and hospital admission details.</p>", unsafe_allow_html=True)
