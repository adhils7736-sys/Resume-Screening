import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(page_title="AI Resume Scanner", page_icon="🔍")

# Load model
@st.cache_resource
def load_model():
    try:
        # This loads the GridSearchCV object saved in your notebook
        return joblib.load('resume_model.pkl')
    except:
        return None

model = load_model()

st.title("🔍 Smart Resume Screening")
st.write("This version uses **Skills + Job Role** to determine hiring potential.")

if model is None:
    st.error("Model file not found. Please re-train your notebook with combined features and save 'resume_model.pkl'.")
else:
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Candidate Name")
            job_role = st.selectbox("Target Job Role", 
                ['AI Researcher', 'Data Scientist', 'Cybersecurity Analyst', 'Software Engineer'])
        
        with col2:
            experience = st.number_input("Years of Experience", 0, 40, 5)
            # This is the critical field the model now needs:
            skills = st.text_area("Skills", placeholder="e.g. Python, SQL, TensorFlow")

        submit = st.form_submit_button("Analyze Candidate")

    if submit:
        # WE MUST FORMAT THE INPUT EXACTLY LIKE THE TRAINING DATA
        # We combine Skills and Job Role into one string
        combined_text = f"{skills} {job_role}"
        
        # Predict using the pipeline (which includes the TF-IDF vectorizer)
        prediction = model.predict([combined_text])[0]
        
        st.divider()
        if prediction == "Hire":
            st.success(f"### Result: {prediction}")
            st.balloons()
            st.write(f"Candidate **{name}** matches the profile for **{job_role}**.")
        else:
            st.error(f"### Result: {prediction}")
            st.write(f"Candidate **{name}** does not currently meet the high-match criteria for **{job_role}**.")

# Show data sample for reference
if st.checkbox("View Training Data Sample"):
    df = pd.read_csv('AI_Resume_Screening.csv')
    st.dataframe(df.head())