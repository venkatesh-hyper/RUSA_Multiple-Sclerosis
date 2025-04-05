import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import base64

# Function to add a background image using base64
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set image background
set_background("image/DNA_hero.png")  # Replace with your image file path

# Load models and encoders
model_dir = "models"
encoder_path = "encoders/label_encoders.pkl"
label_encoders = joblib.load(encoder_path)

# Classification models
clf_models = {
    "Random Forest": joblib.load(os.path.join(model_dir, "rf_classifier.pkl")),
    "XGBoost": joblib.load(os.path.join(model_dir, "xgb_classifier.pkl")),
    "Logistic Regression": joblib.load(os.path.join(model_dir, "logreg_classifier.pkl")),
    "Voting Classifier": joblib.load(os.path.join(model_dir, "voting_classifier.pkl")),
}

# Regression models for OR or BETA
reg_models_or = {
    "Random Forest": joblib.load(os.path.join(model_dir, "rf_or_beta.pkl")),
    "XGBoost": joblib.load(os.path.join(model_dir, "xgb_or_beta.pkl")),
    "Linear Regression": joblib.load(os.path.join(model_dir, "linreg_or_beta.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(model_dir, "gbr_or_beta.pkl")),
    "Stacking": joblib.load(os.path.join(model_dir, "stacking_or_beta.pkl")),
}

# Regression models for Polygenic Score
reg_models_poly = {
    "Random Forest": joblib.load(os.path.join(model_dir, "rf_polygenic.pkl")),
    "XGBoost": joblib.load(os.path.join(model_dir, "xgb_polygenic.pkl")),
    "Linear Regression": joblib.load(os.path.join(model_dir, "linreg_polygenic.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(model_dir, "gbr_polygenic.pkl")),
    "Stacking": joblib.load(os.path.join(model_dir, "stacking_polygenic.pkl")),
}

# Trait descriptions (example)
trait_info = {
    "Multiple Sclerosis": "An autoimmune disease that affects the brain and spinal cord.",
    "Type 2 Diabetes": "A chronic condition that affects how the body processes blood sugar.",
    "Alzheimer's Disease": "A progressive neurological disorder that leads to memory loss."
}

# App Title
st.title("🧬 GWAS Model Dashboard — Multiple Sclerosis Risk Prediction")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
option = st.sidebar.radio("Go to", ["Manual Input & Predict", "Upload SNP Data & Predict", "Model Accuracies", "About Project"])

# Prediction Page (Default Home)
if option == "Manual Input & Predict":
    st.subheader("📝 Enter SNP Details")

    snp = st.text_input("SNP ID (e.g., rs1234)")
    location = st.text_input("Chromosome Location (e.g., chr6)")
    risk_allele = st.text_input("Risk Allele (e.g., A)")
    mapped_gene = st.text_input("Mapped Gene (e.g., HLA-DRB1)")
    risk_freq = st.number_input("Risk Allele Frequency", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Predict"):
        try:
            df = pd.DataFrame([{ 'SNPS': snp, 'LOCATION': location, 'RISK_ALLELE': risk_allele, 'MAPPED_GENE': mapped_gene, 'RISK ALLELE FREQUENCY': risk_freq }])
            for col in ['SNPS', 'LOCATION', 'RISK_ALLELE', 'MAPPED_GENE']:
                df[col] = label_encoders[col].transform(df[col].astype(str))

            X = df[['SNPS', 'LOCATION', 'RISK_ALLELE', 'MAPPED_GENE', 'RISK ALLELE FREQUENCY']]
            trait = label_encoders['DISEASE/TRAIT'].inverse_transform(clf_models["Voting Classifier"].predict(X))[0]
            beta = reg_models_or["Stacking"].predict(X)[0]
            poly_score = reg_models_poly["Gradient Boosting"].predict(X)[0]

            st.success(f"🧬 Predicted Trait: **{trait}**")
            st.markdown(f"🧠 **Trait Info**: {trait_info.get(trait, 'No info available')}")

            if beta < 1:
                beta_msg = "🟡 This SNP has a **neutral or low effect** on the associated trait."
            elif beta < 2:
                beta_msg = "🟠 This SNP has a **moderate risk** association with the trait."
            else:
                beta_msg = "🔴 This SNP has a **strong risk association** with the trait."

            st.info(f"📊 OR/BETA Value: **{beta:.4f}**")
            st.caption(beta_msg)

            if poly_score < 25:
                poly_msg = "🟢 Low polygenic risk."
            elif poly_score < 50:
                poly_msg = "🟡 Moderate polygenic risk."
            else:
                poly_msg = "🔴 High polygenic risk — consider deeper clinical evaluation."

            st.info(f"📈 Polygenic Score: **{poly_score:.2f}**")
            st.caption(poly_msg)

            st.markdown("---")
            st.markdown("## 🔍 Summary Suggestion")
            st.write(f"""
            Based on your SNP input:
            - You are **likely genetically predisposed** to *{trait}*.
            - The **effect size** (OR/BETA) indicates: {beta_msg.split(': ')[1]}
            - The **polygenic score** places you in the: {poly_msg.split(': ')[1]}
            """)

        except Exception as e:
            st.error(f"🚨 Error: {e}")

# Upload Page
elif option == "Upload SNP Data & Predict":
    st.subheader("📤 Upload Your SNP File (.csv)")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        try:
            input_cols = ['SNPS', 'LOCATION', 'RISK_ALLELE', 'MAPPED_GENE', 'RISK ALLELE FREQUENCY']
            for col in ['SNPS', 'LOCATION', 'RISK_ALLELE', 'MAPPED_GENE']:
                le = label_encoders[col]
                user_df[col] = le.transform(user_df[col].astype(str))

            X_input = user_df[input_cols]
            clf_pred = clf_models["Voting Classifier"].predict(X_input)
            decoded = label_encoders['DISEASE/TRAIT'].inverse_transform(clf_pred)
            beta_pred = reg_models_or["Stacking"].predict(X_input)
            poly_pred = reg_models_poly["Gradient Boosting"].predict(X_input)

            user_df['Predicted Trait'] = decoded
            user_df['Pred OR/BETA'] = beta_pred
            user_df['Pred Polygenic Score'] = poly_pred
            st.dataframe(user_df)

            csv_data = user_df.to_csv(index=False)
            st.download_button("📥 Download Results", data=csv_data, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"🚨 Error processing file: {e}")

# Accuracy Page
elif option == "Model Accuracies":
    st.subheader("🎯 Classification Accuracies")
    st.write("""
    | Model | Accuracy |
    |-------|----------|
    | Random Forest | 0.7596 |
    | XGBoost | 0.7760 |
    | Voting Classifier | 0.7650 |
    """)

    st.subheader("📈 Regression MSE (OR or BETA)")
    st.write("""
    | Model | MSE |
    |-------|-----|
    | Linear Regression | 2.6098 |
    | Stacking | **0.0046** |
    | Random Forest | 0.0052 |
    """)

    st.subheader("📈 Regression MSE (Polygenic Score)")
    st.write("""
    | Model | MSE |
    |-------|-----|
    | Gradient Boosting | **2.156** |
    | Stacking | 3.00880 |
    """)

# About Page
elif option == "About Project":
    st.subheader("📚 About This Project")
    st.markdown("""
    This GWAS Model Dashboard is designed to predict disease risk based on SNP data using state-of-the-art Machine Learning models.

    - 🧠 Built for predicting **Multiple Sclerosis**, but also trained on related traits.
    - 💻 Combines classification (disease trait) and regression (OR/BETA and Polygenic Score).
    - 📊 Powered by **Voting Classifier**, **Stacking Regressor**, and **Gradient Boosting**.
    - 🛠️ Developed using **Python**, **Scikit-learn**, **XGBoost**, and **Streamlit**.

    #### Developed by Venkatesh P — aspiring ML Engineer 🚀
    """)
    st.image("image/bio info.jpg", caption="Genomics + ML = Future of Healthcare", use_column_width=True)
    st.markdown("---")
    
    st.markdown(""" 🙏 Special Thanks

I would like to express my sincere gratitude to:

- **Dr. Lavanya, University of Madras, Department of Computer Science**,  
  for her invaluable guidance, encouragement, and expert insights throughout the project.

- **RUSA (Rashtriya Uchchatar Shiksha Abhiyan), University of Madras**,  
  for providing the research infrastructure, funding, and academic support that made this work possible.

This project wouldn't have reached its current state without their support and mentorship.
I'm grateful for their trust in me and my abilities.""")
