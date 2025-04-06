import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


  
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
    "Multiple sclerosis": "A chronic autoimmune disease that affects the central nervous system, leading to nerve damage.",
    "Rheumatoid arthritis or multiple sclerosis": "A co-occurrence or shared genetic risk between MS and rheumatoid arthritis, both autoimmune diseases.",
    "Median relapse-independent longitudinal age-related multiple sclerosis severity score in relapse-onset multiple sclerosis": "A metric for tracking MS progression over time, independent of relapse events.",
    "Median relapse-independent longitudinal Multiple Sclerosis Severity Scale score in relapse-onset multiple sclerosis": "A longitudinal measure to assess MS severity unaffected by relapses.",
    "Multiple sclerosis (OCB status)": "Refers to the presence of oligoclonal bands in cerebrospinal fluid, used as a diagnostic marker in MS.",
    "Multiple sclerosis (severity)": "Indicates how aggressively MS symptoms and progression manifest in an individual.",
    "Multiple sclerosis (age of onset)": "The age at which a patient first shows symptoms of MS.",
    "Neuromyelitis optica": "An autoimmune condition that causes inflammation in the spinal cord and optic nerves, often confused with MS.",
    "Chronic lymphocytic leukemia or multiple sclerosis": "Genetic overlap or comorbidity between MS and chronic lymphocytic leukemia, a blood cancer.",
    "Neuromyelitis optica (AQP4-IgG-positive)": "A form of neuromyelitis optica associated with AQP4 antibodies, indicating more severe immune activity.",
    "Marginal zone lymphoma or multiple sclerosis": "Represents shared risk or pleiotropy between MS and a type of slow-growing non-Hodgkin lymphoma.",
    "B-cell lymphoblastic leukemia or multiple sclerosis (pleiotropy)": "Suggests shared genetic factors affecting both MS and this leukemia subtype.",
    "Multiple sclerosis and HDL levels (pleiotropy)": "Genetic correlation between MS and high-density lipoprotein (HDL) cholesterol levels.",
    "Decreased low contrast letter acuity in multiple sclerosis": "Reduced ability to see contrast, often one of the early visual impairments in MS.",
    "Multiple sclerosis and triglyceride levels (pleiotropy)": "Pleiotropic genetic effects linking MS risk to triglyceride metabolism.",
    "Moderate or severe prolonged lymphopenia in dimethyl fumarate-treated relapsing-remitting multiple sclerosis": "A side effect involving reduced lymphocyte count in MS patients treated with dimethyl fumarate.",
    "Normalized brain volume": "A metric often used to monitor brain atrophy in MS patients over time.",
    "Binding antibody response to interferon beta therapy in multiple sclerosis (antibody levels; all treatment preparations)": "Measures immune response to interferon beta, a common MS therapy.",
    "Multiple sclerosis--Brain Glutamate Levels": "Tracks glutamate concentration in the brain, which may relate to neurotoxicity in MS.",
    "Diffuse large B-cell lymphoma or multiple sclerosis": "Indicates genetic or clinical associations between MS and this aggressive type of lymphoma."
}

st.sidebar.title("🔍 Navigation")
selected = st.sidebar.radio("Go to", ["Home", "Upload SNP Data & Predict", "Working Model Accuracies", "About the Project"])

# Navigation Logic
if selected == "Home":
    
# App Title
    st.title("🧬 GWAS Model Dashboard — Multiple Sclerosis Risk Prediction")


# Prediction Page (Default Home)
    #if selected == "Manual Input & Predict":
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
            st.markdown(f"🧠 **Trait Info**: {trait_info.get(trait, 'For Further clarification consult a medical expect')}")

            if beta < 1:
                beta_msg = "🟡 This SNP has a **neutral or low effect** on the associated trait."
            elif beta < 2:
                beta_msg = "🟠 This SNP has a **moderate risk** association with the trait."
            else:
                beta_msg = "🔴 This SNP has a **strong risk association** with the trait."

            st.info(f"📊 ODD RATIO of RISK : **{beta:.4f}**")
            st.caption(beta_msg)

            st.info(f"📈 P-value factor : **{poly_score:.2f}**")
            

        

        except Exception as e:
            st.error(f"🚨 Error: {e}")

elif selected == "Upload SNP Data & Predict":
    # About and format info
    with st.expander("📘 About the Prediction & Input Format", expanded=True):
        st.markdown("""
        This module predicts **potential disease traits** using your SNP data through ensemble machine learning models.

        #### 🧬 Expected Input Columns:

        | Column Name | Description |
        |-------------|-------------|
        | `SNPS` | SNP ID (e.g., `rs1234`). Unique identifier for the variant. |
        | `LOCATION` | Chromosomal location (e.g., `chr6`). |
        | `RISK_ALLELE` | The allele associated with disease risk (e.g., A, T, C, G). |
        | `MAPPED_GENE` | Gene associated with the SNP (e.g., `HLA-DRB1`). |
        | `RISK ALLELE FREQUENCY` | Frequency of the risk allele in the population (0.0–1.0). |

        #### 🔮 Output Includes:
        - 🧠 Predicted Trait
        - 📊 OR/BETA Value
        - 🧪 Polygenic Risk Score

        🧠 **Note:** Your CSV file **must contain** these exact column headers.
        """)

        # Sample CSV
        sample_df = pd.DataFrame({
            "SNPS": ["rs1234", "rs2345"],
            "LOCATION": ["chr6", "chr1"],
            "RISK_ALLELE": ["A", "T"],
            "MAPPED_GENE": ["HLA-DRB1", "APOE"],
            "RISK ALLELE FREQUENCY": [0.12, 0.45]
        })
        sample_csv = sample_df.to_csv(index=False)
        st.download_button("📄 Download Sample CSV Template", data=sample_csv, file_name="sample_snp.csv", mime="text/csv")
    st.subheader("📤 Upload Your SNP File (.csv)")

    # Upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if not uploaded_file:
        st.info("👆 Please upload a valid SNP file to continue.")
        st.stop()

    try:
        # Read File
        user_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # ✅ 📍 INSERT THIS: SNP Distribution by Chromosome Plot
        st.markdown("### 🧬 SNP Distribution by Chromosome")
        user_df['CHROM'] = user_df['LOCATION'].astype(str).str.replace('chr', '', case=False)
        user_df = user_df[user_df['CHROM'].isin([str(i) for i in range(1, 23)] + ['X', 'Y'])]

        chrom_counts = user_df['CHROM'].value_counts().reset_index()
        chrom_counts.columns = ['Chromosome', 'SNP Count']
        chrom_counts = chrom_counts.sort_values(
            by='Chromosome', 
            key=lambda x: x.map(lambda c: int(c) if c.isdigit() else 23 if c == 'X' else 24)
        )

        fig_chrom = px.bar(
            chrom_counts, x='Chromosome', y='SNP Count',
            color='Chromosome', title='SNPs Mapped Per Chromosome'
        )
        st.plotly_chart(fig_chrom, use_container_width=True)
        
        # 🔍 Preview
        with st.expander("🧾 Preview Uploaded Data"):
            st.dataframe(user_df.head(), use_container_width=True)

        # ✅ Column Check
        expected_cols = ['SNPS', 'LOCATION', 'RISK_ALLELE', 'MAPPED_GENE', 'RISK ALLELE FREQUENCY']
        missing_cols = [col for col in expected_cols if col not in user_df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            st.stop()

        # 🔄 Encode Function
        def safe_encode(col, le):
            return col.apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        for col in ['SNPS', 'LOCATION', 'RISK_ALLELE', 'MAPPED_GENE']:
            le = label_encoders.get(col)
            if le:
                user_df[col] = safe_encode(user_df[col].astype(str), le)
            else:
                st.warning(f"⚠️ No encoder found for {col}")

        # 🎯 Predict
        X_input = user_df[expected_cols]
        clf_pred = clf_models["Voting Classifier"].predict(X_input)
        decoded_traits = label_encoders['DISEASE/TRAIT'].inverse_transform(clf_pred)
        beta_pred = reg_models_or["Stacking"].predict(X_input)
        poly_pred = reg_models_poly["Gradient Boosting"].predict(X_input)

        # 🧠 Enrich Results
        user_df['Predicted Trait'] = decoded_traits
        user_df['Pred OR/BETA'] = beta_pred
        user_df['Pred Polygenic Score'] = poly_pred

        # 📊 Final Output
        st.markdown("### 🧬 SNP Prediction Results")
        st.dataframe(user_df, use_container_width=True)

        # 📥 Download
        st.download_button("📥 Download Results", data=user_df.to_csv(index=False), file_name="predictions.csv")

        # 🔢 Trait Frequency
        st.markdown("### 🧠 Trait Distribution")
        trait_counts = user_df['Predicted Trait'].value_counts().reset_index()
        trait_counts.columns = ['Trait', 'Count']
        st.plotly_chart(px.bar(trait_counts, x='Trait', y='Count', color='Trait'), use_container_width=True)
        st.plotly_chart(px.pie(trait_counts, names='Trait', values='Count'), use_container_width=True)

        # 📈 OR/BETA Spread
        st.markdown("### 📈 OR/BETA Spread")
        fig1, ax1 = plt.subplots()
        sns.boxplot(data=user_df, x='Predicted Trait', y='Pred OR/BETA', palette='coolwarm', ax=ax1)
        st.pyplot(fig1)

        # 🔥 Correlation Heatmap
        st.markdown("### 🔥 RISK SCORE vs OR/BETA Correlation")
        corr_df = user_df[['Pred OR/BETA', 'Pred RISK SCORE Score']]
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr_df.corr(), annot=True, cmap='vlag', ax=ax3)
        st.pyplot(fig3)

        # 🧬 Trait Summaries
        st.markdown("### 📊 Trait-wise Summary")
        for trait in user_df['Predicted Trait'].unique():
            sub_df = user_df[user_df['Predicted Trait'] == trait]
            avg_beta = sub_df['Pred OR/BETA'].mean()
            avg_poly = sub_df['Pred Polygenic Score'].mean()
            with st.expander(f"🧠 {trait}"):
                st.write(f"**OR/BETA Avg:** `{avg_beta:.3f}`")
                st.write(f"**Polygenic Score Avg:** `{avg_poly:.2f}`")
                st.info(trait_info.get(trait, "🧪 No additional info available."))

    except Exception as e:
        st.error(f"🚨 Something went wrong: {e}")


# Accuracy Page
elif selected == "Working Model Accuracies":
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
elif selected == "About the Project":
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
