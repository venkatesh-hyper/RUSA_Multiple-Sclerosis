# 🧬 Genetic Variation Analysis for Predicting Multiple Sclerosis

A Machine Learning-powered Streamlit web app that predicts the likelihood of Multiple Sclerosis (MS) from SNP (Single Nucleotide Polymorphism) genetic data. Upload your dataset, view risk insights, and download predictions with polygenic scores and odds ratios.

> Built by venkatesh P as a bioinformatics solution for genome-based MS risk evaluation.
> Utilizes Python, Streamlit, and scikit-learn libraries for efficient data analysis and visualization
> RUSA UNIVERSITY PROJECT -- UNIVERSITY OF MADRAS

---

## 🚀 Features

- 📁 Upload SNP datasets (.csv or .xlsx)
- ⚙️ Predict disease traits, Odds Ratios, and Polygenic Scores
- 📊 Visualize risk allele distribution, odds ratios, and gene impact
- 🧬 Gene-level risk impact analysis
- ⏱ Real-time interactive predictions
- 💾 Download full prediction report as CSV

---

## 🔬 Input File Format

| Column Name     | Description                                      |
|-----------------|--------------------------------------------------|
| `SNP`           | SNP ID (e.g., rs123456)                          |
| `Location`      | Genomic location                                 |
| `RiskAllele`    | Allele increasing risk                           |
| `RiskFrequency` | Frequency of the risk allele                     |
| `orValue`       | Odds Ratio (OR) value                            |
| `pValue`        | Significance level                               |
| `traitName`     | Target trait (e.g., Multiple Sclerosis)          |

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Seaborn, Plotly, Matplotlib
- **Deployment**: Docker,Streamlit

---

## 📦 Installation Guide

### ✅ Local Run

```bash
# Clone repo
git clone https://github.com/yourusername/ms-genetic-analysis.git
cd ms-genetic-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Launch app
streamlit run app.py

```bash

## 🤖 Machine Learning Models Used

This project utilizes a hybrid ML architecture tailored for genomic prediction:

### 🧠 Main Model
- **XGBoost Classifier**
  - Handles non-linear relationships in genetic markers
  - Great for high-dimensional SNP data
  - Used for predicting Disease/Trait presence and severity

### 🌲 Backup Model
- **Random Forest Classifier**
  - Used as fallback for binary classification
  - Helps improve ensemble stability

### 📈 Outputs Generated
- **Predicted Disease/Trait** (e.g., Multiple Sclerosis)
- **Odds Ratio / BETA**: Risk influence of a SNP
- **Polygenic Risk Score (PRS)**: Cumulative genetic risk estimate
- **Risk Allele Frequency Visuals**

All models were trained using curated SNP datasets with key columns like `riskAllele`, `riskFrequency`, `orValue`, `traitName`, and `pValue`.


## 👨‍💻 Author

**Venkatesh P**  
Machine Learning Engineer | Data Science Enthusiast | Bioinformatics Explorer  
🧬 Passionate about bridging biology and AI for healthtech innovation.  
📫 Email: venkateshpvnky9@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/venkatesh-ml/)  

## 🙏 Special Thanks

I would like to express my sincere gratitude to:

- **Dr. Lavanya, University of Madras, Department of Computer Science**,  
  for her invaluable guidance, encouragement, and expert insights throughout the project.

- **RUSA (Rashtriya Uchchatar Shiksha Abhiyan), University of Madras**,  
  for providing the research infrastructure, funding, and academic support that made this work possible.

This project wouldn't have reached its current state without their support and mentorship.
I'm grateful for their trust in me and my abilities.