🏡 House Price Prediction – End-to-End ML Project

This project is a complete Machine Learning pipeline that predicts house prices using various house features (numerical + categorical).
It includes preprocessing, feature engineering, model training, evaluation, and a Streamlit web application for real-time predictions.

📌 Features of This Project

✔ Automatic data cleaning (remove missing values & duplicates)

✔ Identify numeric + categorical columns automatically
# 🏡 House Price Prediction – End-to-End ML Project

This project is a complete Machine Learning pipeline that predicts house prices using various house features (numerical + categorical).  
It includes preprocessing, feature engineering, model training, evaluation, and a Streamlit web application for real-time predictions.

---

## 📌 Features of This Project

✔ Automatic data cleaning (remove missing values & duplicates)  
✔ Identify numeric + categorical columns automatically  
✔ Apply Standard Scaling & One-Hot Encoding using ColumnTransformer  
✔ Train a Linear Regression model  
✔ Save trained model using pickle  
✔ User-friendly Streamlit app for price prediction  
✔ Dynamic UI based on dataset columns  
✔ Works with any dataset having a `price` column  

---

## 🧰 Tech Stack

| Component | Tools |
|----------|--------|
| Language | Python |
| ML | scikit-learn |
| Data | Pandas, Numpy |
| Deployment | Streamlit |
| Model Storage | Pickle |

---

## 📂 Dataset Requirements

Your dataset must include:

- A column named **`price`** → Target variable  
- Any number of numeric or categorical input columns  
- CSV format (e.g., `data.csv`)  

### Example:

| area | bedrooms | bathrooms | stories | parking | furnishingstatus | price |
|------|----------|-----------|---------|----------|-------------------|--------|
| 1800 | 3 | 2 | 2 | 1 | furnished | 12000000 |

You can modify column names — the code automatically detects numeric vs categorical.

---

## 🛠 Project Structure

✔ Apply Standard Scaling & One-Hot Encoding using ColumnTransformer

✔ Train a Linear Regression model

✔ Save trained model using pickle

✔ User-friendly Streamlit app for price prediction

✔ Dynamic UI based on dataset columns

✔ Works with any dataset having a price column

🧰 Tech Stack
Component	Tools
Language	Python
ML	scikit-learn
Data	Pandas, Numpy
Deployment	Streamlit
Model Storage	Pickle
📂 Dataset Requirements

Your dataset must include:

A column named price → Target variable

Any number of numeric or categorical input columns

CSV format (e.g., data.csv)

Example:

area	bedrooms	bathrooms	stories	parking	furnishingstatus	price
1800	3	2	2	1	furnished	12000000

You can modify the column names — the code automatically detects numeric vs categorical.

🛠 Project Structure
House-price-prediction/
│── train.py
│── app.py
│── data.csv
│── model.pkl
│── requirements.txt
│── README.md

🚀 How the Project Works
1️⃣ train.py (Model Training)

This script:

Loads data.csv

Removes duplicates + missing values

Separates input features (X) and target feature (price)

Detects categorical & numerical columns automatically

Applies:

StandardScaler → For numeric columns

OneHotEncoder → For categorical columns

Creates a Pipeline + Linear Regression Model

Saves the trained model to model.pkl

Run training:

python train.py

2️⃣ app.py (Streamlit Prediction App)

This script:

Loads the trained model.pkl

Loads dataset structure for dynamic UI

Creates sliders/input fields for numeric columns

Creates dropdowns for categorical columns

Takes user input

Predicts price using the ML model

Displays result on UI

Run Streamlit app:

streamlit run app.py

🧪 Model Pipeline Overview
ColumnTransformer(
    - scale numeric columns using StandardScaler
    - encode categorical columns using OneHotEncoder
) 
→ Linear Regression Model


This ensures correct preprocessing during both training & prediction.

📊 Example Prediction Output

When clicking Predict Price, the app shows something like:

🏠 Predicted House Price: ₹ 12,45,678.55

📦 Installation & Setup
Step 1: Install required packages
pip install -r requirements.txt

Step 2: Train the model
python train.py

Step 3: Run the Streamlit app
streamlit run app.py

📘 requirements.txt
streamlit
pandas
numpy
scikit-learn
pickle-mixin

🔮 Future Improvements

Add Random Forest / XGBoost models

Add hyperparameter tuning

Add charts/EDA in Streamlit

Deploy online (Render, Streamlit Cloud, AWS)

✨ Author

Rohit Nimbalkar
ML & AI Developer
(You can add GitHub / LinkedIn here)
